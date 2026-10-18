"""Triage Relay app server. One page (app/static/index.html), seven screens, a JSON API, state in app/state/*.json.

    python server.py [port]        (default 5098, bound to 127.0.0.1)

Login is DEMO-GRADE: a role + a PIN shows the permission idea; it is not hospital security.
Roles: reception, nurse, ward (one department), reviewer, admin. The kiosk needs no login.

PUBLIC DEMO (TRIAGE_PUBLIC=1, for a link on the open internet): every visitor gets their OWN small hospital, kept in memory
only (nothing written to disk), started from the demo patients in demo_seed/ and dropped after LIMITS["hospital_idle_s"]
without use. One visitor's patients, corrections and Setup are never seen by another. The doors that run the models are
rate-limited per visitor (client_ip.py), and request bodies and patient words are capped.
"""
import collections
import concurrent.futures
import copy
import http.server
import json
import os
import re
import secrets
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.environ.get("TRIAGE_PUBLIC") == "1"
STATE = os.environ.get("TRIAGE_STATE") or os.path.join(HERE, "state")
if not PUBLIC:
    os.makedirs(STATE, exist_ok=True)
sys.path.insert(0, HERE)
LOCK = threading.Lock()
# torch keeps working memory for every thread that runs it (measured 15 Oct: about 1 MB per new thread, never given back),
# and the HTTP server starts a thread per request. So each request's work runs on ONE long-lived worker thread. Requests were
# already handled one at a time (LOCK), so nothing changes for a visitor. deploy_box/measure/leak_hunt.py shows the growth.
WORKER = concurrent.futures.ThreadPoolExecutor(max_workers=1)
SESSIONS = {}          # token -> {role, ward, hospital}

# ---- public demo: protections, stated once ---------------------------------------------------------------------------------
LIMITS = {"hospital_idle_s": 2 * 3600,               # a visitor's hospital is dropped after 2 hours without use
          "hospitals": 300,                          # at most this many held at once (oldest dropped first): bounds memory
          "hospitals_per_visitor_hour": 8,           # new hospitals one address may start per hour (then it reuses its newest)
          "model_runs_per_visitor_10min": 40,        # kiosk/nurse steps that run the models (a patient is about 3 to 4)
          "body_bytes": 32 * 1024,                   # largest request body accepted
          "words_chars": 1000}                       # patient words read by the translator
MODEL_DOORS = {"/api/intake/start", "/api/intake/answer", "/api/intake/confirm", "/api/case/refine"}
COOKIE = "triage_relay_demo"
HOSPITALS = {}                                       # id -> {"files": {name: obj}, "seen": time, "ip": address}
STARTED = collections.defaultdict(collections.deque)     # address -> times it started a hospital
RUNS = collections.defaultdict(collections.deque)        # address -> times it ran the models
CURRENT = threading.local()                          # the hospital this request thread is serving
SEED_DIR = os.path.join(HERE, "demo_seed")
SEED = ({n[:-5]: json.load(open(os.path.join(SEED_DIR, n), encoding="utf-8")) for n in os.listdir(SEED_DIR) if n.endswith(".json")}
        if os.path.isdir(SEED_DIR) else {})

DEPARTMENTS = ["Cardiology", "ENT", "Emergency (resus)", "Gastroenterology", "General medicine", "Haematology", "Infectious diseases",
               "Mental health", "Neurology", "Oncology", "Paediatrics", "Respiratory", "Rheumatology", "Trauma and orthopaedics"]
# questions never put on the waiting-room screen: the system hands them to the triage nurse to ask in person.
# A STARTING list (sexual health, drugs, alcohol, pregnancy, cancer, mental health); each hospital edits it in Setup.
NURSE_ONLY = ["E_2", "E_189", "E_115", "E_27", "E_61", "E_62", "E_78", "E_167", "E_34", "E_37", "E_80", "E_111", "E_15"]
DEFAULT_CONFIG = {"departments": {d: {"available": True, "if_unavailable": {"action": "redirect", "to": "Emergency (resus)"}} for d in DEPARTMENTS},
                  "custom_questions": [], "max_rounds": 2, "nurse_only": NURSE_ONLY, "red_flags": True,
                  "injury_route": "Trauma and orthopaedics"}

# An injury or accident is outside what the models were trained on (DDXPlus has no cuts, bleeding or broken bones), so the
# kiosk asks, and an injury is never read by the AI: it goes to the hospital's injury route for a person to assess.
INJURY_NOTE = "Injury or accident: outside what this system was trained on. Not read by the AI; staff to triage in person."
# Nothing medical recognised in what the patient wrote (a joke, a refusal, an attempt to give the system instructions): with no
# evidence there is no route to give, so none is invented. The patient goes to the triage nurse to be talked to in person.
NOTHING_NOTE = "Nothing medical recognised in what they wrote: no route given. Staff to talk to them in person."
NURSE_QUEUE = "Triage nurse"


def nothing_decision(config):
    return dict(injury_decision(config), department=NURSE_QUEUE, likely_department=NURSE_QUEUE, unavailable_note=None,
                outside_scope=NOTHING_NOTE)


def available_route(dept, config):
    """the hospital's setup: a department it does not have (or that is closed today) redirects or transfers, as configured"""
    cfgd = config["departments"].get(dept, {"available": True})
    if cfgd.get("available", True):
        return dept, None
    act = cfgd.get("if_unavailable", {"action": "redirect", "to": "Emergency (resus)"})
    note = f"{dept} unavailable here: " + (f"redirected to {act['to']}" if act["action"] == "redirect"
                                            else f"transfer out ({act.get('to') or 'arrange transfer'})")
    return (act["to"] if act["action"] == "redirect" else f"TRANSFER: {act.get('to') or 'external'}"), note


def injury_decision(config):
    dept, note = available_route(config.get("injury_route") or DEFAULT_CONFIG["injury_route"], config)
    return {"department": dept, "urgency": "Nurse to assess", "likely_department": dept, "unavailable_note": note,
            "rules_applied": [], "danger": None, "in_play": [], "runners_up": [], "heads_up": [], "second_reader": None,
            "novelty": None, "facts": {}, "verdict": "no idea", "lean": 0.0, "ask_next": [], "ask_codes": [],
            "outside_scope": INJURY_NOTE}
DEFAULT_USERS = {"reception": "1111", "nurse": "2222", "ward": "3333", "reviewer": "4444", "admin": "9999"}
CAN_SEE = {"reception": {"reception"}, "nurse": {"reception", "nurse"}, "ward": {"ward"}, "reviewer": {"review", "nurse"},
           "admin": {"reception", "nurse", "ward", "review", "setup"},
           "demo": {"reception", "nurse", "ward", "review", "setup"}}      # DEMO MODE: every screen, no PIN, labelled as demo


def load(name, default):
    if PUBLIC:      # this visitor's hospital, in memory; a copy, so a change only lands when it is saved (as with files)
        files = CURRENT.hospital["files"]
        files.setdefault(name, copy.deepcopy(default))
        return copy.deepcopy(files[name])
    p = os.path.join(STATE, name)
    if not os.path.exists(p):
        json.dump(default, open(p, "w", encoding="utf-8"), indent=1)
    return json.load(open(p, encoding="utf-8"))


def save(name, obj):
    if PUBLIC:
        CURRENT.hospital["files"][name] = copy.deepcopy(obj)
        return
    json.dump(obj, open(os.path.join(STATE, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def recent(times, window):
    """drop the times older than the window; what is left is the count inside it"""
    cut = time.time() - window
    while times and times[0] < cut:
        times.popleft()
    return times


def new_hospital(ip):
    """a fresh demo hospital from demo_seed/ (the seeded patients arrive 'now')"""
    files = copy.deepcopy({k + ".json": v for k, v in SEED.items()})
    for c in files.get("cases.json", []):
        c["arrived"] = c["finished"] = now()
    hid = secrets.token_hex(16)
    HOSPITALS[hid] = {"files": files, "seen": time.time(), "ip": ip}
    STARTED[ip].append(time.time())
    idle = [h for h, x in HOSPITALS.items() if time.time() - x["seen"] > LIMITS["hospital_idle_s"]]
    oldest = sorted(HOSPITALS, key=lambda h: HOSPITALS[h]["seen"])[:max(0, len(HOSPITALS) - LIMITS["hospitals"])]
    for h in set(idle) | set(oldest):
        HOSPITALS.pop(h, None)
    for tok in [t for t, s in SESSIONS.items() if s.get("hospital") not in HOSPITALS]:
        SESSIONS.pop(tok)
    return hid


print("loading the relay models...", flush=True)
from engine import Engine      # noqa: E402
import news2                   # noqa: E402
from redflags import RED_FLAGS, BY_ID   # noqa: E402
ENGINE = Engine(device=os.environ.get("TRIAGE_DEVICE") or None)   # TRIAGE_DEVICE=cpu: the tiny models run fine without a GPU
print("ready", flush=True)


def kiosk_questions(case, cfg):
    """the next questions for the kiosk; any the hospital keeps for the nurse are set aside on the case for the nurse instead"""
    private = set(cfg.get("nurse_only", NURSE_ONLY))
    case.setdefault("for_nurse", [])
    for _ in range(5):
        qs = ENGINE.next_questions(case["yes"], case["no"], skip=case["asked"] + case["for_nurse"])
        held = [q for q in qs if q in private]
        if not held:
            break
        case["for_nurse"] += held
    case["asked"] += qs
    return qs


def now():
    return time.strftime("%H:%M")


def approved_rules():
    return [x for x in load("corrections.json", []) if x["status"] == "approved as rule"]


def similar_corrections(case, corr):
    """notes for the nurse: past corrections on patients like this one. Shown when this patient gave MOST (more than half) of
    the answers the corrected patient gave: a plain line a nurse can check on the card. Notes only: they never change a route."""
    mine = set(case.get("yes", []))
    out = []
    for x in corr:
        if x["case"] == case["id"] or x["status"] in ("dismissed", "revoked"):
            continue
        theirs = set(x.get("fields", []))
        shared = len(mine & theirs)
        if theirs and shared * 2 > len(theirs):
            out.append({"id": x["id"], "from": x["from"], "to": x["to"], "why": x.get("why", ""), "status": x["status"],
                        "shared": shared, "of": len(theirs)})
    return sorted(out, key=lambda o: -o["shared"] / o["of"])[:2]


def finish(case, config):
    custom_yes = [q for q in config["custom_questions"] if case["custom_answers"].get(q["id"]) == "yes"]
    d = (injury_decision(config) if case.get("injury") else nothing_decision(config) if case.get("nothing_read") else
         ENGINE.decide(case["text"], case["yes"], case["no"], case["conf"], config, custom_yes, rules=approved_rules()))
    flags = [BY_ID[f] for f in case.get("red_flags", []) if f in BY_ID]
    if flags:       # NHS red flag ticked at the kiosk: staff look now, and the FLAG picks the department (published guidance)
        routes = {f["route"] for f in flags}
        dept, note = available_route(routes.pop() if len(routes) == 1 else "Emergency (resus)", config)
        d.update(red_flags=[f"{f['from']}: {f['text']}" for f in flags], urgency="Immediate", model_department=d["department"],
                 department=dept, unavailable_note=note or d.get("unavailable_note"))
    case.update(status="waiting", decision=d, finished=now(),
                custom_yes=[q["text"] for q in custom_yes],
                said=[ENGINE.plain(f) for f in case["yes"]], said_no=[ENGINE.plain(f) for f in case["no"]],
                # the nurse card keeps these apart: what was read from the patient's own words, and their answers to the
                # system's yes/no questions (an answered "no" is not something the patient said)
                said_words=[ENGINE.plain(f) for f in case["yes"] if f in case.get("read", [])],
                said_yes_q=[ENGINE.plain(f) for f in case["yes"] if f not in case.get("read", [])],
                for_nurse_q=[{"code": q, "text": ENGINE.question_text(q)} for q in case.get("for_nurse", [])
                             if q not in case["yes"] and q not in case["no"]])
    return case


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Robots-Tag", "noindex, nofollow")      # a demo, never a search result
        self.send_header("X-Content-Type-Options", "nosniff")
        if getattr(self, "_cookie", None):
            # scoped to the folder the demo is served under (the proxy says which, X-Forwarded-Prefix), not the whole site
            prefix = self.headers.get("X-Forwarded-Prefix", "")
            path = prefix if re.fullmatch(r"/[A-Za-z0-9/_-]{1,80}", prefix) else "/"
            self.send_header("Set-Cookie", f"{COOKIE}={self._cookie}; Path={path}; HttpOnly; Secure; SameSite=Lax; "
                                           f"Max-Age={LIMITS['hospital_idle_s']}")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)

    def log_message(self, *a):
        pass

    def _ip(self):
        import client_ip
        return client_ip.resolve(self.client_address[0], self.headers.get("X-Forwarded-For", ""),
                                 self.headers.get("CF-Connecting-IP", ""))

    def _hospital(self):
        """PUBLIC: the visitor's own hospital (from their cookie), or a new one. Call inside LOCK."""
        ip = self._ip()
        jar = dict(p.strip().split("=", 1) for p in self.headers.get("Cookie", "").split(";") if "=" in p)
        hid = jar.get(COOKIE)
        if hid not in HOSPITALS:
            mine = [h for h, x in HOSPITALS.items() if x["ip"] == ip]
            if len(recent(STARTED[ip], 3600)) >= LIMITS["hospitals_per_visitor_hour"] and mine:
                hid = max(mine, key=lambda h: HOSPITALS[h]["seen"])      # a client that drops cookies keeps one hospital
            else:
                hid = new_hospital(ip)
            self._cookie = hid
        HOSPITALS[hid]["seen"] = time.time()
        CURRENT.hospital = HOSPITALS[hid]
        CURRENT.hid = hid
        return hid

    def _too_many(self, path):
        """PUBLIC: True (and a 429 sent) when this visitor has run the models too often in the last 10 minutes"""
        if not PUBLIC or path not in MODEL_DOORS:
            return False
        runs = recent(RUNS[self._ip()], 600)
        if len(runs) >= LIMITS["model_runs_per_visitor_10min"]:
            self._json({"error": "This demo is resting for you for a few minutes. Please try again shortly."}, 429)
            return True
        runs.append(time.time())
        return False

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _who(self):
        s = SESSIONS.get(self.headers.get("X-Token", ""))
        return s if s and (not PUBLIC or s.get("hospital") == CURRENT.hid) else None

    def _need(self, screen):
        who = self._who()
        return who if who and screen in CAN_SEE[who["role"]] else None

    # files (the page, fonts) are sent from the request's own thread; the API, which touches state and the models, runs on
    # WORKER. A slow connection reading or writing never holds up anyone else.
    def do_GET(self):
        if urllib.parse.urlparse(self.path).path.startswith("/api/"):
            return WORKER.submit(self._get).result()
        return super().do_GET()

    def do_POST(self):
        size = int(self.headers.get("Content-Length") or 0)
        if size > LIMITS["body_bytes"]:
            return self._json({"error": "That is more than this form takes."}, 413)
        try:
            body = json.loads(self.rfile.read(size) or b"{}")
        except ValueError:
            return self._json({"error": "not understood"}, 400)
        return WORKER.submit(self._do_post, urllib.parse.urlparse(self.path).path, body).result()

    def _get(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        with LOCK:
            if PUBLIC and u.path.startswith("/api/"):
                self._hospital()
            if u.path == "/api/me":
                return self._json({k: v for k, v in (self._who() or {}).items() if k != "hospital"})
            if u.path == "/api/departments":
                return self._json(DEPARTMENTS)
            if u.path == "/api/questions":          # every yes/no question the system can ask, for Setup
                return self._json([{"code": c, "text": ENGINE.question_text(c)} for c in sorted(ENGINE.binary, key=lambda c: int(c[2:]))])
            if u.path == "/api/kiosk_config":
                cfg = load("config.json", DEFAULT_CONFIG)
                return self._json({"custom_questions": cfg["custom_questions"], "red_flags": RED_FLAGS if cfg.get("red_flags", True) else []})
            if u.path == "/api/cases":
                screen = q.get("screen", [""])[0]
                who = self._need(screen)
                if not who:
                    return self._json({"error": "not allowed"}, 403)
                cases = [c for c in load("cases.json", []) if c["status"] != "asking"]
                if screen == "ward":
                    ward = who.get("ward") or q.get("ward", [""])[0]
                    cases = [c for c in cases if c["decision"]["department"] == ward or ward in c["decision"]["heads_up"]]
                if screen == "nurse":
                    corr = load("corrections.json", [])
                    for c in cases:
                        c["similar_corrections"] = similar_corrections(c, corr)
                return self._json(sorted(cases, key=lambda c: -c["id"]))
            if u.path == "/api/corrections":
                if not self._need("review"):
                    return self._json({"error": "not allowed"}, 403)
                return self._json(sorted(load("corrections.json", []), key=lambda c: -c["id"]))
            if u.path == "/api/config":
                if not self._need("setup"):
                    return self._json({"error": "not allowed"}, 403)
                return self._json({"nurse_only": NURSE_ONLY, **load("config.json", DEFAULT_CONFIG)})
        return self._json({"error": "unknown"}, 404)

    def _do_post(self, path, body):
        with LOCK:
            if PUBLIC:
                self._hospital()
            if self._too_many(path):
                return
            try:
                return self._post(path, body if isinstance(body, dict) else {})
            except (StopIteration, KeyError, TypeError, ValueError):     # an unknown case id or a malformed field
                return self._json({"error": "not understood"}, 400)

    def _post(self, path, b):
        if path == "/api/login":
            users = load("users.json", DEFAULT_USERS)
            if b.get("role") != "demo" and users.get(b.get("role")) != str(b.get("pin", "")):
                return self._json({"error": "That PIN does not match this role."}, 403)
            tok = secrets.token_hex(12)
            SESSIONS[tok] = {"role": b["role"], "ward": b.get("ward") if b["role"] == "ward" else None,
                             "hospital": getattr(CURRENT, "hid", None) if PUBLIC else None}
            return self._json({"token": tok, "role": b["role"], "ward": SESSIONS[tok]["ward"]})
        if path == "/api/logout":
            SESSIONS.pop(self.headers.get("X-Token", ""), None)
            return self._json({"ok": True})

        # ---- the kiosk -----------------------------------------------------------------------------------------------
        if path == "/api/intake/start":
            cases, cfg = load("cases.json", []), load("config.json", DEFAULT_CONFIG)
            injury = bool(b.get("injury"))
            if injury:          # never read by the AI: outside what it was trained on
                text, fields, conf = (b.get("words") or "").strip(), [], {}
            else:
                text, fields, conf = ENGINE.read_words(b.get("age", ""), b.get("sex", ""),
                                                       str(b.get("words") or "")[:LIMITS["words_chars"]])
            case = {"id": max([c["id"] for c in cases] or [0]) + 1, "name": (b.get("name") or "").strip()[:60] or "Walk-in",
                    "age": b.get("age"), "sex": b.get("sex"), "words": b.get("words", "")[:1000], "text": text,
                    "arrived": now(), "status": "asking", "yes": fields, "no": [], "conf": conf, "rounds": 0,
                    "custom_answers": {}, "asked": [], "red_flags": [f for f in (b.get("red_flags") or []) if f in BY_ID],
                    "read": list(fields), "injury": injury, "areas": [str(a)[:40] for a in (b.get("areas") or [])][:10],
                    "nothing_read": (not injury) and not fields and not b.get("red_flags")}   # a red flag IS something
            if case["red_flags"] or injury or case["nothing_read"]:       # a red flag: no more questions, straight onto every staff list
                finish(case, cfg)
                cases.append(case)
                save("cases.json", cases)
                return self._json({"id": case["id"], "done": True, "red_flag": bool(case["red_flags"]), "injury": injury,
                                   "nothing_read": case["nothing_read"]})
            qs = kiosk_questions(case, cfg)
            cases.append(case)
            save("cases.json", cases)
            return self._json({"id": case["id"], "read": [ENGINE.plain(f) for f in fields],
                               "questions": [{"code": c, "text": ENGINE.question_text(c)} for c in qs]})
        if path == "/api/intake/answer":
            cases, cfg = load("cases.json", []), load("config.json", DEFAULT_CONFIG)
            case = next(c for c in cases if c["id"] == int(b["id"]))
            for code, ans in (b.get("answers") or {}).items():
                (case["yes"] if ans == "yes" else case["no"]).append(code) if ans in ("yes", "no") else None
            case["custom_answers"].update(b.get("custom") or {})
            case["rounds"] += 1
            done = case["rounds"] >= cfg.get("max_rounds", 2)
            if not done:
                probe = ENGINE.decide(case["text"], case["yes"], case["no"], case["conf"], cfg)
                done = probe["verdict"] == "yes"
            if done:
                finish(case, cfg)
                case["patient_check"] = {"state": "not yet checked"}
                save("cases.json", cases)
                # the patient sees what was understood, in plain words, and confirms it (no department, no condition names)
                return self._json({"done": True, "id": case["id"],
                                   "understood": [{"code": f, "text": ENGINE.plain(f)} for f in case["yes"]]})
            qs = kiosk_questions(case, cfg)
            save("cases.json", cases)
            return self._json({"done": False, "questions": [{"code": c, "text": ENGINE.question_text(c)} for c in qs]})

        if path == "/api/intake/confirm":
            # the patient is the authority on what they said: anything marked "not right" is taken out and the route is
            # worked out again; anything they add goes to the nurse in their own words
            cases, cfg = load("cases.json", []), load("config.json", DEFAULT_CONFIG)
            case = next(c for c in cases if c["id"] == int(b["id"]))
            wrong = [f for f in (b.get("wrong") or []) if f in case["yes"]]
            added = (b.get("added") or "").strip()[:500]
            if wrong:
                case["yes"] = [f for f in case["yes"] if f not in wrong]
                corrected = case.get("corrected_to")
                finish(case, cfg)
                if corrected:
                    case["decision"]["department"] = corrected
            case["patient_check"] = {"state": "changed" if (wrong or added) else "confirmed", "time": now(),
                                     "removed": [ENGINE.plain(f) for f in wrong], "added": added}
            case.setdefault("log", []).append(f"{now()} patient checked the summary: {case['patient_check']['state']}")
            save("cases.json", cases)
            return self._json({"ok": True})

        # ---- staff -----------------------------------------------------------------------------------------------------
        if path == "/api/case/status":
            who = self._need("reception") or self._need("nurse") or self._need("ward")
            if not who:
                return self._json({"error": "not allowed"}, 403)
            cases = load("cases.json", [])
            c = next(c for c in cases if c["id"] == int(b["id"]))
            c["status"] = b["status"]
            c.setdefault("log", []).append(f"{now()} {who['role']}: {b['status']}")
            save("cases.json", cases)
            return self._json(c)
        if path == "/api/case/notes":
            who = self._need("nurse")
            if not who:
                return self._json({"error": "not allowed"}, 403)
            cases = load("cases.json", [])
            c = next(c for c in cases if c["id"] == int(b["id"]))
            c["vitals"] = {k: str(v)[:20] for k, v in (b.get("vitals") or {}).items() if str(v).strip()}
            c["notes"] = (b.get("notes") or "")[:2000]
            c["news2"] = news2.score(c["vitals"])
            c["qsofa"] = news2.qsofa(c["vitals"])     # Sepsis-3 quick check; shown to staff, not used for routing     # published NEWS2 chart; shown to staff, not used for routing
            c.setdefault("log", []).append(f"{now()} nurse saved observations and notes")
            save("cases.json", cases)
            return self._json(c)
        if path == "/api/case/refine":
            who = self._need("nurse")
            if not who:
                return self._json({"error": "not allowed"}, 403)
            cases, cfg = load("cases.json", []), load("config.json", DEFAULT_CONFIG)
            c = next(c for c in cases if c["id"] == int(b["id"]))
            for code, ans in (b.get("answers") or {}).items():
                if ans in ("yes", "no") and code not in c["yes"] and code not in c["no"]:
                    (c["yes"] if ans == "yes" else c["no"]).append(code)
            corrected = c.get("corrected_to")
            finish(c, cfg)
            if corrected:
                c["decision"]["department"] = corrected
            c.setdefault("log", []).append(f"{now()} nurse asked more questions; card updated")
            save("cases.json", cases)
            return self._json(c)
        if path == "/api/correct":
            who = self._need("nurse")
            if not who:
                return self._json({"error": "not allowed"}, 403)
            cases, corr = load("cases.json", []), load("corrections.json", [])
            c = next(c for c in cases if c["id"] == int(b["id"]))
            if b.get("department") == c["decision"]["department"]:
                return self._json({"error": "That is already where this patient is going."}, 400)
            item = {"id": max([x["id"] for x in corr] or [0]) + 1, "case": c["id"], "from": c["decision"]["department"],
                    "to": b["department"], "why": (b.get("why") or "")[:400], "by": who["role"], "when": now(),
                    "fields": c["yes"], "fields_no": c["no"], "said": c.get("said", []), "status": "proposed", "scope": "this case"}
            corr.append(item)
            c["decision"]["department"] = b["department"]
            c["corrected_to"] = b["department"]
            c.setdefault("log", []).append(f"{now()} nurse corrected to {b['department']}: {item['why']}")
            save("corrections.json", corr)
            save("cases.json", cases)
            return self._json(item)
        if path == "/api/review":
            if not self._need("review"):
                return self._json({"error": "not allowed"}, 403)
            corr = load("corrections.json", [])
            x = next(c for c in corr if c["id"] == int(b["id"]))
            if b["decision"] == "revoked" and x["status"] != "approved as rule":
                return self._json({"error": "Only an approved rule can be revoked."}, 400)
            if b["decision"] == "approved as rule":
                rule = ENGINE.rule_from_correction(x["fields"], x.get("fields_no", []), x["to"])
                if not rule:
                    return self._json({"error": f"None of this patient's answers points to {x['to']} more than to the other "
                                                f"department, so it cannot become a rule for everyone. Keep it for this case."}, 400)
                x.update(rule)
            scope = {"approved as rule": "new patients with all the key answers", "revoked": "no longer applies"}
            x.update(status=b["decision"], review_why=(b.get("why") or "")[:400], reviewed=now(),
                     scope=scope.get(b["decision"], x["scope"]))
            save("corrections.json", corr)
            return self._json(x)
        if path == "/api/config":
            if not self._need("setup"):
                return self._json({"error": "not allowed"}, 403)
            save("config.json", b)
            return self._json(b)
        if path == "/api/demo_reset":
            if not self._need("setup"):
                return self._json({"error": "not allowed"}, 403)
            save("cases.json", [])
            save("corrections.json", [])
            return self._json({"ok": True})
        return self._json({"error": "unknown"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5098
    print(f"Triage Relay app on http://127.0.0.1:{port}" + (" (public demo: one hospital per visitor, in memory)" if PUBLIC else ""),
          flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

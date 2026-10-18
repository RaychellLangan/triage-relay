"""The app's engine: one patient from words + taps to a routed case. Wraps the relay built in src.

  words  -> translator writes fields
  taps   -> the system picks yes/no questions (worst-first rule-out, then what separates the top two), patient answers
  fields -> condition planks (identity, blind to severity) -> danger first (most severe condition still in play)
         -> department + urgency -> the hospital's setup (departments it has; what to do when one cannot take the patient)
         -> second reader + novelty + verdict seesaw -> who should see it, and how sure
Nothing here names a diagnosis to the patient; condition names appear only on clinical screens, as "in play".
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # the repo folder

import json
import math
import os
import re
import sys

import torch

R = ROOT + ""
for p in ("train", "data", "route_from_fields", "monitors", "probes"):
    sys.path.insert(0, f"{R}/src/{p}")
import common                                  # noqa: E402
import convert_ddxplus as conv                 # noqa: E402
from ward_card import Chain                    # noqa: E402
from build_messy_data import short_phrase      # noqa: E402
import verdict as V                            # noqa: E402
from wording import staff_line                 # noqa: E402

SEV_URGENCY = {1: "Immediate", 2: "Very urgent", 3: "Urgent", 4: "Standard", 5: "Non-urgent"}
LEVERS_OLD = f"{R}/runs/route_from_fields/verdict2/levers_banded.json"
LEVERS_NATURAL = f"{R}/runs/route_from_fields/verdict_natural/levers_banded.json"
# model sets: tag -> (levers, novelty bank). TRIAGE_MODELS=w picks the words-round models; otherwise n when present, else m
SETS = {"m": (LEVERS_OLD, "messy_train"), "n": (LEVERS_NATURAL, "natural_train"),
        "w": (f"{R}/runs/route_from_fields/verdict_words/levers_banded.json", "natural2_train")}


class Engine:
    def __init__(self, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # the natural-language models when they exist (trained on everyday phrasing), else the first messy-language ones
        nat = os.path.exists(f"{R}/runs/n_tiny_translator/model") and os.path.exists(f"{R}/runs/n_tiny_router/model")
        # w (words round) when trained: same routing as n on identical patients (.9745 vs .974) and better on real phrasing
        # (17/30, 9 of 9 auto-routes right, vs 16/30 and 6 of 7); see RESULTS_LOG 14 Oct
        words = os.path.exists(f"{R}/runs/w_tiny_translator/model") and os.path.exists(f"{R}/runs/w_tiny_router/model")
        tag = os.environ.get("TRIAGE_MODELS") or ("w" if words else "n" if nat else "m")
        from ward_card import CFG
        cfg = dict(CFG, translator=f"{R}/runs/{tag}_tiny_translator/model", router=f"{R}/runs/{tag}_tiny_router/model")
        self.models = tag
        self.chain = Chain(cfg=cfg, device=self.dev)
        self.table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
        self.sev = {c: self.table[c]["severity"] for c in self.table}
        counted = self.chain.counted
        self.cond_lean = counted["kinds"]["condition"]["lean"]
        self.cond_no = counted["kinds"]["condition"]["no_lean"]
        lev = SETS[tag][0] if os.path.exists(SETS[tag][0]) else LEVERS_OLD
        self.levers = json.load(open(lev, encoding="utf-8"))["levers"]
        V.FACTS = tuple(self.levers)
        from novelty_sensor import NoveltySensor
        self.sensor = NoveltySensor(f"{R}/runs/{tag}_tiny_router/model", device=self.dev,
                                    bank_file=f"{R}/data_built/{SETS[tag][1]}.picked.jsonl", pick_rate=1.0,   # pick_bank.py
                                    cache_file=f"{R}/runs/{tag}_tiny_router/novelty_cache.npz")
        self.evid = self.chain.evid
        self.binary = [f for f in self.cond_no if self.evid.get(f, {}).get("data_type") == "B"]

    # ---- reading ------------------------------------------------------------------------------------------------------
    def read_words(self, age, sex, words):
        # no age given: a placeholder keeps the input in its trained shape (the routers were measured to ignore age,
        # RESULTS_LOG "fingerprint shortcut"); staff screens show "age not given", never the placeholder
        text = f"{str(age).strip() or '40'}{(sex or '').lower()[:1]} {words.strip()}"
        fields, conf = self.chain.translate(text)
        return text, fields, conf

    def plain(self, code):
        return staff_line(code, self.evid)          # everyday wording for staff screens (wording.py); display only

    def question_text(self, code):
        return self.evid[code]["question_en"].strip()

    # ---- the questions the system asks -----------------------------------------------------------------------------------
    def next_questions(self, yes, no, k=3, skip=()):
        planks = self.chain.seesaw.route(yes, "condition", negatives=no)
        have = {f.partition("_@_")[0] for f in yes} | set(no) | set(skip)     # skip: already put to this patient ("not sure" counts)
        win = planks[0]
        in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
        picks = []
        # 1. rule out the worst: the most severe condition near the top that is not yet the danger pick
        danger = min(in_play, key=lambda p: (self.sev[p["label"]], -p["force"]))
        for p in planks[:10]:
            if self.sev[p["label"]] <= 2 and p["label"] != danger["label"]:
                c = p["label"]
                q = max((f for f in self.binary if f not in have and f not in picks), key=lambda f: self.cond_lean[f].get(c, -9), default=None)
                if q:
                    picks.append(q)
                break
        # 2. what best separates the top two conditions
        a, b = planks[0]["label"], planks[1]["label"]
        cand = sorted((max(abs(self.cond_lean[f].get(a, 0) - self.cond_lean[f].get(b, 0)),
                           abs(self.cond_no[f].get(a, 0) - self.cond_no[f].get(b, 0))), f)
                      for f in self.binary if f not in have and f not in picks)
        picks += [f for _, f in reversed(cand)][:k - len(picks)]
        return picks[:k]

    # ---- reviewed rules (people teach; the system remembers exactly what it was taught) ----------------------------------
    def rule_from_correction(self, yes, no, to):
        """A nurse moved a patient to `to` and a reviewer approved it as a rule. Which of the patient's answers does the rule
        hang on, and how hard does it push? Nothing here is learned by the system on its own: a person made the correction,
        a person approved it, and both choices below come from that one patient and the published counted table.
          key answers: the patient's answers that pull toward `to` more than toward the department the seesaw preferred
                       (up to 5, strongest first); none at all = it cannot become a general rule
          push:        the gap that patient had between the seesaw's choice and `to`, plus one unit, so a patient like the
                       corrected one lands where the nurse said, clearly and not on a tie
        Returns None when no answer separates the two departments."""
        planks = self.chain.seesaw.route(yes, "department", negatives=no)
        force = {p["label"]: p["force"] for p in planks}
        other = planks[0]["label"] if planks[0]["label"] != to else planks[1]["label"]
        lean = self.chain.counted["kinds"]["department"]["lean"]
        sep = sorted(((lean[f].get(to, 0.0) - lean[f].get(other, 0.0), f) for f in yes if f in lean), reverse=True)
        key = [f for d, f in sep if d > 0][:5]
        if not key:
            return None
        gap = max(0.0, force[planks[0]["label"]] - force.get(to, 0.0))
        return {"rule_fields": key, "rule_fields_plain": [self.plain(f) for f in key], "compared_with": other,
                "gap": round(gap, 3), "push": round(gap + 1.0, 3)}

    def apply_rules(self, yes, no, rules, dept, danger):
        """Approved rules whose key answers this patient ALL gave become extra levers on the department seesaw.
        A rule can move the route only if, with its lever, its department wins the seesaw, and never away from an Immediate
        or Very urgent possibility still in play (danger first always holds)."""
        applying = [r for r in rules if set(r["rule_fields"]) <= set(yes)]
        if not applying:
            return dept, []
        extra = [{"id": f"rule:{r['id']}", "beam": r["to"], "driver": f"_rule_{r['id']}", "gates": [],
                  "if_yes": math.tanh(r["push"]), "if_no": 0.0} for r in applying]
        planks = self.chain.seesaw.route(yes, "department", negatives=no, extra=extra)
        winner = planks[0]["label"]
        out = []
        for r in applying:
            info = {"id": r["id"], "to": r["to"], "fields": r["rule_fields_plain"], "why": r.get("why", ""),
                    "approved": r.get("reviewed", "")}
            if winner != r["to"]:
                info["effect"] = f"outweighed: this patient's other answers still point to {winner}"
            elif self.sev[danger["label"]] <= 2 and r["to"] != self.table[danger["label"]]["department"]:
                info["effect"] = f"danger first keeps {dept}: {danger['label']} is still in play"
            elif r["to"] == dept:
                info["effect"] = "agrees with the route"
            else:
                info["effect"] = f"moved the route from {dept} to {r['to']}"
                dept = r["to"]
            out.append(info)
        return dept, out

    # ---- the decision ----------------------------------------------------------------------------------------------------
    def decide(self, text, yes, no, conf, config, custom_yes=(), rules=()):
        planks = self.chain.seesaw.route(yes, "condition", negatives=no)
        win = planks[0]
        in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
        danger = min(in_play, key=lambda p: (self.sev[p["label"]], -p["force"]))
        likely_dept = self.table[win["label"]]["department"]
        dept = self.table[danger["label"]]["department"]
        urgency = SEV_URGENCY[self.sev[danger["label"]]]
        # second reader reads the words plus the yes-taps as short phrases
        extra = "".join(". " + short_phrase(self.question_text(f)) for f in yes if f in self.evid and "_@_" not in f and f not in text)
        r_pick, r_conf = self.chain.router_read(text + extra)
        nov, fam = self.sensor.read([text + extra])
        known = set(self.chain.counted["kinds"]["department"]["lean"])
        facts = {"evidence": 1.0 if any(f in known for f in yes) else 0.0,
                 "department_calm": -1.0 if win["fight"] else 1.0,
                 "router_agrees": 1.0 if r_pick == likely_dept else -1.0,
                 "translator_sure": round(2 * min(conf.values()) - 1, 4) if conf else 1.0,
                 "router_sure": round(2 * r_conf - 1, 4), "familiar": round(float(fam[0]), 4)}
        v = V.verdict_of(facts, self.levers)
        dept, rules_applied = self.apply_rules(yes, no, rules, dept, danger)
        # custom questions with a hard rule ("if yes, send to X"): a plain rule, because the hospital set it as a fixed constraint
        hard = [c for c in custom_yes if c.get("route_to")]
        if hard:
            dept = hard[0]["route_to"]
        # the hospital's setup: departments it has, and what happens when one cannot take the patient
        cfgd = config["departments"].get(dept, {"available": True})
        note = None
        if not cfgd.get("available", True):
            act = cfgd.get("if_unavailable", {"action": "redirect", "to": "Emergency (resus)"})
            note = (f"{dept} unavailable here: " + (f"redirected to {act['to']}" if act["action"] == "redirect"
                                                    else f"transfer out ({act.get('to') or 'arrange transfer'})"))
            dept = act["to"] if act["action"] == "redirect" else f"TRANSFER: {act.get('to') or 'external'}"
        dplanks = self.chain.seesaw.route(yes, "department", negatives=no)
        heads_up = [d["label"] for d in dplanks[:3] if d["label"] != dept][:2]
        if likely_dept != dept and likely_dept not in heads_up:
            heads_up.insert(0, likely_dept)
        return {
            "department": dept, "urgency": urgency, "likely_department": likely_dept, "unavailable_note": note,
            "rules_applied": rules_applied,
            "danger": {"condition": danger["label"], "severity": self.sev[danger["label"]], "department": self.table[danger["label"]]["department"]},
            "in_play": [{"condition": p["label"], "department": self.table[p["label"]]["department"], "severity": self.sev[p["label"]],
                         "gap": p["gap_to_winner"]} for p in in_play[:5]],
            "runners_up": [{"department": d["label"], "gap": d["gap_to_winner"]} for d in dplanks if d["label"] != dept][:2],
            "heads_up": heads_up[:2], "second_reader": {"department": r_pick, "confidence": round(r_conf, 3)},
            "novelty": round(float(nov[0]), 5), "facts": facts, "verdict": v["verdict"], "lean": v["tilt"],
            "ask_next": [self.question_text(q) for q in self.next_questions(yes, no)] if v["verdict"] != "yes" else [],
            "ask_codes": [{"code": q, "text": self.question_text(q)} for q in self.next_questions(yes, no)],
        }

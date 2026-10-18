"""Follow-up questions (NEXICOG, 13 Oct, fall-throughs were often resolved by asking): for every case the verdict
does not route, ASK the yes/no questions that pull hardest between the top two departments, let the (synthetic) patient
answer from their own record, re-route, re-judge. Up to ROUNDS rounds of QUESTIONS questions.

usage: run_env.sh route_from_fields/followup.py <facts_*.jsonl> <source patients csv> <router dir> <counted_table.json>
                                                <levers.json> <out.json> [rounds] [questions]

The patient answers YES if the question's field is in their DDXPlus record (or was already reported), NO otherwise.
A YES adds the field; a NO is a clear negative with its own counted push (no_lean). An unmentioned field still weighs nothing.
The router (second reader) re-reads the messy text with the yes-answers appended as short phrases.
Reported: per round, how many un-routed cases now route ("yes"), their accuracy, overall department accuracy, questions asked,
and wall-clock per round per case (the "is it fast and cheap enough for a desk" number).
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import ast
import csv
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import common                                    # noqa: E402
from seesaw_router import FieldRouter            # noqa: E402
from what_did_it_learn import Router             # noqa: E402
from build_messy_data import short_phrase        # noqa: E402
import verdict as V                              # noqa: E402

EVID = ROOT + "/data_raw/ddxplus_en/release_evidences.json"


def pick_questions(counted, evid, yes, no, a, b, k):
    lean = counted["kinds"]["department"]["lean"]
    no_lean = counted["kinds"]["department"]["no_lean"]
    asked = {f.partition("_@_")[0] for f in yes} | set(no)
    cand = []
    for f, per in no_lean.items():
        if f in asked or evid.get(f, {}).get("data_type") != "B":
            continue
        y = abs(lean[f].get(a, 0) - lean[f].get(b, 0))
        n = abs(per.get(a, 0) - per.get(b, 0))
        cand.append((max(y, n), f))
    return [f for _, f in sorted(cand, reverse=True)[:k]]


def main(facts_path, patients_csv, router_dir, counted_path, levers_path, out_path, rounds=2, k=3):
    rounds, k = int(rounds), int(k)
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    common.TASK = "router"
    router = Router(router_dir, dev)
    seesaw = FieldRouter(counted_path)
    counted = json.load(open(counted_path, encoding="utf-8"))
    levers = json.load(open(levers_path, encoding="utf-8"))["levers"]
    evid = json.load(open(EVID, encoding="utf-8"))
    facts = [json.loads(l) for l in open(facts_path, encoding="utf-8")]
    facts = [f for f in facts if f.get("kind", "patient") in ("patient",)]
    need = {f["id"] for f in facts}
    record = {}
    with open(patients_csv, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i in need:
                record[i] = {it for it in ast.literal_eval(row["EVIDENCES"])}
    known = set(counted["kinds"]["department"]["lean"])
    texts = {}
    data_file = facts_path.replace("/runs/route_from_fields/verdict/facts_", "/data_built/")
    start_v = {}
    for f in facts:
        start_v[(f["id"], f["v"])] = V.verdict_of(f, levers)["verdict"]
    todo = [f for f in facts if start_v[(f["id"], f["v"])] != "yes"]
    summary = {"facts": facts_path, "cases": len(facts), "not_routed_at_start": len(todo),
               "start_auto_share": round(1 - len(todo) / len(facts), 4), "rounds": []}
    state = {(f["id"], f["v"]): {"yes": list(f["fields"]), "no": [], "asked": 0, "text_extra": "", "f": f,
                                 "verdict": start_v[(f["id"], f["v"])], "pick": f["dept_seesaw"]} for f in todo}
    # the messy text each case was read from (for the router's re-read)
    want = set(state)
    src = os.environ.get("FOLLOWUP_TEXTS")
    if src:
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            if (r["id"], r["v"]) in want:
                texts[(r["id"], r["v"])] = r["text"]
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        live = [key for key, s in state.items() if s["verdict"] != "yes"]
        for key in live:
            s = state[key]
            planks = seesaw.route(s["yes"], "department", negatives=s["no"])
            qs = pick_questions(counted, evid, s["yes"], s["no"], planks[0]["label"], planks[1]["label"], k)
            for q in qs:
                s["asked"] += 1
                if q in record.get(key[0], set()) or q in s["yes"]:
                    s["yes"].append(q)
                    s["text_extra"] += ". " + short_phrase(evid[q]["question_en"])
                else:
                    s["no"].append(q)
        reads = router.mouth([texts.get(key, "") + state[key]["text_extra"] for key in live]) if live else []
        for key, m in zip(live, reads):
            s = state[key]
            planks = seesaw.route(s["yes"], "department", negatives=s["no"])
            r_pick = common.DEPARTMENTS[max(range(14), key=m.__getitem__)]
            fx = dict(s["f"])
            fx.update(evidence=1.0 if any(x in known for x in s["yes"]) else 0.0,
                      department_calm=-1.0 if planks[0]["fight"] else 1.0,
                      router_agrees=1.0 if r_pick == planks[0]["label"] else -1.0, router_sure=round(2 * max(m) - 1, 4))
            s["verdict"] = V.verdict_of(fx, levers)["verdict"]
            s["pick"] = planks[0]["label"]
        secs = (time.time() - t0) / max(len(live), 1)
        now_yes = [s for s in state.values() if s["verdict"] == "yes"]
        all_right = sum(s["pick"] == s["f"]["dept_true"] for s in state.values())
        summary["rounds"].append({
            "round": rnd, "asked_this_round": len(live),
            "now_routed_of_not_routed": len(now_yes), "share_resolved": round(len(now_yes) / max(len(state), 1), 4),
            "acc_of_resolved": round(sum(s["pick"] == s["f"]["dept_true"] for s in now_yes) / max(len(now_yes), 1), 4),
            "acc_of_all_not_routed_cases_now": round(all_right / max(len(state), 1), 4),
            "mean_questions_per_case_so_far": round(sum(s["asked"] for s in state.values()) / max(len(state), 1), 2),
            "seconds_per_case_this_round": round(secs, 4)})
        print(json.dumps(summary["rounds"][-1]), flush=True)
    start_right = sum(f["dept_seesaw"] == f["dept_true"] for f in todo)
    summary["acc_of_not_routed_at_start"] = round(start_right / max(len(todo), 1), 4)
    json.dump(summary, open(out_path, "w", encoding="utf-8"), indent=1)
    print("start accuracy of the not-routed cases:", summary["acc_of_not_routed_at_start"])


if __name__ == "__main__":
    main(*sys.argv[1:9])

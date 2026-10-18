"""RULE OUT THE WORST FIRST: danger-first plus questions aimed at the dangerous conditions near the top.

usage: pyd.sh -I rule_out_worst.py <messy file> <source patients csv> <out.json> [n] [top_n] [per_condition] [rounds]

After danger-first (danger_first.py), for every SEVERE condition (DDXPlus severity 1-2) ranked in the top_n condition planks
but not chosen, ask its per_condition most characteristic unanswered yes/no questions (largest counted yes-lean for that
condition). The synthetic patient answers from their own record (yes if the field is in it, else a clear no with its counted
push). Re-run danger-first. Reported against danger-first alone: under-triage, severe cases missed, over-triage, department
accuracy, and how many questions were asked per case.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import ast
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seesaw_router import FieldRouter     # noqa: E402

R = ROOT + ""


def danger_pick(router, sev, yes, no):
    planks = router.route(yes, "condition", negatives=no)
    win = planks[0]
    in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
    return min(in_play, key=lambda p: (sev[p["label"]], -p["force"])), planks


def main(data, patients_csv, out_path, n=1000, top_n=10, per_condition=2, rounds=1):
    n, top_n, per_condition, rounds = int(n), int(top_n), int(per_condition), int(rounds)
    router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    counted = json.load(open(f"{R}/runs/route_from_fields/counted_table.json", encoding="utf-8"))
    lean = counted["kinds"]["condition"]["lean"]
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    evid = json.load(open(f"{R}/data_raw/ddxplus_en/release_evidences.json", encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}
    recs = [json.loads(l) for l in open(data, encoding="utf-8")]
    recs = random.Random(3).sample(recs, min(n, len(recs)))
    need = {r["id"] for r in recs}
    record = {}
    with open(patients_csv, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i in need:
                record[i] = set(ast.literal_eval(row["EVIDENCES"]))
    yes_q = [f for f in counted["kinds"]["condition"]["no_lean"] if evid.get(f, {}).get("data_type") == "B"]
    t = {m: dict(dept=0, under=0, over=0, severe_missed=0, n_severe=0, asked=0) for m in ("danger_first", "rule_out_worst")}
    for r in recs:
        yes, no = r["codes"].split(), []
        before, _ = danger_pick(router, sev, yes, no)
        asked = 0
        for _ in range(rounds):
            pick, planks = danger_pick(router, sev, yes, no)
            have = {f.partition("_@_")[0] for f in yes} | set(no)
            for p in planks[:top_n]:
                c = p["label"]
                if sev[c] > 2 or c == pick["label"]:
                    continue
                qs = sorted((f for f in yes_q if f not in have), key=lambda f: -lean[f].get(c, -9))[:per_condition]
                for q in qs:
                    asked += 1
                    have.add(q)
                    (yes if q in record.get(r["id"], set()) else no).append(q)
        after, _ = danger_pick(router, sev, yes, no)
        ts = sev[r["pathology"]]
        for m, pick, a in (("danger_first", before, 0), ("rule_out_worst", after, asked)):
            x, ps = t[m], sev[pick["label"]]
            x["dept"] += table[pick["label"]]["department"] == r["dept_true"]
            x["under"] += ps > ts
            x["over"] += ps < ts
            x["n_severe"] += ts <= 2
            x["severe_missed"] += ts <= 2 and ps > ts
            x["asked"] += a
    k = len(recs)
    rep = {m: {"dept_acc": round(x["dept"] / k, 4), "under_triage": round(x["under"] / k, 4),
               "severe_missed_of_severe": round(x["severe_missed"] / max(x["n_severe"], 1), 4),
               "over_triage": round(x["over"] / k, 4), "questions_per_case": round(x["asked"] / k, 2)} for m, x in t.items()}
    rep["settings"] = {"n": k, "top_n": top_n, "per_condition": per_condition, "rounds": rounds}
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:8])

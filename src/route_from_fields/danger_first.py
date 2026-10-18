"""DANGER FIRST (NEXICOG, 13 Oct): "when somebody has two conflicting complaints, it's what is going to kill them first...
there ought to be the winner, not what the system's sure about."

usage: pyd.sh -I danger_first.py <counted_table.json> <routing_table.json> <messy file> <out.json> [n] [fields_key]

Condition planks (49, counted like the rest). The winner is the plank with the most force. A runner-up is STILL IN PLAY when
its force gap to the winner is smaller than the weight of evidence AGAINST the winner (the winner plank's contest): two
measured weights, no chosen number. Danger-first routes to the most SEVERE condition still in play (DDXPlus severity; ties go
to the stronger plank) and takes its department and urgency.

Scored against the recorded condition, as triage is scored:
  dept_acc        department matches the recorded condition's department
  under_triage    routed urgency LESS urgent than the recorded one (the dangerous error)
  severe_missed   recorded severity 1-2 and routed less urgent than that
  over_triage     routed MORE urgent than the recorded one (costs a bed, not a life)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seesaw_router import FieldRouter     # noqa: E402


def main(counted, table_path, data, out_path, n=1000, key="codes"):
    router = FieldRouter(counted)
    table = json.load(open(table_path, encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}
    recs = [json.loads(l) for l in open(data, encoding="utf-8")]
    recs = random.Random(3).sample(recs, min(int(n), len(recs)))
    tally = {m: {"dept": 0, "under": 0, "severe_missed": 0, "over": 0, "n_severe": 0, "in_play": 0} for m in ("plain", "danger_first")}
    for r in recs:
        fields = r[key].split() if isinstance(r[key], str) else r[key]
        planks = router.route(fields, "condition")
        win = planks[0]
        in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
        danger = min(in_play, key=lambda p: (sev[p["label"]], -p["force"]))
        true_cond = r["pathology"]
        ts = sev[true_cond]
        for m, pick in (("plain", win), ("danger_first", danger)):
            t = tally[m]
            ps = sev[pick["label"]]
            t["dept"] += table[pick["label"]]["department"] == r["dept_true"]
            t["under"] += ps > ts
            t["over"] += ps < ts
            t["n_severe"] += ts <= 2
            t["severe_missed"] += ts <= 2 and ps > ts
            t["in_play"] += len(in_play)
    k = len(recs)
    rep = {m: {"dept_acc": round(t["dept"] / k, 4), "under_triage": round(t["under"] / k, 4),
               "severe_missed_of_severe": round(t["severe_missed"] / max(t["n_severe"], 1), 4),
               "over_triage": round(t["over"] / k, 4), "mean_in_play": round(t["in_play"] / k, 2)} for m, t in tally.items()}
    rep["n"] = k
    rep["n_severe_cases"] = tally["plain"]["n_severe"]
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:7])

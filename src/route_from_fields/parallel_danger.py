"""PARALLEL DANGER ENGINE (NEXICOG, 13 Oct): danger must be a separate engine the identification experts never see, so they
cannot weigh what something is by how dangerous it would be.

usage: pyd.sh -I parallel_danger.py <out.json> <messy file> [<messy file> ...]

IDENTITY ENGINE   condition planks (49), counted on fields; knows nothing of severity. Gives the winner and the set still in
                  play (gap to winner < winner's contest).
DANGER ENGINE     urgency planks (5), counted on fields -> urgency directly; knows nothing of conditions.
JOIN              urgency = the danger engine's most forceful urgency plank. Department = the in-play condition whose severity
                  is CLOSEST to that urgency (ties: more urgent, then stronger plank); if none in play is as urgent as the
                  danger engine says, take the most severe in play, and the card says the danger engine wanted more.
Compared with plain (identity winner) and danger-first (most severe in play) on the same cases: department accuracy,
under-triage (routed urgency less urgent than recorded), severe missed, over-triage.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seesaw_router import FieldRouter     # noqa: E402

R = ROOT + ""
WORST = os.environ.get("DANGER_WORST") == "1"   # 1 = report the worst urgency still in play
URG_SEV = {"Immediate": 1, "Very urgent": 2, "Urgent": 3, "Standard": 4, "Non-urgent": 5}


def main(out_path, *files):
    router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}
    rep = {}
    for path in files:
        recs = random.Random(3).sample([json.loads(l) for l in open(path, encoding="utf-8")], 1000)
        t = {m: defaultdict(float) for m in ("plain", "danger_first", "parallel")}
        wanted_more = 0
        for r in recs:
            fields = r["codes"].split()
            planks = router.route(fields, "condition")
            win = planks[0]
            in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
            danger_first = min(in_play, key=lambda p: (sev[p["label"]], -p["force"]))
            uplanks = router.route(fields, "urgency")
            if WORST:      # the danger engine reports how urgent it COULD plausibly be: the most urgent plank still in play
                uin = [u for u in uplanks if u["gap_to_winner"] < uplanks[0]["contest"]] or [uplanks[0]]
                urg = min(uin, key=lambda u: URG_SEV[u["label"]])["label"]
            else:          # the danger engine reports the most likely urgency
                urg = uplanks[0]["label"]
            target = URG_SEV[urg]
            reachable = [p for p in in_play if sev[p["label"]] <= target]
            if reachable:
                par = min(reachable, key=lambda p: (target - sev[p["label"]], sev[p["label"]], -p["force"]))
            else:
                par = danger_first
                wanted_more += 1
            ts = sev[r["pathology"]]
            for m, pick in (("plain", win), ("danger_first", danger_first), ("parallel", par)):
                x = t[m]
                ps = min(sev[pick["label"]], target) if m == "parallel" else sev[pick["label"]]
                x["dept"] += table[pick["label"]]["department"] == r["dept_true"]
                x["under"] += ps > ts
                x["over"] += ps < ts
                x["n_sev"] += ts <= 2
                x["sev_missed"] += ts <= 2 and ps > ts
        name = os.path.basename(path)
        rep[name] = {m: {"dept_acc": round(x["dept"] / 1000, 4), "under": round(x["under"] / 1000, 4),
                         "severe_missed": round(x["sev_missed"] / max(x["n_sev"], 1), 4), "over": round(x["over"] / 1000, 4)}
                     for m, x in t.items()}
        rep[name]["danger_engine_wanted_more_than_anything_in_play"] = wanted_more / 1000
        print(name, json.dumps(rep[name]), flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

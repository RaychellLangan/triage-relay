"""LOUD CONDITIONS DROWN QUIET DANGEROUS ONES: does scoring each condition by the SHARE of its typical evidence fix it?

usage: pyd.sh -I loud_bias.py <messy_train.jsonl> <out.json> <messy file> [<messy file> ...]

EXPECTED force per condition = the mean force a training patient WITH that condition puts on that condition's own plank
(counted once over 300 training variants per condition, then frozen). RATIO = observed force / expected force.
Compared on each file: ranking by raw FORCE vs by RATIO, then danger-first on each (in play = within the winner's contest,
measured on the score being ranked). Scored: department accuracy, under-triage, severe missed, over-triage.
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


def main(train_path, out_path, *files):
    router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}
    by = defaultdict(list)
    rng = random.Random(9)
    for l in open(train_path, encoding="utf-8"):
        r = json.loads(l)
        if len(by[r["pathology"]]) < 300 and rng.random() < 0.05:
            by[r["pathology"]].append(r["codes"].split())
    expected = {}
    for c, rows in by.items():
        f = [next(p for p in router.route(x, "condition") if p["label"] == c)["force"] for x in rows]
        expected[c] = max(sum(f) / len(f), 1e-6)
    rep = {"expected_force": {c: round(v, 2) for c, v in sorted(expected.items(), key=lambda kv: kv[1])}}
    print("expected force, quietest 5:", list(rep["expected_force"].items())[:5], " loudest 5:", list(rep["expected_force"].items())[-5:])
    for path in files:
        recs = random.Random(3).sample([json.loads(l) for l in open(path, encoding="utf-8")], 1000)
        t = {m: defaultdict(float) for m in ("force", "ratio", "force_danger", "ratio_danger")}
        for r in recs:
            planks = router.route(r["codes"].split(), "condition")
            for p in planks:
                p["ratio"] = p["force"] / expected.get(p["label"], 1e-6)
            ts = sev[r["pathology"]]
            for score in ("force", "ratio"):
                ranked = sorted(planks, key=lambda p: -p[score])
                win = ranked[0]
                # in play: the gap on this score is smaller than the winner's contest expressed on the same score
                contest = win["contest"] / (expected.get(win["label"], 1.0) if score == "ratio" else 1.0)
                in_play = [p for p in ranked if win[score] - p[score] < contest] or [win]
                danger = min(in_play, key=lambda p: (sev[p["label"]], -p[score]))
                for m, pick in ((score, win), (score + "_danger", danger)):
                    x, ps = t[m], sev[pick["label"]]
                    x["dept"] += table[pick["label"]]["department"] == r["dept_true"]
                    x["under"] += ps > ts
                    x["over"] += ps < ts
                    x["n_sev"] += ts <= 2
                    x["sev_missed"] += ts <= 2 and ps > ts
        name = os.path.basename(path)
        rep[name] = {m: {"dept_acc": round(x["dept"] / 1000, 4), "under": round(x["under"] / 1000, 4),
                         "severe_missed": round(x["sev_missed"] / max(x["n_sev"], 1), 4), "over": round(x["over"] / 1000, 4)}
                     for m, x in t.items()}
        print(name, json.dumps(rep[name]), flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

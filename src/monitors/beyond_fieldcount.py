"""DIAGNOSTIC (learned ceiling, never shipped): does the router's novelty add anything BEYOND "how much did they tell us"?

usage: run_env.sh monitors/beyond_fieldcount.py <out.json> <messy file> [...]

Per case: field count (true fields in the text), text length, the tiny router's middle-block novelty (NoveltySensor), its mouth
confidence. Targets: the seesaw routes wrong / danger-first misses a severe case (true fields). Held-out AUROC of a logistic fit,
5 folds split by patient id, for: field count + length | + novelty | + mouth confidence | all. Also novelty's AUROC WITHIN each
field-count tercile (no fit at all).
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "route_from_fields"))
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
import common                                  # noqa: E402
from trajectory import auroc                   # noqa: E402
from adds_information import cv_auroc          # noqa: E402
from novelty_sensor import NoveltySensor       # noqa: E402
from seesaw_router import FieldRouter          # noqa: E402
from what_did_it_learn import Router           # noqa: E402

R = ROOT + ""


def main(out_path, *files):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sensor = NoveltySensor(f"{R}/runs/m_tiny_router/model", device=dev)
    common.TASK = "router"
    rtr = Router(f"{R}/runs/m_tiny_router/model", dev)
    seesaw = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}
    recs = []
    for p in files:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        recs += random.Random(3).sample(rows, min(1000, len(rows)))
    texts = [r["text"] for r in recs]
    nov, _ = sensor.read(texts)
    conf = np.array([max(m) for m in rtr.mouth(texts)])
    nf = np.array([len(r["codes"].split()) for r in recs], float)
    ln = np.array([len(t) for t in texts], float)
    wrong, severe = [], []
    for r in recs:
        pl = seesaw.route(r["codes"].split(), "condition")
        win = pl[0]
        in_play = [p for p in pl if p["gap_to_winner"] < win["contest"]] or [win]
        d = min(in_play, key=lambda p: (sev[p["label"]], -p["force"]))
        ts = sev[r["pathology"]]
        wrong.append(table[win["label"]]["department"] != r["dept_true"])
        severe.append(ts <= 2 and sev[d["label"]] > ts)
    groups = np.array([r["id"] for r in recs])
    rep = {"n": len(recs)}
    for name, y in (("seesaw_wrong", np.array(wrong)), ("severe_missed", np.array(severe))):
        base = np.c_[nf, ln]
        rep[name] = {"n_target": int(y.sum()),
                     "fieldcount+length": cv_auroc(base, y, groups),
                     "+novelty": cv_auroc(np.c_[base, nov], y, groups),
                     "+mouth_conf": cv_auroc(np.c_[base, conf], y, groups),
                     "+novelty+mouth_conf": cv_auroc(np.c_[base, nov, conf], y, groups)}
        t1, t2 = np.quantile(nf, [1 / 3, 2 / 3])
        bins = np.digitize(nf, [t1, t2])
        rep[name]["novelty_AUROC_within_fieldcount_tercile (few -> many)"] = [auroc(nov[bins == b], y[bins == b]) for b in range(3)]
        rep[name]["target_rate_by_tercile"] = [round(float(y[bins == b].mean()), 4) for b in range(3)]
        print(name, json.dumps(rep[name]), flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Why does GPT-2's middle find the seesaw's worst misses MORE familiar (novelty AUROC below chance)?

usage: run_env.sh monitors/gpt2_reversal.py <model dir> <out.json> <messy file> [...]

Tests, per layer, on the same 4,000 cases and targets as novelty.py:
  A  novelty as before (z-scored, 5 nearest of 4,000 training states)
  B  the same after projecting out the top-8 principal directions of the calibration states (rogue / shared axes)
  C  the AUROC of TEXT LENGTH (characters) and of FIELD COUNT alone, and each layer's novelty correlation with length
If B removes the reversal: rogue/shared directions. If length alone reproduces it and novelty tracks length: a length effect.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "route_from_fields"))
import common                                  # noqa: E402
from novelty import read_states                # noqa: E402
from trajectory import auroc                   # noqa: E402
from seesaw_router import FieldRouter          # noqa: E402

R = ROOT + ""


def knn_novelty(Q, B):
    Qu = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-9)
    Bu = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-9)
    return 1 - np.sort(Qu @ Bu.T, -1)[:, -5:].mean(-1)


def main(model_dir, out_path, *files):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    common.TASK = "router"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(dev).eval()
    router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}
    rng, train = random.Random(17), []
    for l in open(f"{R}/data_built/messy_train.jsonl", encoding="utf-8"):
        if rng.random() < 0.0125:
            train.append(json.loads(l)["text"])
        if len(train) >= 5000:
            break
    calib, _ = read_states(model, tok, train[:1000], dev)
    bank, _ = read_states(model, tok, train[1000:], dev)
    recs = []
    for p in files:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        recs += random.Random(3).sample(rows, min(1000, len(rows)))
    S, _ = read_states(model, tok, [r["text"] for r in recs], dev)
    sev_missed, wrong = [], []
    for r in recs:
        pl = router.route(r["codes"].split(), "condition")
        win = pl[0]
        in_play = [p for p in pl if p["gap_to_winner"] < win["contest"]] or [win]
        d = min(in_play, key=lambda p: (sev[p["label"]], -p["force"]))
        ts = sev[r["pathology"]]
        sev_missed.append(ts <= 2 and sev[d["label"]] > ts)
        wrong.append(table[win["label"]]["department"] != r["dept_true"])
    sev_missed, wrong = np.array(sev_missed), np.array(wrong)
    length = np.array([len(r["text"]) for r in recs], float)
    nfields = np.array([len(r["codes"].split()) for r in recs], float)
    mu, sd = calib.mean(0), calib.std(0) + 1e-6
    rep = {"model": model_dir, "n": len(recs), "n_severe_missed": int(sev_missed.sum()), "n_wrong": int(wrong.sum()),
           "length_AUROC": {"severe_missed": auroc(length, sev_missed), "wrong": auroc(length, wrong)},
           "fieldcount_AUROC": {"severe_missed": auroc(nfields, sev_missed), "wrong": auroc(nfields, wrong)},
           "layers": []}
    for l in range(S.shape[1]):
        cz, bz, qz = (calib[:, l] - mu[l]) / sd[l], (bank[:, l] - mu[l]) / sd[l], (S[:, l] - mu[l]) / sd[l]
        a = knn_novelty(qz, bz)
        _, _, vt = np.linalg.svd(cz - cz.mean(0), full_matrices=False)
        P = vt[:8]
        b = knn_novelty(qz - (qz @ P.T) @ P, bz - (bz @ P.T) @ P)
        rep["layers"].append({"layer": l, "A_severe": auroc(a, sev_missed), "B_severe_top8_removed": auroc(b, sev_missed),
                              "A_wrong": auroc(a, wrong), "B_wrong_top8_removed": auroc(b, wrong),
                              "corr_A_length": round(float(np.corrcoef(a, length)[0, 1]), 3)})
        print(rep["layers"][-1], flush=True)
    print("length AUROC", rep["length_AUROC"], "| field count AUROC", rep["fieldcount_AUROC"])
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""DIAGNOSTIC ONLY (learned; never the shipped monitor): do the middle readings carry information about a wrong answer
BEYOND the mouth's own confidence?

usage: run_env.sh monitors/adds_information.py <readings.npz> <out.json>

A logistic fit, 5-fold cross-validated with folds split BY PATIENT (the same person's stressed copies never sit on both
sides), predicts "the mouth is wrong" from (a) mouth confidence alone, (b) the middle readings alone, (c) both. Held-out
AUROC of (c) above (a) is the ceiling on what a monitor reading the middle could add. If (c) == (a), the middle adds
nothing a threshold on confidence could not do, and the seesaw would be decoration.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import auroc, features     # noqa: E402


def fit_logistic(X, y, l2=1e-2, iters=300, lr=0.5):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def cv_auroc(X, y, groups, folds=5):
    out = np.zeros(len(y))
    for f in range(folds):
        test = (groups % folds) == f
        mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-9
        w, b = fit_logistic((X[~test] - mu) / sd, y[~test].astype(float))
        out[test] = (X[test] - mu) / sd @ w + b
    return auroc(out, y)


def main(npz_path, out_path):
    z = np.load(npz_path)
    fit_acc = z["fit_acc"]
    best_layer, best_k = map(int, np.unravel_index(int(np.argmax(fit_acc)), fit_acc.shape))
    layer_k = [int(np.argmax(fit_acc[l])) for l in range(fit_acc.shape[0])]
    truth = z["truth"]
    names = sorted({k.split("__")[0] for k in z.files if "__" in k and not k.startswith("ood")})
    rows_mouth, rows_mid, ys, groups = [], [], [], []
    for name in names:
        f = features(z[f"{name}__cos"], z[f"{name}__mouth"], z[f"{name}__vocab"], z[f"{name}__znorm"],
                     best_layer, best_k, layer_k)
        cos = z[f"{name}__cos"]
        # per-layer margins along the trajectory (each layer at its own k): the SHAPE of the climb
        traj_margin = np.stack([np.sort(cos[:, l, layer_k[l]], -1)[:, -1] - np.sort(cos[:, l, layer_k[l]], -1)[:, -2]
                                for l in range(cos.shape[1])], 1)
        rows_mouth.append(np.c_[f["mouth_conf"]])
        rows_mid.append(np.c_[f["margin"], f["familiar"], (f["best_pick"] != f["mouth_pick"]).astype(float), f["settle"],
                              f["flips"], f["znorm_best"], (f["vocab_pick"] != f["mouth_pick"]).astype(float), traj_margin])
        ys.append(f["mouth_pick"] != truth)
        groups.append(np.arange(len(truth)))
    Xm, Xd, y, g = np.concatenate(rows_mouth), np.concatenate(rows_mid), np.concatenate(ys), np.concatenate(groups)
    rep = {"readings": npz_path, "n": int(len(y)), "n_wrong": int(y.sum()),
           "held_out_AUROC": {"mouth_confidence_only": cv_auroc(Xm, y, g),
                              "middle_readings_only": cv_auroc(Xd, y, g),
                              "mouth_plus_middle": cv_auroc(np.c_[Xm, Xd], y, g)},
           "note": "learned diagnostic ceiling, folds by patient; not the monitor"}
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Does the middle know something the mouth does not? Scores every reading against the case where the mouth is wrong.

usage: run_env.sh monitors/trajectory.py <readings.npz> <out.json>

No thresholds anywhere: each signal is scored by AUROC (how well it RANKS wrong answers above right ones; 0.5 = chance,
1.0 = perfect). The bar every middle reading has to clear is MOUTH CONFIDENCE, the signal an ordinary system already has.

Per patient, from the de-rogued centre angles at every layer (k = 0, and the k each layer's fit table preferred):
  trajectory   the middle's pick at each layer, embedding -> top
  settle       how early the pick stops changing (layer index from which it holds to the top; later = settled late)
  flips        how many times the pick changes on the way up
  disagree     the BEST layer's pick (chosen on the fit table, never on eval) differs from what the mouth is about to say
  margin       best layer: top centre angle minus second
  familiar     best layer: the top centre angle itself (low = far from every department's patients)
Baselines: mouth confidence (top probability) and the VOCABULARY read (pooled embeddings).
Out-of-domain: how well each signal separates ood texts from full intakes.
"""
import json
import sys

import numpy as np


def auroc(score_bad, is_bad):
    """P(a random bad case scores higher than a random good case); ties count half. Higher score = more suspicious."""
    s, y = np.asarray(score_bad, float), np.asarray(is_bad, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                           # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def features(cos, mouth, vocab, znorm, best_layer, best_k, layer_k):
    # pick per layer, each layer read at the k ITS fit row preferred (GPT-2 needs rogue directions removed low down)
    picks = np.stack([cos[:, l, layer_k[l]].argmax(-1) for l in range(cos.shape[1])], 1)   # (N, L)
    L = picks.shape[1]
    same_as_top = picks == picks[:, -1:]
    settle = np.array([L - np.argmax(~row[::-1]) if (~row).any() else 0 for row in same_as_top])
    flips = (np.diff(picks, axis=1) != 0).sum(1)
    b = cos[:, best_layer, best_k]
    srt = np.sort(b, -1)
    return {
        "picks": picks, "best_pick": b.argmax(-1), "mouth_pick": mouth.argmax(-1), "vocab_pick": vocab.argmax(-1),
        "mouth_conf": mouth.max(-1), "margin": srt[:, -1] - srt[:, -2], "familiar": srt[:, -1],
        "settle": settle, "flips": flips, "znorm_best": znorm[:, best_layer],
    }


def main(npz_path, out_path):
    z = np.load(npz_path)
    fit_acc, ks = z["fit_acc"], [int(k) for k in z["ks"]]
    # chosen on the FIT table only, (layer, k) jointly; ties go to the earlier layer / smaller k
    best_layer, best_k = map(int, np.unravel_index(int(np.argmax(fit_acc)), fit_acc.shape))
    layer_k = [int(np.argmax(fit_acc[l])) for l in range(fit_acc.shape[0])]
    truth = z["truth"]
    stress_names = sorted({k.split("__")[0] for k in z.files if "__" in k and not k.startswith("ood")})
    report = {"readings": npz_path, "best_layer (0 = embedding)": best_layer, "best_k": ks[best_k],
              "fit_acc_k0_by_layer": [round(float(a), 3) for a in fit_acc[:, 0]],
              "fit_acc_bestk_by_layer": [round(float(fit_acc[l].max()), 3) for l in range(fit_acc.shape[0])],
              "k_by_layer": [ks[k] for k in layer_k],
              "vocab_fit_acc": round(float(z["vocab_fit_acc"]), 3), "stresses": {}}
    pooled = {k: [] for k in ("bad", "mouth_conf", "disagree", "margin", "familiar", "settle", "flips", "vocab_disagree",
                              "middle_right_mouth_wrong", "mouth_right_middle_wrong")}
    for name in stress_names:
        f = features(z[f"{name}__cos"], z[f"{name}__mouth"], z[f"{name}__vocab"], z[f"{name}__znorm"], best_layer, best_k, layer_k)
        bad = f["mouth_pick"] != truth
        disagree = (f["best_pick"] != f["mouth_pick"]).astype(float)
        layer_acc = [round(float((f["picks"][:, l] == truth).mean()), 3) for l in range(f["picks"].shape[1])]
        r = {
            "mouth_acc": round(float(1 - bad.mean()), 3),
            "middle_best_layer_acc": round(float((f["best_pick"] == truth).mean()), 3),
            "vocab_acc": round(float((f["vocab_pick"] == truth).mean()), 3),
            "middle_acc_by_layer_bestk": layer_acc,
            "middle_right_when_mouth_wrong": int(((f["best_pick"] == truth) & bad).sum()),
            "middle_wrong_when_mouth_right": int(((f["best_pick"] != truth) & ~bad).sum()),
            "n_mouth_wrong": int(bad.sum()),
            "AUROC_flag_mouth_wrong": {
                "mouth_low_confidence (baseline)": auroc(-f["mouth_conf"], bad),
                "middle_disagrees_with_mouth": auroc(disagree, bad),
                "middle_low_margin": auroc(-f["margin"], bad),
                "middle_unfamiliar": auroc(-f["familiar"], bad),
                "settles_late": auroc(f["settle"], bad),
                "flips_on_the_way_up": auroc(f["flips"], bad),
                "vocab_disagrees_with_mouth": auroc((f["vocab_pick"] != f["mouth_pick"]).astype(float), bad),
            },
        }
        report["stresses"][name] = r
        for k, v in (("bad", bad), ("mouth_conf", f["mouth_conf"]), ("disagree", disagree), ("margin", f["margin"]),
                     ("familiar", f["familiar"]), ("settle", f["settle"]), ("flips", f["flips"]),
                     ("vocab_disagree", (f["vocab_pick"] != f["mouth_pick"]).astype(float))):
            pooled[k].append(v)
        print(f"{name:12s} mouth {r['mouth_acc']:.3f} middle {r['middle_best_layer_acc']:.3f} vocab {r['vocab_acc']:.3f} | "
              f"middle right/mouth wrong {r['middle_right_when_mouth_wrong']:3d} of {r['n_mouth_wrong']:3d} | AUROC "
              + " ".join(f"{k.split(' ')[0][:14]}={v}" for k, v in r["AUROC_flag_mouth_wrong"].items()), flush=True)

    P = {k: np.concatenate(v) for k, v in pooled.items() if v}
    report["pooled_all_stresses_AUROC_flag_mouth_wrong"] = {
        "n": int(len(P["bad"])), "n_mouth_wrong": int(P["bad"].sum()),
        "mouth_low_confidence (baseline)": auroc(-P["mouth_conf"], P["bad"]),
        "middle_disagrees_with_mouth": auroc(P["disagree"], P["bad"]),
        "middle_low_margin": auroc(-P["margin"], P["bad"]),
        "middle_unfamiliar": auroc(-P["familiar"], P["bad"]),
        "settles_late": auroc(P["settle"], P["bad"]),
        "flips_on_the_way_up": auroc(P["flips"], P["bad"]),
        "vocab_disagrees_with_mouth": auroc(P["vocab_disagree"], P["bad"]),
    }
    print("POOLED", json.dumps(report["pooled_all_stresses_AUROC_flag_mouth_wrong"]), flush=True)

    # out-of-domain vs full intake: does anything notice?
    full = features(z["full__cos"], z["full__mouth"], z["full__vocab"], z["full__znorm"], best_layer, best_k, layer_k)
    report["ood_vs_full_AUROC"] = {}
    for name in ("ood_english", "ood_wordsalad"):
        if f"{name}__cos" not in z.files:
            continue
        o = features(z[f"{name}__cos"], z[f"{name}__mouth"], z[f"{name}__vocab"], z[f"{name}__znorm"], best_layer, best_k, layer_k)
        is_ood = np.r_[np.ones(len(o["familiar"])), np.zeros(len(full["familiar"]))].astype(bool)
        cat = lambda key, sign=1: sign * np.r_[o[key], full[key]]     # noqa: E731
        report["ood_vs_full_AUROC"][name] = {
            "n_ood": int(is_ood.sum()),
            "mouth_low_confidence": auroc(cat("mouth_conf", -1), is_ood),
            "middle_unfamiliar": auroc(cat("familiar", -1), is_ood),
            "middle_low_margin": auroc(cat("margin", -1), is_ood),
            "far_from_calibration (znorm)": auroc(cat("znorm_best"), is_ood),
            "mean_mouth_conf_on_ood": round(float(o["mouth_conf"].mean()), 3),
        }
    print("OOD", json.dumps(report["ood_vs_full_AUROC"]), flush=True)
    json.dump(report, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])

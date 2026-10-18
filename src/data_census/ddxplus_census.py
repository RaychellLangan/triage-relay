"""First look at DDXPlus: how flat are the ground-truth differentials?

usage: pyd.sh -I ddxplus_census.py <patients.csv> <release_conditions.json> <out.json>

For every patient: entropy of the given differential (in bits), the top probability,
the gap between the top two, and whether the recorded PATHOLOGY is the differential's top pick.
"""
import ast
import csv
import json
import math
import sys
from collections import Counter


def main(patients_path, conditions_path, out_path):
    conditions = json.load(open(conditions_path, encoding="utf-8"))
    severity = {name: c.get("severity") for name, c in conditions.items()}

    n = 0
    top_is_truth = 0
    truth_in_diff = 0
    entropies, tops, gaps, sizes = [], [], [], []
    pathologies = Counter()
    truth_rank = Counter()
    with open(patients_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            diff = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            diff.sort(key=lambda d: -d[1])
            probs = [p for _, p in diff]
            total = sum(probs)
            probs = [p / total for p in probs]
            names = [name for name, _ in diff]
            truth = row["PATHOLOGY"]

            n += 1
            pathologies[truth] += 1
            entropies.append(-sum(p * math.log2(p) for p in probs if p > 0))
            tops.append(probs[0])
            gaps.append(probs[0] - (probs[1] if len(probs) > 1 else 0.0))
            sizes.append(len(probs))
            if names[0] == truth:
                top_is_truth += 1
            if truth in names:
                truth_in_diff += 1
                truth_rank[min(names.index(truth) + 1, 6)] += 1
            else:
                truth_rank["absent"] += 1

    def quantiles(xs):
        xs = sorted(xs)
        return {q: round(xs[int(q / 100 * (len(xs) - 1))], 4) for q in (5, 25, 50, 75, 95)}

    out = {
        "file": patients_path,
        "patients": n,
        "pathologies": len(pathologies),
        "pathology_counts": dict(pathologies.most_common()),
        "severity_of_pathology": {k: severity.get(k) for k in pathologies},
        "truth_is_top_of_differential": round(top_is_truth / n, 4),
        "truth_inside_differential": round(truth_in_diff / n, 4),
        "truth_rank_in_differential (6 = 6th or lower)": {str(k): v for k, v in sorted(truth_rank.items(), key=lambda kv: str(kv[0]))},
        "differential_size": quantiles(sizes),
        "differential_entropy_bits": quantiles(entropies),
        "top_probability": quantiles(tops),
        "gap_top1_minus_top2": quantiles(gaps),
        "share_with_gap_under_0.05": round(sum(g < 0.05 for g in gaps) / n, 4),
        "share_with_top_under_0.30": round(sum(t < 0.30 for t in tops) / n, 4),
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("pathology_counts", "severity_of_pathology")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])

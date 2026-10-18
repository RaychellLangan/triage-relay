"""Spread each patient's differential over departments and urgency, and count how many cases stay torn.

usage: pyd.sh -I route_census.py <patients.csv> <routing_table.json> <out.json>

Per patient: department mass (sum of differential probability per department), urgency mass,
top department share, gap to the 2nd department, and the probability that the case is severity 1-2.
"""
import ast
import csv
import json
import sys
from collections import Counter, defaultdict


def main(patients_path, table_path, out_path):
    table = json.load(open(table_path, encoding="utf-8"))
    n = 0
    top_share, gap, severe_mass = [], [], []
    truth_dept_is_top = 0
    truth_dept = Counter()
    torn_pairs = Counter()
    with open(patients_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            diff = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            total = sum(p for _, p in diff)
            dept = defaultdict(float)
            severe = 0.0
            for name, p in diff:
                r = table[name]
                dept[r["department"]] += p / total
                if r["severity"] <= 2:
                    severe += p / total
            ranked = sorted(dept.items(), key=lambda kv: -kv[1])
            first = ranked[0]
            second = ranked[1] if len(ranked) > 1 else ("-", 0.0)
            n += 1
            top_share.append(first[1])
            gap.append(first[1] - second[1])
            severe_mass.append(severe)
            t = table[row["PATHOLOGY"]]["department"]
            truth_dept[t] += 1
            truth_dept_is_top += (first[0] == t)
            if first[1] - second[1] < 0.10:
                torn_pairs[" / ".join(sorted((first[0], second[0])))] += 1

    def quantiles(xs):
        xs = sorted(xs)
        return {q: round(xs[int(q / 100 * (len(xs) - 1))], 4) for q in (5, 25, 50, 75, 95)}

    out = {
        "patients": n,
        "truth_department_counts": dict(truth_dept.most_common()),
        "truth_department_is_top_department": round(truth_dept_is_top / n, 4),
        "top_department_share": quantiles(top_share),
        "gap_top_department_minus_second": quantiles(gap),
        "share_department_gap_under_0.10": round(sum(g < 0.10 for g in gap) / n, 4),
        "share_top_department_over_0.80": round(sum(s > 0.80 for s in top_share) / n, 4),
        "mass_on_severity_1_or_2": quantiles(severe_mass),
        "share_with_over_0.25_mass_on_severity_1_or_2": round(sum(s > 0.25 for s in severe_mass) / n, 4),
        "most_common_torn_department_pairs (gap < 0.10)": dict(torn_pairs.most_common(12)),
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])

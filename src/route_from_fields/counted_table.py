"""COUNTED, ONCE, THEN FROZEN: how much each field leans toward each department and each urgency.

usage: pyd.sh -I counted_table.py <train_patients.csv> <routing_table.json> <out.json> [n_patients]

For every field f (a yes/no question, or a question=value pair) and every department d, from n training patients:
    lean(f, d) = 0.5 * ln( P(f | d) / P(f | not d) )        counts smoothed with +1 (a stated choice)
That half log-ratio is exactly a seesaw lever's FORCE (atanh of its push), so a lever's push is tanh(lean).
The same for the five urgencies, and a base-rate lean per department/urgency.
Nothing is trained; every number in the output can be recomputed from the counts it also stores.
"""
import ast
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict


def main(patients, table_path, out_path, n=200000):
    n = int(n)
    table = json.load(open(table_path, encoding="utf-8"))
    rng, rows = random.Random(31), []
    with open(patients, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < n:
                rows.append((row["PATHOLOGY"], row["EVIDENCES"]))
            elif rng.randint(0, i) < n:
                rows[rng.randint(0, n - 1)] = (row["PATHOLOGY"], row["EVIDENCES"])
    out = {"patients_counted": len(rows), "smoothing": "+1 on every count", "kinds": {}}
    for kind, label_of in (("department", lambda p: table[p]["department"]),
                           ("urgency", lambda p: table[p]["urgency"]),
                           ("condition", lambda p: p)):          # condition planks: for danger-first (NEXICOG 13 Oct)
        n_label = Counter()
        n_field_label = defaultdict(Counter)
        for path, ev in rows:
            lab = label_of(path)
            n_label[lab] += 1
            for item in set(ast.literal_eval(ev)):
                n_field_label[item][lab] += 1
        total = sum(n_label.values())
        labels = sorted(n_label)
        lean = {}
        for item, c in n_field_label.items():
            tot_f = sum(c.values())
            lean[item] = {}
            for lab in labels:
                p_f_d = (c[lab] + 1) / (n_label[lab] + 2)
                p_f_nd = (tot_f - c[lab] + 1) / (total - n_label[lab] + 2)
                lean[item][lab] = round(0.5 * math.log(p_f_d / p_f_nd), 5)
        # a clear NO (follow-up questions): 0.5 ln( P(not f | d) / P(not f | not d) ), yes/no questions only
        no_lean = {}
        for item, c in n_field_label.items():
            if "_@_" in item:
                continue
            tot_f = sum(c.values())
            no_lean[item] = {}
            for lab in labels:
                p_nf_d = (n_label[lab] - c[lab] + 1) / (n_label[lab] + 2)
                p_nf_nd = ((total - n_label[lab]) - (tot_f - c[lab]) + 1) / (total - n_label[lab] + 2)
                no_lean[item][lab] = round(0.5 * math.log(p_nf_d / p_nf_nd), 5)
        base = {lab: round(0.5 * math.log(n_label[lab] / (total - n_label[lab])), 5) for lab in labels}
        out["kinds"][kind] = {"labels": labels, "base_lean": base, "lean": lean, "no_lean": no_lean,
                              "counts": {"label": dict(n_label), "field_label": {k: dict(v) for k, v in n_field_label.items()}}}
        print(kind, len(labels), "labels,", len(lean), "fields counted", flush=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(*sys.argv[1:5])

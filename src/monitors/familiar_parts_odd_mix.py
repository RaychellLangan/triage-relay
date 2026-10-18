"""Are the seesaw's stray-symptom mistakes made of UNFAMILIAR symptoms, or of FAMILIAR symptoms in a combination never seen?

usage: pyd.sh -I familiar_parts_odd_mix.py <messy stress file> [n]

For every case the condition planks route to the wrong department: (a) the rarest single field's count among the 200,000 counted
training patients, (b) for each stray field (one not listed for the patient's own condition), how many training patients had it
TOGETHER with any of the patient's own fields; the case's "combination familiarity" = the largest of those counts.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import ast
import csv
import json
import random
import sys
from collections import Counter

sys.path.insert(0, ROOT + "/src/route_from_fields")
from seesaw_router import FieldRouter     # noqa: E402

R = ROOT + ""


def main(path, n=1000):
    counted = json.load(open(f"{R}/runs/route_from_fields/counted_table.json", encoding="utf-8"))
    field_count = {f: sum(c.values()) for f, c in counted["kinds"]["condition"]["counts"]["field_label"].items()}
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    conds = json.load(open(f"{R}/data_raw/ddxplus_en/release_conditions.json", encoding="utf-8"))
    router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    recs = random.Random(3).sample([json.loads(l) for l in open(path, encoding="utf-8")], int(n))
    wrong = []
    for r in recs:
        if table[router.route(r["codes"].split(), "condition")[0]["label"]]["department"] != r["dept_true"]:
            own_set = set(conds[r["pathology"]]["symptoms"]) | set(conds[r["pathology"]]["antecedents"])
            codes = r["codes"].split()
            strays = [c for c in codes if c.partition("_@_")[0] not in own_set]
            own = [c for c in codes if c.partition("_@_")[0] in own_set]
            wrong.append((codes, strays, own))
    pairs = {(s, o) for _, st, ow in wrong for s in st for o in ow}
    together = Counter()
    with open(f"{R}/data_raw/ddxplus_en_unzipped/release_train_patients", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= 200000:
                break
            ev = set(ast.literal_eval(row["EVIDENCES"]))
            for s, o in pairs:
                if s in ev and o in ev:
                    together[(s, o)] += 1
    rarest = [min(field_count.get(c, 0) for c in codes) for codes, _, _ in wrong]
    combo = [max([together[(s, o)] for s in st for o in ow] or [0]) for _, st, ow in wrong]
    with_strays = [c for c, (_, st, _) in zip(combo, wrong) if st]
    print(f"{path.split('/')[-1]}: {len(wrong)} seesaw-wrong cases of {len(recs)}")
    print(f"  rarest single field, median count in 200k training patients: {sorted(rarest)[len(rarest)//2]} "
          f"| cases whose every field appears >= 1,000 times: {sum(x >= 1000 for x in rarest)} of {len(rarest)}")
    print(f"  cases with stray fields: {len(with_strays)}; their best stray+own pairing seen together in training: median "
          f"{sorted(with_strays)[len(with_strays)//2] if with_strays else '-'} patients; never together: {sum(x == 0 for x in with_strays)}; "
          f"fewer than 10: {sum(x < 10 for x in with_strays)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

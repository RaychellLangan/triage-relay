"""Score the seesaw router on a fields file.

usage: pyd.sh -I route_eval.py <counted_table.json> <messy_split.jsonl> <out.json> [n] [fields_key]

fields_key: which list of fields to route from: "codes" (the TRUE fields in the text: the ceiling a perfect translator
would reach) or the key a translator's predictions were saved under. Reports department / urgency accuracy, how often the
winning plank FIGHTS (the losing side outweighs its margin), accuracy with and without a fight, and the same split by
the keep / stray conditions the messy text was made under.
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seesaw_router import FieldRouter     # noqa: E402


def main(counted, data, out_path, n=2000, key="codes"):
    router = FieldRouter(counted)
    recs = [json.loads(l) for l in open(data, encoding="utf-8")]
    recs = random.Random(3).sample(recs, min(int(n), len(recs)))
    stats = defaultdict(lambda: defaultdict(float))
    for r in recs:
        fields = r[key].split() if isinstance(r[key], str) else r[key]
        d = router.route(fields, "department")
        u = router.route(fields, "urgency")
        top = d[0]
        right = top["label"] == r["dept_true"]
        for group in ("all", f"keep_{r['keep_p']}", f"stray_{r['stray_p']}"):
            s = stats[group]
            s["n"] += 1
            s["dept_right"] += right
            s["urgency_right"] += u[0]["label"] == r["urgency_true"]
            s["fight"] += top["fight"]
            s["right_no_fight"] += right and not top["fight"]
            s["n_no_fight"] += not top["fight"]
            s["right_fight"] += right and top["fight"]
            s["truth_in_top2"] += r["dept_true"] in (d[0]["label"], d[1]["label"])
    rep = {}
    for g, s in sorted(stats.items()):
        n_ = s["n"]
        rep[g] = {"n": int(n_), "dept_acc": round(s["dept_right"] / n_, 4), "urgency_acc": round(s["urgency_right"] / n_, 4),
                  "truth_in_top2": round(s["truth_in_top2"] / n_, 4), "fight_share": round(s["fight"] / n_, 4),
                  "dept_acc_when_no_fight": round(s["right_no_fight"] / max(s["n_no_fight"], 1), 4),
                  "dept_acc_when_fight": round(s["right_fight"] / max(s["fight"], 1), 4)}
        print(g, rep[g], flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])

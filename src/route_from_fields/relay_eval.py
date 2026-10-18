"""NEXICOG's relay (13 Oct): the seesaw routes every case; where it FALLS THROUGH (its winning plank fights), the transformer
router reads the messy text and has its say. Are the two wrong on the same cases, or different ones?

usage: run_env.sh route_from_fields/relay_eval.py <router model dir> <counted_table.json> <messy_split.jsonl> <out.json>
                                                  [n] [fields_key]

For the same n test variants (same seed as route_eval.py): the seesaw's department from the fields (fields_key: "codes" =
true fields, the ceiling), and the router's department read at the cue from the messy TEXT. Reported on all cases, on the
seesaw's fight cases and on its calm cases: each one's accuracy, how often they AGREE, accuracy when they agree, and what a
relay would score (seesaw where calm, router where it fights). Agreement on a fight case is the "second reader" signal;
disagreement is the clearest hand-to-a-person there is.
"""
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
import common                                   # noqa: E402
from seesaw_router import FieldRouter           # noqa: E402
from what_did_it_learn import Router            # noqa: E402


def main(model_dir, counted, data, out_path, n=2000, key="codes"):
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    common.TASK = "router"
    router = Router(model_dir, dev)
    seesaw = FieldRouter(counted)
    recs = [json.loads(l) for l in open(data, encoding="utf-8")]
    recs = random.Random(3).sample(recs, min(int(n), len(recs)))
    mouths = router.mouth([r["text"] for r in recs])
    groups = {"all": [], "seesaw_fights": [], "seesaw_calm": []}
    for r, m in zip(recs, mouths):
        fields = r[key].split() if isinstance(r[key], str) else r[key]
        s = seesaw.route(fields, "department")[0]
        t = common.DEPARTMENTS[max(range(14), key=m.__getitem__)]
        row = {"truth": r["dept_true"], "seesaw": s["label"], "router": t, "router_conf": max(m), "fight": s["fight"]}
        groups["all"].append(row)
        groups["seesaw_fights" if s["fight"] else "seesaw_calm"].append(row)
    rep = {}
    for g, rows in groups.items():
        k = len(rows)
        if not k:
            continue
        agree = [x for x in rows if x["seesaw"] == x["router"]]
        relay = sum((x["router"] if x["fight"] else x["seesaw"]) == x["truth"] for x in rows)
        rep[g] = {"n": k,
                  "seesaw_acc": round(sum(x["seesaw"] == x["truth"] for x in rows) / k, 4),
                  "router_acc": round(sum(x["router"] == x["truth"] for x in rows) / k, 4),
                  "agree_share": round(len(agree) / k, 4),
                  "acc_when_agree": round(sum(x["seesaw"] == x["truth"] for x in agree) / max(len(agree), 1), 4),
                  "both_wrong": round(sum(x["seesaw"] != x["truth"] and x["router"] != x["truth"] for x in rows) / k, 4),
                  "relay_acc (seesaw calm, router on fights)": round(relay / k, 4)}
        print(g, rep[g], flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:7])

"""NEXICOG's relay on the REAL chain: seesaw on the TRANSLATOR's predicted fields; where it fights, the router reads the text.

usage: run_env.sh route_from_fields/relay_real_chain.py <router model dir> <translate_*.jsonl> <messy_split.jsonl> <out.json>

Same report as relay_eval.py (all / seesaw fights / seesaw calm: each one's accuracy, agreement, accuracy when they agree,
both wrong, relay accuracy), but the seesaw's fields come from translate_eval.py's free-running predictions.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
import common                                   # noqa: E402
from what_did_it_learn import Router            # noqa: E402


def main(model_dir, preds_path, data, out_path):
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    common.TASK = "router"
    router = Router(model_dir, dev)
    preds = [json.loads(l) for l in open(preds_path, encoding="utf-8")]
    want = {(p["id"], p["v"]) for p in preds}
    texts = {}
    for l in open(data, encoding="utf-8"):
        r = json.loads(l)
        if (r["id"], r["v"]) in want:
            texts[(r["id"], r["v"])] = r["text"]
    mouths = router.mouth([texts[(p["id"], p["v"])] for p in preds])
    groups = {"all": [], "seesaw_fights": [], "seesaw_calm": []}
    for p, m in zip(preds, mouths):
        row = {"truth": p["dept_true"], "seesaw": p["dept_pred"], "fight": p["dept_fight"],
               "router": common.DEPARTMENTS[max(range(14), key=m.__getitem__)]}
        groups["all"].append(row)
        groups["seesaw_fights" if row["fight"] else "seesaw_calm"].append(row)
    rep = {}
    for g, rows in groups.items():
        k = len(rows)
        if not k:
            continue
        agree = [x for x in rows if x["seesaw"] == x["router"]]
        rep[g] = {"n": k,
                  "seesaw_acc": round(sum(x["seesaw"] == x["truth"] for x in rows) / k, 4),
                  "router_acc": round(sum(x["router"] == x["truth"] for x in rows) / k, 4),
                  "agree_share": round(len(agree) / k, 4),
                  "acc_when_agree": round(sum(x["seesaw"] == x["truth"] for x in agree) / max(len(agree), 1), 4),
                  "both_wrong": round(sum(x["seesaw"] != x["truth"] and x["router"] != x["truth"] for x in rows) / k, 4),
                  "relay_acc": round(sum((x["router"] if x["fight"] else x["seesaw"]) == x["truth"] for x in rows) / k, 4)}
        print(g, rep[g], flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])

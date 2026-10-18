"""The verdict: should this case route automatically, and what does the ward see? A seesaw, levers COUNTED on validation.

usage:
  run_env.sh route_from_fields/verdict.py facts <router dir> <translate_*.jsonl> <messy file> <counted_table.json> <facts.jsonl>
  pyd.sh -I  route_from_fields/verdict.py fit   <facts_validation.jsonl> <levers.json>
  pyd.sh -I  route_from_fields/verdict.py eval  <levers.json> <facts.jsonl> <out.json>

FACTS per case (each -1..+1; 0 = don't know):
  department_calm    +1 the seesaw's winning department plank does not fight, -1 it fights
  router_agrees      +1 the transformer router (reading the messy text) picks the same department, -1 it does not
  translator_sure    2 x (the lowest confidence among the fields the translator wrote) - 1
  router_sure        2 x (the router's top probability) - 1
  evidence           +1 if the translator wrote at least one field the counted table knows, else 0: an EMPTY case has
                     no evidence and its verdict is NO IDEA (the seesaw's "empty and level"), whatever the base rate says
GOAL PLANK "safe to route automatically": one lever per fact. Each lever's push (for a yes reading and for a no reading) is
COUNTED once on the validation cases, the same way as the routing table: 0.5 ln(P(reading | routed right) / P(reading | wrong)),
+1 smoothing, readings split at 0 (a fact is a yes when > 0, a no when < 0; don't-know weighs nothing). Then frozen.
VERDICT: empty -> "no idea"; the goal plank fights -> "hand to a person"; else the plank's lean, shown in five words
(display only): yes / maybe yes / unsure / maybe no / no. The lean's NUMBER is what is scored (AUROC against right/wrong).
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FACTS = tuple(os.environ.get("VERDICT_FACTS", "department_calm,router_agrees,translator_sure,router_sure,familiar").split(","))


def facts_mode(router_dir, preds_path, data, counted, out_path):
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "train"))
    sys.path.insert(0, os.path.join(HERE, "..", "probes"))
    import common
    from what_did_it_learn import Router
    from seesaw_router import FieldRouter
    common.TASK = "router"
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    router = Router(router_dir, dev)
    known = set(json.load(open(counted, encoding="utf-8"))["kinds"]["department"]["lean"])
    seesaw = FieldRouter(counted)
    preds = [json.loads(l) for l in open(preds_path, encoding="utf-8")]
    want = {(p["id"], p["v"]) for p in preds}
    texts = {}
    for l in open(data, encoding="utf-8"):
        r = json.loads(l)
        if (r["id"], r["v"]) in want:
            texts[(r["id"], r["v"])] = r
    mouths = router.mouth([texts[(p["id"], p["v"])]["text"] for p in preds])
    sys.path.insert(0, os.path.join(HERE, "..", "monitors"))
    from novelty_sensor import NoveltySensor
    sensor = NoveltySensor(router_dir, device=dev, bank_file=os.environ.get("NOVELTY_BANK"))
    nov, fam = sensor.read([texts[(p["id"], p["v"])]["text"] for p in preds])
    with open(out_path, "w", encoding="utf-8") as out:
        for p, m, nv, fm in zip(preds, mouths, nov, fam):
            planks = seesaw.route(p["pred"], "department")
            r_pick = common.DEPARTMENTS[max(range(14), key=m.__getitem__)]
            conf = list(p["field_conf"].values())
            rec = {"id": p["id"], "v": p["v"], "kind": texts[(p["id"], p["v"])].get("kind", "patient"),
                   "dept_true": p["dept_true"], "dept_seesaw": p["dept_pred"], "dept_router": r_pick,
                   "top2": [x["label"] for x in planks[:2]], "planks": planks[:3], "fields": p["pred"],
                   "department_calm": -1.0 if p["dept_fight"] else 1.0,
                   "router_agrees": 1.0 if r_pick == p["dept_pred"] else -1.0,
                   "translator_sure": round(2 * min(conf) - 1, 4) if conf else 0.0,
                   "router_sure": round(2 * max(m) - 1, 4),
                   "evidence": 1.0 if any(f in known for f in p["pred"]) else 0.0,
                   "novelty": round(float(nv), 5), "familiar": round(float(fm), 4)}
            out.write(json.dumps(rec) + "\n")
    print("facts written:", len(preds), "->", out_path)


GRADED = ("translator_sure", "router_sure", "familiar", "memory_close")   # graded facts: counted per band, not split at 0
BANDS = 5


def band_of(x, edges):
    return sum(x > e for e in edges)


def fit_mode(facts_path, out_path):
    if os.environ.get("VERDICT_BANDS") == "1":
        return fit_banded(facts_path, out_path)
    rows = [json.loads(l) for l in open(facts_path, encoding="utf-8")]
    rows = [r for r in rows if r["evidence"] > 0]
    right = [r for r in rows if r["dept_seesaw"] == r["dept_true"]]
    wrong = [r for r in rows if r["dept_seesaw"] != r["dept_true"]]
    levers = {}
    for f in FACTS:
        def share(group, side):
            k = sum(1 for r in group if (r[f] > 0 if side == "yes" else r[f] < 0))
            return (k + 1) / (len(group) + 2)
        levers[f] = {side: round(math.tanh(0.5 * math.log(share(right, side) / share(wrong, side))), 5) for side in ("yes", "no")}
    rep = {"counted_on": facts_path, "n_right": len(right), "n_wrong": len(wrong), "levers": levers}
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


def fit_banded(facts_path, out_path):
    """Graded facts: band edges at the fact's own validation quantiles (20/40/60/80%), and one counted push per band:
    0.5 ln(P(band | right) / P(band | wrong)), +1 smoothing. Yes/no facts as before. Counted once, frozen."""
    rows = [json.loads(l) for l in open(facts_path, encoding="utf-8")]
    rows = [r for r in rows if r["evidence"] > 0]
    right = [r for r in rows if r["dept_seesaw"] == r["dept_true"]]
    wrong = [r for r in rows if r["dept_seesaw"] != r["dept_true"]]
    levers = {}
    for f in FACTS:
        if f in GRADED:
            vals = sorted(r[f] for r in rows)
            edges = [vals[int(q * (len(vals) - 1))] for q in (0.2, 0.4, 0.6, 0.8)]
            pushes = []
            for b in range(BANDS):
                pr = (sum(band_of(r[f], edges) == b for r in right) + 1) / (len(right) + 2)
                pw = (sum(band_of(r[f], edges) == b for r in wrong) + 1) / (len(wrong) + 2)
                pushes.append(round(math.tanh(0.5 * math.log(pr / pw)), 5))
            levers[f] = {"edges": edges, "band_push": pushes}
        else:
            def share(group, side):
                k = sum(1 for r in group if (r[f] > 0 if side == "yes" else r[f] < 0))
                return (k + 1) / (len(group) + 2)
            levers[f] = {side: round(math.tanh(0.5 * math.log(share(right, side) / share(wrong, side))), 5) for side in ("yes", "no")}
    rep = {"counted_on": facts_path, "n_right": len(right), "n_wrong": len(wrong), "banded": True, "levers": levers}
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


def verdict_of(rec, levers):
    sys.path.insert(0, ROOT + "/seesaw")
    import seesaw as engine
    if rec["evidence"] <= 0:
        return {"verdict": "no idea", "tilt": 0.0, "fight": False}
    use = [f for f in FACTS if f in levers and f in rec]
    lv, facts = [], {}
    for f in use:
        if "band_push" in levers[f]:           # a banded fact: its band's counted push, as a plain yes-fact
            push = levers[f]["band_push"][band_of(rec[f], levers[f]["edges"])]
            lv.append({"id": f, "beam": "safe", "driver": f, "gates": [], "if_yes": push, "if_no": 0.0})
            facts[f] = 1.0
        else:
            lv.append({"id": f, "beam": "safe", "driver": f, "gates": [], "if_yes": levers[f]["yes"], "if_no": levers[f]["no"]})
            facts[f] = rec[f]
    kb = {"beams": [{"name": "safe", "high": "route automatically", "low": "a person looks"}], "levers": lv}
    res = engine.Seesaws(kb).run(facts)["safe"]
    fight = engine.fights(res)
    t = res["tilt"]
    words = ("no", "maybe no", "unsure", "maybe yes", "yes")
    word = words[min(4, int((t + 1) / 2 * 5))]       # display only; the number is what is scored
    return {"verdict": "hand to a person" if fight else word, "tilt": round(t, 4), "fight": fight}


def eval_mode(levers_path, facts_path, out_path):
    sys.path.insert(0, os.path.join(HERE, "..", "monitors"))
    from trajectory import auroc
    levers = json.load(open(levers_path, encoding="utf-8"))["levers"]
    rows = [json.loads(l) for l in open(facts_path, encoding="utf-8")]
    out, by = [], {}
    for r in rows:
        v = verdict_of(r, levers)
        ok = r["dept_seesaw"] == r["dept_true"]
        out.append((v, ok, r))
        b = by.setdefault(v["verdict"], [0, 0])
        b[0] += 1
        b[1] += ok
    has = [(v, ok) for v, ok, r in out if r["evidence"] > 0]
    rep = {"facts": facts_path, "n": len(rows),
           "AUROC_goal_tilt_flags_wrong": auroc([-v["tilt"] for v, _ in has], [not ok for _, ok in has]),
           "AUROC_router_sure_alone": auroc([-r["router_sure"] for v, ok, r in out if r["evidence"] > 0],
                                            [not ok for v, ok, r in out if r["evidence"] > 0]),
           "verdicts": {k: {"n": a, "share": round(a / len(rows), 4), "dept_acc": round(b / a, 4)} for k, (a, b) in sorted(by.items())}}
    auto = [(v, ok) for v, ok in has if v["verdict"] in ("yes", "maybe yes")]
    rep["auto_routed_share"] = round(len(auto) / len(rows), 4)
    rep["auto_routed_acc"] = round(sum(ok for _, ok in auto) / max(len(auto), 1), 4)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    {"facts": facts_mode, "fit": fit_mode, "eval": eval_mode}[mode](*args)

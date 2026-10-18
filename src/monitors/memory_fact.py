"""The memory vote as a verdict fact: resolved cases remembered with their answers.

usage: run_env.sh monitors/memory_fact.py <out dir> <facts_val_all.jsonl> <facts files to augment...>

MEMORY  middle-block states (z-scored, unit length, as the novelty sensor reads them) of person-RESOLVED cases + the department
        the person chose (here: the recorded department of the validation cases).
FACTS   memory_agrees  +1 the vote of the 5 nearest remembered cases equals the seesaw's pick, -1 it does not
        memory_close   mean similarity of those 5 (graded; banded and counted like the other graded facts)
FOLD-SAFE  validation facts are augmented with a memory built from the OTHER half of validation (split by patient id parity),
           so the levers are counted on cases the memory never held. Any other file uses the memory of ALL validation cases.
Writes <out dir>/<same name> with the two facts added.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from novelty_sensor import NoveltySensor       # noqa: E402

R = ROOT + ""
SOURCES = {"val_clean": "messy_validate", "val_strays5": "messyval_strays5", "val_keep30": "messyval_keep30",
           "val_keep50_strays3": "messyval_keep50_strays3", "test2000": "messy_test", "strays5": "messystress_strays5",
           "keep30": "messystress_keep30", "keep50_strays3": "messystress_keep50_strays3", "ood": "messystress_ood"}


def texts_for(facts_file):
    name = os.path.basename(facts_file).replace("facts_", "").replace(".jsonl", "")
    src = SOURCES[name]
    want = {(f["id"], f["v"]) for f in map(json.loads, open(facts_file, encoding="utf-8"))}
    out = {}
    for l in open(f"{R}/data_built/{src}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if (r["id"], r["v"]) in want:
            out[(r["id"], r["v"])] = r["text"]
    return out


def main(out_dir, val_all, *others):
    os.makedirs(out_dir, exist_ok=True)
    sensor = NoveltySensor(f"{R}/runs/m_tiny_router/model", device="cuda" if torch.cuda.is_available() else "cpu")

    def mid_states(texts):
        U = sensor._unit(sensor._z(sensor._states(texts)))
        return U[:, sensor.mid]                                      # (N, mid layers, D)

    vdir = os.path.dirname(val_all)
    val_files = [os.path.join(vdir, f"facts_{k}.jsonl") for k in SOURCES if k.startswith("val_")]
    val_rows, val_txt = [], {}
    for vf in val_files:
        val_rows += [(vf, json.loads(l)) for l in open(vf, encoding="utf-8")]
        val_txt.update({(vf,) + k: t for k, t in texts_for(vf).items()})
    V = mid_states([val_txt[(vf, r["id"], r["v"])] for vf, r in val_rows])
    ans = np.array([r["dept_true"] for _, r in val_rows])
    parity = np.array([r["id"] % 2 for _, r in val_rows])

    def vote(Q, mem, mem_ans):
        sims = np.mean([Q[:, l] @ mem[:, l].T for l in range(Q.shape[1])], 0)
        top = np.argsort(-sims, -1)[:, :5]
        return [Counter(mem_ans[t]).most_common(1)[0][0] for t in top], np.sort(sims, -1)[:, -5:].mean(-1)

    # validation, fold-safe: each half voted by the other half's memory
    v_vote, v_close = [None] * len(val_rows), np.zeros(len(val_rows))
    for half in (0, 1):
        q, m = parity == half, parity != half
        vv, cc = vote(V[q], V[m], ans[m])
        for i, (a, c) in zip(np.where(q)[0], zip(vv, cc)):
            v_vote[i], v_close[i] = a, c
    by_file = {}
    for (vf, r), a, c in zip(val_rows, v_vote, v_close):
        r = dict(r, memory_agrees=1.0 if a == r["dept_seesaw"] else -1.0, memory_close=round(float(c), 5), memory_vote=a)
        by_file.setdefault(vf, []).append(r)
    allv = []
    for vf, rows in by_file.items():
        with open(os.path.join(out_dir, os.path.basename(vf)), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        allv += rows
    with open(os.path.join(out_dir, "facts_val_all.jsonl"), "w", encoding="utf-8") as fh:
        for r in allv:
            fh.write(json.dumps(r) + "\n")
    # any other file: memory = all validation cases
    for f in others:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        txt = texts_for(f)
        Q = mid_states([txt[(r["id"], r["v"])] for r in rows])
        vv, cc = vote(Q, V, ans)
        with open(os.path.join(out_dir, os.path.basename(f)), "w", encoding="utf-8") as fh:
            for r, a, c in zip(rows, vv, cc):
                fh.write(json.dumps(dict(r, memory_agrees=1.0 if a == r["dept_seesaw"] else -1.0,
                                         memory_close=round(float(c), 5), memory_vote=a)) + "\n")
        print("augmented", os.path.basename(f), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])

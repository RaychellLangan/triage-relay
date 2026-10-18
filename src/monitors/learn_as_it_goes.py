"""LEARN AS IT GOES (an open, ordinary-methods take on learning from resolved cases): resolved cases are REMEMBERED WITH THEIR ANSWER.

usage: run_env.sh monitors/learn_as_it_goes.py <out.json> <messy file> [<messy file> ...]

The stream: the files' cases shuffled together. The first half is THE PAST: each case was handed to a person, who resolved it
(the recorded department); its middle-layer state is remembered WITH that answer. The second half is NOW, read twice:
  BEFORE   with the memory bank of ordinary training cases only
  AFTER    with the past's resolved cases added
Reported on NOW:
  mean novelty before / after                (does the past make similar cases look familiar?)
  novelty AUROC for the seesaw's mistakes, before / after   (familiar is not the same as right)
  MEMORY VOTE: the department most common among the 5 nearest REMEMBERED past cases (their resolved answers), and its accuracy
  on all of NOW and on the cases the seesaw gets wrong, next to the seesaw's and the router's accuracy on those cases.
Nothing is trained: the memory is a list of states with answers, read by nearest neighbours.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "route_from_fields"))
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
import common                                  # noqa: E402
from trajectory import auroc                   # noqa: E402
from novelty_sensor import NoveltySensor       # noqa: E402
from seesaw_router import FieldRouter          # noqa: E402
from what_did_it_learn import Router           # noqa: E402

R = ROOT + ""


def main(out_path, *files):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sensor = NoveltySensor(f"{R}/runs/m_tiny_router/model", device=dev)
    router = Router(f"{R}/runs/m_tiny_router/model", dev)
    seesaw = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    recs = []
    for path in files:
        recs += [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(23).shuffle(recs)
    half = len(recs) // 2
    past, now = recs[:half], recs[half:]

    def seesaw_pick(r):
        return table[seesaw.route(r["codes"].split(), "condition")[0]["label"]]["department"]

    now_text = [r["text"] for r in now]
    truth = np.array([r["dept_true"] for r in now])
    s_pick = np.array([seesaw_pick(r) for r in now])
    s_wrong = s_pick != truth
    common.TASK = "router"
    r_pick = np.array([common.DEPARTMENTS[int(np.argmax(m))] for m in router.mouth(now_text)])

    nov_before, _ = sensor.read(now_text)
    # remember the past WITH answers
    past_U = sensor._unit(sensor._z(sensor._states([r["text"] for r in past])))
    past_ans = np.array([r["dept_true"] for r in past])
    sensor.bank = np.concatenate([sensor.bank, past_U], 0)
    nov_after, _ = sensor.read(now_text)

    now_U = sensor._unit(sensor._z(sensor._states(now_text)))
    sims = np.mean([now_U[:, l] @ past_U[:, l].T for l in sensor.mid], 0)       # (now, past), averaged over middle blocks
    top = np.argsort(-sims, -1)[:, :5]
    vote = np.array([Counter(past_ans[t]).most_common(1)[0][0] for t in top])
    nearest_sim = np.sort(sims, -1)[:, -5:].mean(-1)

    def acc(p, mask=None):
        m = np.ones(len(p), bool) if mask is None else mask
        return round(float((p[m] == truth[m]).mean()), 4) if m.any() else None

    rep = {"files": list(files), "past": len(past), "now": len(now), "seesaw_wrong_now": int(s_wrong.sum()),
           "mean_novelty_before": round(float(nov_before.mean()), 4), "mean_novelty_after": round(float(nov_after.mean()), 4),
           "novelty_AUROC_seesaw_wrong_before": auroc(nov_before, s_wrong),
           "novelty_AUROC_seesaw_wrong_after": auroc(nov_after, s_wrong),
           "accuracy_all_now": {"seesaw": acc(s_pick), "router": acc(r_pick), "memory_vote": acc(vote)},
           "accuracy_on_seesaw_wrong": {"seesaw": acc(s_pick, s_wrong), "router": acc(r_pick, s_wrong), "memory_vote": acc(vote, s_wrong)},
           "memory_vote_when_it_disagrees_with_seesaw": acc(vote, vote != s_pick),
           "share_vote_disagrees_with_seesaw": round(float((vote != s_pick).mean()), 4)}
    # does the memory vote get better the closer the remembered cases are? (no threshold: quartiles of closeness)
    q = np.quantile(nearest_sim, [0.25, 0.5, 0.75])
    bins = np.digitize(nearest_sim, q)
    rep["memory_vote_acc_by_closeness_quartile (far->near)"] = [acc(vote, bins == b) for b in range(4)]
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

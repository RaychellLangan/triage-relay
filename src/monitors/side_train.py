"""SIDE TRAINING ON THE SHAPE OF APPROVED RULES (NEXICOG, 13 Oct: "the transformer part of the engine is allowed to look at the new
rule but isn't allowed to memorize it verbatim... absorb just the shape... this would stop the transformer flagging things the
experts know". Precedent: trained so it could only learn the shape of the thing.)

usage: run_env.sh monitors/side_train.py <out dir>

CORRECTIONS  seesaw-wrong cases from the STRESSED VALIDATION sets (stand-ins for cases a person resolved and a reviewer approved).
SHAPE        for each corrected situation (its recorded condition), OTHER training patients with that condition, in messy form.
             The corrected texts themselves are NEVER trained on.
SIDE LOOP    a copy of the messy tiny router, trained (router task) on shape examples + an equal replay of ordinary training data.
MONITORS     every 25 steps:
               memorisation gap = (answer loss on fresh held-out shape examples) / (answer loss on the exact corrected texts)
                                  a gap growing well above 1 means the exact cases are getting easy faster than the shape: verbatim
               canary           = department accuracy on 1,000 ordinary validation cases
             stop on the first: gap > GAP_BUDGET, canary drop > CANARY_BUDGET, max steps; roll back to the last inside both.
AFTER        on the TEST stress sets: novelty (middle-block, memory of ordinary training states) for test cases whose condition was
             corrected vs the rest, before and after; novelty AUROC for the seesaw's mistakes before and after; router accuracy.
Budgets are stated choices.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import copy
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "route_from_fields"))
import common                                    # noqa: E402
from trajectory import auroc                     # noqa: E402
from seesaw_router import FieldRouter            # noqa: E402
from novelty import read_states                  # noqa: E402

R = ROOT + ""
GAP_BUDGET, CANARY_BUDGET, MAX_STEPS, LR, BATCH = 1.5, 0.01, 400, 1e-4, 32


def answer_loss(model, tok, recs, dev):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(recs), 64):
            ids, att, lab, _ = common.encode_batch(tok, recs[i:i + 64], tok.eos_token_id, dev)
            lg, tg = common.answer_logits(model, ids, att, lab)
            tot += torch.nn.functional.cross_entropy(lg.float(), tg, reduction="sum").item()
            n += len(tg)
    model.train()
    return tot / max(n, 1)


def dept_acc(model, tok, recs, dev):
    model.eval()
    ids_d = common.first_token_ids(tok, common.DEPARTMENTS)
    ok = 0
    with torch.no_grad():
        for i in range(0, len(recs), 64):
            ids, att, _, cue = common.encode_batch(tok, recs[i:i + 64], tok.eos_token_id, dev)
            lg = common.logits_at(model, ids, att, cue)[:, ids_d]
            ok += sum(common.DEPARTMENTS[k] == r["dept_true"] for k, r in zip(lg.argmax(-1).tolist(), recs[i:i + 64]))
    model.train()
    return ok / len(recs)


def novelty_of(model, tok, dev, bank_texts, calib_texts, texts):
    calib, _ = read_states(model, tok, calib_texts, dev)
    bank, _ = read_states(model, tok, bank_texts, dev)
    S, _ = read_states(model, tok, texts, dev)
    mu, sd = calib.mean(0), calib.std(0) + 1e-6
    mid = list(range(2, S.shape[1] - 1))

    def u(X):
        Z = (X - mu) / sd
        return Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-9)
    B, Q = u(bank), u(S)
    return np.mean([1 - np.sort(Q[:, l] @ B[:, l].T, -1)[:, -5:].mean(-1) for l in mid], 0)


def pattern_shapes(corr, rng, n_train=10, n_held=2):
    """v2 shape = the corrected case's PATTERN: its condition + the stray fields that do not belong to that condition, given to
    NEW training patients with the same condition, answers dropped like the case's, written messily. Never the case itself."""
    import ast, csv
    sys.path.insert(0, os.path.join(HERE, "..", "data"))
    import convert_ddxplus as conv
    from build_messy_data import short_phrase, typo, code_key, SEPS
    evid = conv.load_evidences(f"{R}/data_raw/ddxplus_en/release_evidences.json")
    conds = json.load(open(f"{R}/data_raw/ddxplus_en/release_conditions.json", encoding="utf-8"))
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    want = defaultdict(int)
    for r in corr:
        want[r["pathology"]] += n_train + n_held
    pool = defaultdict(list)
    with open(f"{R}/data_raw/ddxplus_en_unzipped/release_train_patients", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            p = row["PATHOLOGY"]
            if len(pool[p]) < want.get(p, 0) and rng.random() < 0.2:
                pool[p].append(row)
    used = defaultdict(int)

    def render(row, strays, keep_p):
        groups = {}
        for item in ast.literal_eval(row["EVIDENCES"]):
            groups.setdefault(item.partition("_@_")[0], []).append(item)
        kept = {n: its for n, its in groups.items() if n == row["INITIAL_EVIDENCE"] or rng.random() < keep_p}
        for e in strays:
            kept.setdefault(e.partition("_@_")[0], []).append(e)
        bits = []
        for name, its in kept.items():
            ev = evid[name]
            vals = [conv.answer_text(ev, it.partition("_@_")[2]) for it in its if "_@_" in it]
            bits.append(" ".join(typo(w, rng) for w in (short_phrase(ev["question_en"]) + (" " + ", ".join(vals) if vals else "")).split()))
        rng.shuffle(bits)
        text = f"{row['AGE']}{row['SEX'].lower()} " + "".join(b + rng.choice(SEPS) for b in bits).rstrip(" ,;.") + "."
        lab = conv.labels(row, table)
        return dict(lab, text=text, codes=" ".join(sorted((i for its in kept.values() for i in its), key=code_key)))

    tr, he = [], []
    for r in corr:
        own = set(conds[r["pathology"]]["symptoms"]) | set(conds[r["pathology"]]["antecedents"])
        strays = [c for c in r["codes"].split() if c.partition("_@_")[0] not in own]
        rows = pool[r["pathology"]][used[r["pathology"]]:used[r["pathology"]] + n_train + n_held]
        used[r["pathology"]] += n_train + n_held
        for k, row in enumerate(rows):
            (tr if k < n_train else he).append(render(row, strays, r["keep_p"]))
    return tr, he


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    common.TASK = "router"
    base_dir = f"{R}/runs/m_tiny_router/model"
    tok = AutoTokenizer.from_pretrained(base_dir)
    base = GPT2LMHeadModel.from_pretrained(base_dir).to(dev)
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    seesaw = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")

    # corrections: seesaw-wrong STRESSED-VALIDATION cases (their texts are never trained on)
    corr = []
    for s in ("strays5", "keep30", "keep50_strays3"):
        for l in open(f"{R}/data_built/messyval_{s}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            pick = table[seesaw.route(r["codes"].split(), "condition")[0]["label"]]["department"]
            if pick != r["dept_true"]:
                corr.append(r)
    corrected_conditions = sorted({r["pathology"] for r in corr})
    rng = random.Random(41)
    replay = []
    for l in open(f"{R}/data_built/messy_train.jsonl", encoding="utf-8"):
        if rng.random() < 0.01 and len(replay) < 6000:
            replay.append(json.loads(l))
    if os.environ.get("SHAPE", "pattern") == "condition":
        # v1 shape: other training patients with the same condition (too coarse: see RESULTS_LOG 13 Oct)
        by_cond = defaultdict(list)
        for l in open(f"{R}/data_built/messy_train.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["pathology"] in corrected_conditions and len(by_cond[r["pathology"]]) < 120 and rng.random() < 0.05:
                by_cond[r["pathology"]].append(r)
        shape_train, shape_held = [], []
        for c, rows in by_cond.items():
            shape_train += rows[:100]
            shape_held += rows[100:]
    else:
        shape_train, shape_held = pattern_shapes(corr, rng)
    val = [json.loads(l) for l in open(f"{R}/data_built/messy_validate.jsonl", encoding="utf-8")][:1000]
    print(f"corrections {len(corr)} over {len(corrected_conditions)} conditions | shape train {len(shape_train)} | "
          f"shape held-out {len(shape_held)} | replay {len(replay)}", flush=True)

    model = copy.deepcopy(base).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    canary0 = dept_acc(model, tok, val, dev)
    log, good, why = [], (copy.deepcopy(model.state_dict()), 0), "max steps"
    for step in range(1, MAX_STEPS + 1):
        batch = rng.sample(shape_train, BATCH // 2) + rng.sample(replay, BATCH // 2)
        ids, att, lab, _ = common.encode_batch(tok, batch, tok.eos_token_id, dev)
        lg, tg = common.answer_logits(model, ids, att, lab)
        loss = torch.nn.functional.cross_entropy(lg.float(), tg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            exact, held = answer_loss(model, tok, corr, dev), answer_loss(model, tok, shape_held, dev)
            gap = held / max(exact, 1e-6)
            canary = dept_acc(model, tok, val, dev)
            row = {"step": step, "loss_exact_corrected": round(exact, 4), "loss_heldout_shape": round(held, 4),
                   "memorisation_gap": round(gap, 3), "canary_acc": round(canary, 4)}
            log.append(row)
            print(row, flush=True)
            if gap > GAP_BUDGET:
                why = "memorisation gap over budget"
                break
            if canary0 - canary > CANARY_BUDGET:
                why = "canary dropped"
                break
            good = (copy.deepcopy(model.state_dict()), step)
    model.load_state_dict(good[0])
    model.eval()
    model.save_pretrained(os.path.join(out_dir, "model"))
    tok.save_pretrained(os.path.join(out_dir, "model"))
    print("stopped:", why, "| kept step", good[1], flush=True)

    # AFTER: on the TEST stress sets
    test = []
    for s in ("strays5", "keep30", "keep50_strays3"):
        test += [json.loads(l) for l in open(f"{R}/data_built/messystress_{s}.jsonl", encoding="utf-8")]
    known = np.array([r["pathology"] in corrected_conditions for r in test])
    wrong = np.array([table[seesaw.route(r["codes"].split(), "condition")[0]["label"]]["department"] != r["dept_true"] for r in test])
    calib_t = [r["text"] for r in replay[:1000]]
    bank_t = [r["text"] for r in replay[1000:5000]]
    texts = [r["text"] for r in test]
    rep = {"corrections": len(corr), "corrected_conditions": corrected_conditions, "stopped": why, "kept_step": good[1],
           "canary_start": round(canary0, 4), "log": log, "after": {}}
    for name, m in (("before", base.eval()), ("after", model)):
        nov = novelty_of(m, tok, dev, bank_t, calib_t, texts)
        acc = dept_acc(m, tok, test, dev)
        rep["after"][name] = {"novelty_mean_known_conditions": round(float(nov[known].mean()), 4),
                              "novelty_mean_other_conditions": round(float(nov[~known].mean()), 4),
                              "novelty_AUROC_seesaw_wrong": auroc(nov, wrong),
                              "novelty_AUROC_seesaw_wrong_within_known": auroc(nov[known], wrong[known]),
                              "router_acc_test_stress": round(acc, 4)}
        print(name, rep["after"][name], flush=True)
    json.dump(rep, open(os.path.join(out_dir, "side_train.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

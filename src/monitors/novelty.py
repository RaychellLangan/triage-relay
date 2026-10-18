"""Does the transformer SEE that a case is new, on exactly the cases the seesaw cannot resolve?

usage: run_env.sh monitors/novelty.py <model dir> <router|translator> <out.json> <messy file> [<messy file> ...]

TARGETS (from the seesaw on the TRUE fields, i.e. what hand-authored routing cannot resolve):
  plain_wrong      the condition planks' winner is in the wrong department
  severe_missed    danger-first routes a severity 1-2 patient to something less urgent
  fight            the winning condition plank fights
READINGS (nothing learned; all at the cue position, embedding + every block, raw block outputs):
  novelty_L        1 - mean cosine to the 5 NEAREST training states (a bank of 4,000 training variants read the same way,
                   each dimension z-scored on a separate 1,000 training variants): "has it seen anything like this?"
  znorm_L          length of the z-scored state (how far from the training cloud's centre)
  mouth_conf       router only: top probability over the 14 department first-tokens
Scored: AUROC of each reading for each target, per layer; plus the seesaw's own fight flag as a reference for the other targets.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys

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

R = ROOT + ""


@torch.no_grad()
def read_states(model, tok, texts, dev, batch=32):
    tap = common.BlockTap(model)
    S, M = [], []
    dept_ids = common.first_token_ids(tok, common.DEPARTMENTS) if common.TASK == "router" else None
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids = [tok(common.prompt_of({"text": t}))["input_ids"][-760:] for t in chunk]
        width = max(len(x) for x in ids)
        x = torch.full((len(ids), width), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros_like(x)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        x, att = x.to(dev), att.to(dev)
        cue = torch.tensor([len(s) - 1 for s in ids], device=dev)
        rows = torch.arange(len(ids), device=dev)
        out = model.transformer(input_ids=x, attention_mask=att, output_hidden_states=True)
        layers = [out.hidden_states[0][rows, cue]] + [tap.out[l][rows, cue] for l in range(len(model.transformer.h))]
        S.append(torch.stack(layers, 1).float().cpu())
        if dept_ids is not None:
            M.append(torch.softmax(model.lm_head(out.last_hidden_state[rows, cue]).float()[:, dept_ids], -1).max(-1).values.cpu())
    tap.remove()
    return torch.cat(S).numpy(), (torch.cat(M).numpy() if M else None)


def main(model_dir, task, out_path, *files):
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    common.TASK = task
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(dev).eval()
    router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
    table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
    sev = {c: table[c]["severity"] for c in table}

    rng = random.Random(17)
    train = []
    for l in open(f"{R}/data_built/messy_train.jsonl", encoding="utf-8"):
        if rng.random() < 0.0125:
            train.append(json.loads(l)["text"])
        if len(train) >= 5000:
            break
    calib_S, _ = read_states(model, tok, train[:1000], dev)
    bank_S, _ = read_states(model, tok, train[1000:5000], dev)
    mu, sd = calib_S.mean(0), calib_S.std(0) + 1e-6
    bank = (bank_S - mu) / sd
    bank_u = bank / (np.linalg.norm(bank, axis=-1, keepdims=True) + 1e-9)

    recs, file_of = [], []
    for path in files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        got = random.Random(3).sample(rows, min(1000, len(rows)))
        recs += got
        file_of += [os.path.basename(path)] * len(got)
    file_of = np.array(file_of)
    S, mouth = read_states(model, tok, [r["text"] for r in recs], dev)
    Z = (S - mu) / sd
    Zu = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-9)
    n_layer = Z.shape[1]
    novelty = np.zeros((len(recs), n_layer))
    for l in range(n_layer):
        sims = Zu[:, l] @ bank_u[:, l].T                       # (N, bank)
        novelty[:, l] = 1 - np.sort(sims, -1)[:, -5:].mean(-1)
    znorm = np.linalg.norm(Z, axis=-1) / np.sqrt(Z.shape[-1])

    targets = {"plain_wrong": [], "severe_missed": [], "fight": []}
    for r in recs:
        planks = router.route(r["codes"].split(), "condition")
        win = planks[0]
        in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
        danger = min(in_play, key=lambda p: (sev[p["label"]], -p["force"]))
        ts = sev[r["pathology"]]
        targets["plain_wrong"].append(table[win["label"]]["department"] != r["dept_true"])
        targets["severe_missed"].append(ts <= 2 and sev[danger["label"]] > ts)
        targets["fight"].append(win["fight"])
    T = {k: np.array(v) for k, v in targets.items()}
    rep = {"model": model_dir, "task": task, "files": list(files), "n": len(recs),
           "target_counts": {k: int(v.sum()) for k, v in T.items()}, "auroc": {}}
    for name, y in T.items():
        a = {"novelty_by_layer": [auroc(novelty[:, l], y) for l in range(n_layer)],
             "znorm_by_layer": [auroc(znorm[:, l], y) for l in range(n_layer)]}
        if mouth is not None:
            a["mouth_low_confidence"] = auroc(-mouth, y)
        if name != "fight":
            a["seesaw_fight_flag (reference)"] = auroc(T["fight"].astype(float), y)
        rep["auroc"][name] = a
        best = int(np.nanargmax([x if x is not None else np.nan for x in a["novelty_by_layer"]]))
        print(f"{name:14s} (n={int(y.sum()):4d})  novelty by layer {a['novelty_by_layer']}  best L{best}"
              + (f"  | mouth {a['mouth_low_confidence']}" if mouth is not None else "")
              + (f"  | seesaw fight {a.get('seesaw_fight_flag (reference)')}" if name != "fight" else ""), flush=True)
    # CONTROL: the same scores WITHIN each file (so a reading cannot win just by telling the stress sets apart)
    rep["within_file"] = {}
    for fname in sorted(set(file_of)):
        mfile = file_of == fname
        rep["within_file"][fname] = {}
        for name, y in T.items():
            yy = y[mfile]
            best_l = int(np.argmax([auroc(novelty[mfile, l], yy) or 0 for l in range(n_layer)]))
            rep["within_file"][fname][name] = {
                "n_target": int(yy.sum()),
                "novelty_best_layer": best_l, "novelty": auroc(novelty[mfile, best_l], yy),
                "mouth_low_confidence": auroc(-mouth[mfile], yy) if mouth is not None else None,
                "seesaw_fight": auroc(T["fight"][mfile].astype(float), yy) if name != "fight" else None}
            print("   within", fname, name, rep["within_file"][fname][name], flush=True)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Readings only (no judgements): what every layer of a router holds at the cue, before it writes anything.

usage: run_env.sh monitors/readings.py <model dir> <out.npz> [n_eval] [n_fit]

1. CENTRES. n_fit training patients (full intake) go through the router. At the cue position ("Route:"), at the embedding
   and after every block (raw block output, before the final layer-norm), each state is DE-ROGUED: every dimension
   z-scored with mean/sd from a separate CALIBRATION sample (fold-safe). Then, per layer, optionally the top-k shared
   principal directions of the calibration states are projected out (the TAIL), k in KS; k=0 is always kept.
   A department's centre = the mean of its patients' states, unit length. Nothing is trained; centres are averages.
2. EVAL. n_eval test patients under every stress (probes/stresses.py), plus out-of-domain texts. For each:
     cos[layer, k, dept]   angle between the state and each department's centre
     mouth[dept]           restricted softmax over the 14 department first-tokens at the cue (what it is about to say)
     vocab[dept]           the same centre read on MEAN-POOLED token embeddings of the prompt (the vocabulary baseline:
                           if this routes as well as the middle, the middle adds nothing)
     znorm[layer]          length of the de-rogued state (how far from the calibration cloud)
3. A held-out slice of the fit patients scores each (layer, k) as a plain nearest-centre router: fit_acc[layer, k].
   That table is where k is chosen; the eval patients never choose anything.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import csv
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import common                       # noqa: E402
import convert_ddxplus as conv      # noqa: E402
import stresses                     # noqa: E402

RAW = ROOT + "/data_raw"
TABLE = os.path.join(HERE, "..", "labels", "routing_table.json")
KS = (0, 1, 2, 4, 8)


def reservoir(path, n, seed):
    rng, keep = random.Random(seed), []
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < n:
                keep.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    keep[j] = row
    return keep


class Reader:
    def __init__(self, model_dir, device):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(device).eval()
        self.dev = device
        self.eos = self.tok.eos_token_id
        self.dept_ids = common.first_token_ids(self.tok, common.DEPARTMENTS)
        self.n_layers = len(self.model.transformer.h)

    @torch.no_grad()
    def states(self, texts, batch=32):
        """-> layer states (N, 1+n_layers, D) at the cue [embedding, block0..], mouth (N, 14), pooled embeddings (N, D)"""
        tap = common.BlockTap(self.model)
        S, M, V = [], [], []
        wte = self.model.transformer.wte.weight
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ids = [self.tok(t + common.ROUTE_CUE)["input_ids"][-760:] for t in chunk]
            width = max(len(x) for x in ids)
            x = torch.full((len(ids), width), self.eos, dtype=torch.long)
            att = torch.zeros_like(x)
            for j, s in enumerate(ids):
                x[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            x, att = x.to(self.dev), att.to(self.dev)
            cue = torch.tensor([len(s) - 1 for s in ids], device=self.dev)
            rows = torch.arange(len(ids), device=self.dev)
            out = self.model.transformer(input_ids=x, attention_mask=att, output_hidden_states=True)
            emb = out.hidden_states[0][rows, cue]
            layers = [emb] + [tap.out[l][rows, cue] for l in range(self.n_layers)]
            S.append(torch.stack(layers, 1).float().cpu())
            M.append(torch.softmax(self.model.lm_head(out.last_hidden_state[rows, cue]).float()[:, self.dept_ids], -1).cpu())
            pooled = (wte[x] * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True)
            V.append(pooled.float().cpu())
        tap.remove()
        return torch.cat(S).numpy(), torch.cat(M).numpy(), torch.cat(V).numpy()


def tail_basis(calib_z, k):
    """top-k principal directions of the calibration states (rows = directions)"""
    if k == 0:
        return None
    c = calib_z - calib_z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[:k]


def project_out(x, basis):
    return x if basis is None else x - (x @ basis.T) @ basis


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main(model_dir, out_path, n_eval=500, n_fit=4000):
    n_eval, n_fit = int(n_eval), int(n_fit)
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    reader = Reader(model_dir, dev)
    evid = conv.load_evidences(f"{RAW}/ddxplus_en/release_evidences.json")
    conds = json.load(open(f"{RAW}/ddxplus_en/release_conditions.json", encoding="utf-8"))
    table = json.load(open(TABLE, encoding="utf-8"))
    st = stresses.Stresses(evid, conds, table)
    dept_of = lambda row: common.DEPARTMENTS.index(table[row["PATHOLOGY"]]["department"])   # noqa: E731

    # fit + calibration patients come from TRAIN; eval patients from TEST
    pool = reservoir(f"{RAW}/ddxplus_en_unzipped/release_train_patients", n_fit + 1000, seed=21)
    calib_rows, fit_rows = pool[:1000], pool[1000:]
    calib_S, _, calib_V = reader.states([st.full(r) for r in calib_rows])
    fit_S, _, fit_V = reader.states([st.full(r) for r in fit_rows])
    fit_y = np.array([dept_of(r) for r in fit_rows])
    n_layer = calib_S.shape[1]

    mu, sd = calib_S.mean(0), calib_S.std(0) + 1e-6                       # (L, D) each
    vmu, vsd = calib_V.mean(0), calib_V.std(0) + 1e-6
    calib_Z = (calib_S - mu) / sd
    bases = [[tail_basis(calib_Z[:, l], k) for k in KS] for l in range(n_layer)]

    # centres from the first 75% of fit, nearest-centre accuracy on the last 25% (where k gets chosen)
    cut = int(len(fit_rows) * 0.75)
    fit_Z = (fit_S - mu) / sd
    centres = np.zeros((n_layer, len(KS), 14, fit_S.shape[2]), dtype=np.float32)
    fit_acc = np.zeros((n_layer, len(KS)))
    for l in range(n_layer):
        for ki in range(len(KS)):
            z = project_out(fit_Z[:, l], bases[l][ki])
            for d in range(14):
                members = z[:cut][fit_y[:cut] == d]
                if len(members):
                    centres[l, ki, d] = unit(members.mean(0))
            pred = (unit(z[cut:]) @ centres[l, ki].T).argmax(-1)
            fit_acc[l, ki] = (pred == fit_y[cut:]).mean()
    vz = (fit_V - vmu) / vsd
    vcent = np.stack([unit(vz[:cut][fit_y[:cut] == d].mean(0)) if (fit_y[:cut] == d).any() else np.zeros(vz.shape[1])
                      for d in range(14)])
    vocab_fit_acc = float(((unit(vz[cut:]) @ vcent.T).argmax(-1) == fit_y[cut:]).mean())
    print("fit_acc (rows = embedding, block0.., cols = k in", KS, ")\n", np.round(fit_acc, 3),
          "\nvocabulary baseline fit acc", round(vocab_fit_acc, 3), flush=True)

    def read(texts):
        S, M, V = reader.states(texts)
        Z = (S - mu) / sd
        cos = np.zeros((len(texts), n_layer, len(KS), 14), dtype=np.float32)
        for l in range(n_layer):
            for ki in range(len(KS)):
                cos[:, l, ki] = unit(project_out(Z[:, l], bases[l][ki])) @ centres[l, ki].T
        vocab = unit((V - vmu) / vsd) @ vcent.T
        znorm = np.linalg.norm(Z, axis=-1) / np.sqrt(Z.shape[-1])
        return cos, M, vocab.astype(np.float32), znorm.astype(np.float32)

    eval_rows = reservoir(f"{RAW}/ddxplus_en_unzipped/release_test_patients", n_eval, seed=7)
    truth = np.array([dept_of(r) for r in eval_rows])
    diff = np.array([[json.loads(json.dumps(conv.labels(r, table)["dept_mass"])).get(d, 0.0) for d in common.DEPARTMENTS]
                     for r in eval_rows], dtype=np.float32)
    save = {"fit_acc": fit_acc, "vocab_fit_acc": np.array(vocab_fit_acc), "ks": np.array(KS), "truth": truth, "diff": diff}
    variants = [st.patient_variants(r) for r in eval_rows]
    for name in variants[0]:
        cos, M, V, zn = read([v[name] for v in variants])
        save[f"{name}__cos"], save[f"{name}__mouth"], save[f"{name}__vocab"], save[f"{name}__znorm"] = cos, M, V, zn
        print(name, "mouth acc", round(float((M.argmax(-1) == truth).mean()), 3), flush=True)
    ood = {"ood_english": [st.ood_english(i) for i in range(len(stresses.PASSAGES))],
           "ood_wordsalad": [st.ood_wordsalad(r) for r in eval_rows[:200]]}
    for name, texts in ood.items():
        cos, M, V, zn = read(texts)
        save[f"{name}__cos"], save[f"{name}__mouth"], save[f"{name}__vocab"], save[f"{name}__znorm"] = cos, M, V, zn
    np.savez_compressed(out_path, **save)
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])

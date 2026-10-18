"""Shared pieces for both routers: the data, the answer format, the tokenizers, and the readings taken while training.

ANSWER FORMAT. Every patient becomes

    <intake text>
    Route: <department>
    Urgency: <urgency><eos>

and the model is trained (loss on the answer tokens only) to write the last two lines. Each department and each urgency
starts with its own first token in both tokenizers (checked by data_census/first_tokens.py), so "what the model is about
to say" is ONE next-token choice: the logits at the last prompt token, restricted to the 14 department first-tokens.
That restricted softmax is THE MOUTH. Monitors read the middle layers at that same position, before anything is written.
"""
import json
import math
import random

import numpy as np
import torch

DEPARTMENTS = ["Cardiology", "ENT", "Emergency (resus)", "Gastroenterology", "General medicine", "Haematology",
               "Infectious diseases", "Mental health", "Neurology", "Oncology", "Paediatrics", "Respiratory",
               "Rheumatology", "Trauma and orthopaedics"]
URGENCIES = ["Immediate", "Very urgent", "Urgent", "Standard", "Non-urgent"]
ROUTE_CUE = "\nRoute:"
URGENCY_CUE = "\nUrgency:"


FIELDS_CUE = "\nFields:"
TASK = "router"          # set by train.py: "router" (department + urgency) or "translator" (the patient's field codes)


def prompt_of(rec):
    return rec["text"] + (ROUTE_CUE if TASK == "router" else FIELDS_CUE)


def answer_of(rec):
    if TASK == "translator":
        return " " + rec["codes"]
    return " " + rec["dept_true"] + URGENCY_CUE + " " + rec["urgency_true"]


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class JsonlLines:
    """Random access into a big .jsonl without loading it: one pass to record line offsets, then seek."""

    def __init__(self, path):
        self.path = path
        offsets, pos = [], 0
        with open(path, "rb") as f:
            for line in f:
                offsets.append(pos)
                pos += len(line)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self._f = open(path, "rb")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        self._f.seek(int(self.offsets[i]))
        return json.loads(self._f.readline())

    def sample(self, n, seed):
        rng = random.Random(seed)
        return [self[i] for i in rng.sample(range(len(self)), n)]


def encode_batch(tok, recs, eos_id, device, max_len=768):
    """-> input_ids, attention_mask, labels (-100 on the prompt), and the index of the last prompt token per row."""
    rows, label_rows, cue_pos = [], [], []
    for r in recs:
        p = tok(prompt_of(r))["input_ids"]
        a = tok(answer_of(r), add_special_tokens=False)["input_ids"] + [eos_id]   # no start-of-text token mid-sequence
        ids = (p + a)[:max_len]
        rows.append(ids)
        label_rows.append(([-100] * len(p) + a)[:max_len])
        cue_pos.append(len(p) - 1)
    width = max(len(x) for x in rows)
    pad = eos_id
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    att = torch.zeros((len(rows), width), dtype=torch.long)
    lab = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(rows, label_rows)):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
        lab[i, :len(y)] = torch.tensor(y)
    return ids.to(device), att.to(device), lab.to(device), torch.tensor(cue_pos, device=device)


def answer_logits(model, ids, att, lab):
    """Vocabulary logits ONLY where they are needed: at the positions that predict an answer token.
    (Full-sequence logits are batch x length x 50,257; in float32 that is ~3 GB per batch on GPT-2 and it spilled the
    12 GB card into system RAM and crashed a run on 13 Oct.) -> (logits for the answer positions, their target ids)"""
    h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    m = tgt != -100
    return model.lm_head(h[:, :-1][m]), tgt[m]


def logits_at(model, ids, att, positions):
    """Vocabulary logits at one position per row (e.g. the cue). -> (batch, vocab)"""
    h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
    return model.lm_head(h[torch.arange(len(positions), device=h.device), positions])


def first_token_ids(tok, names):
    ids = [tok(" " + n, add_special_tokens=False)["input_ids"][0] for n in names]   # (Gemma would put <bos> first)
    assert len(set(ids)) == len(ids), "first tokens collide"
    return ids


# ── tiny model's own word-level tokenizer ────────────────────────────────────────────────────────────────────────────
def build_word_tokenizer(texts, out_dir):
    """Word-level vocabulary built from the training intake texts and the answer strings. Small and exact."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    extra = [ROUTE_CUE, URGENCY_CUE] + [" " + d for d in DEPARTMENTS] + [" " + u for u in URGENCIES]
    t.train_from_iterator(list(texts) + extra, trainers.WordLevelTrainer(special_tokens=["[UNK]", "[EOS]"], min_frequency=1))
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
    fast.save_pretrained(out_dir)
    return fast


def blocks_of(model):
    """the transformer blocks, whatever the family calls them (GPT-2: transformer.h; Gemma/Llama-style: model.layers)"""
    return model.transformer.h if hasattr(model, "transformer") else model.model.layers


def block_prefix(model):
    return "transformer.h." if hasattr(model, "transformer") else "model.layers."


# ── readings taken while training (no decisions in here) ─────────────────────────────────────────────────────────────
class BlockTap:
    """Forward hooks on every transformer block: the RAW block output (before the final layer-norm), so every layer is
    read the same way. (Trap from the layer safaris: HF's last hidden_state already has ln_f applied.)"""

    def __init__(self, model):
        self.out = {}
        self.handles = [blk.register_forward_hook(self._hook(i)) for i, blk in enumerate(blocks_of(model))]

    def _hook(self, i):
        def f(_m, _a, output):
            self.out[i] = output[0] if isinstance(output, tuple) else output
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def spectrum_readings(h, top_k=3):
    """h: (N, D) states. -> dark-matter ratio (energy outside the top-k principal directions) and participation ratio
    (how many directions really carry the spread)."""
    h = h.float()
    h = h - h.mean(0, keepdim=True)
    s = torch.linalg.svdvals(h)
    lam = s ** 2
    total = lam.sum().item() + 1e-12
    dark = 1.0 - lam[:top_k].sum().item() / total
    pr = (lam.sum().item() ** 2) / ((lam ** 2).sum().item() + 1e-12)
    return dark, pr


@torch.no_grad()
def measure(model, tok, val_recs, probe_recs, dept_ids, urg_ids, eos_id, device, canary_ids=None, batch=32):
    """Everything recorded at a measurement. Returns a dict of plain numbers."""
    model.eval()
    out = {}
    if TASK == "translator":
        out.update(_validate_translator(model, tok, val_recs, eos_id, device, batch))
        return _geometry_and_canary(model, tok, probe_recs, eos_id, device, canary_ids, batch, out)
    # validation: answer loss, mouth accuracy, mouth entropy, and distance of the mouth from the ground-truth differential
    tot_loss, tot_tok, dept_ok, urg_ok, ent, kl = 0.0, 0, 0, 0, 0.0, 0.0
    for i in range(0, len(val_recs), batch):
        recs = val_recs[i:i + batch]
        ids, att, lab, cue = encode_batch(tok, recs, eos_id, device)
        h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
        tgt = lab[:, 1:]
        m = tgt != -100
        alog = model.lm_head(h[:, :-1][m]).float()
        tot_loss += torch.nn.functional.cross_entropy(alog, tgt[m], reduction="sum").item()
        tot_tok += m.sum().item()
        rows = torch.arange(len(recs), device=device)
        mouth = torch.softmax(model.lm_head(h[rows, cue]).float()[:, dept_ids], -1)
        truth = torch.tensor([DEPARTMENTS.index(r["dept_true"]) for r in recs], device=device)
        dept_ok += (mouth.argmax(-1) == truth).sum().item()
        ent += -(mouth * mouth.clamp_min(1e-12).log()).sum(-1).sum().item()
        diff = torch.tensor([[r["dept_mass"].get(d, 0.0) for d in DEPARTMENTS] for r in recs], device=device)
        kl += (diff * (diff.clamp_min(1e-12).log() - mouth.clamp_min(1e-12).log())).sum(-1).sum().item()
        # urgency is read one step later, with the TRUE department written (teacher forcing)
        upos = []
        for r, c in zip(recs, cue.tolist()):
            upos.append(c + len(tok(" " + r["dept_true"] + URGENCY_CUE, add_special_tokens=False)["input_ids"]))
        upos = torch.tensor(upos, device=device)
        um = model.lm_head(h[rows, upos]).float()[:, urg_ids]
        utruth = torch.tensor([URGENCIES.index(r["urgency_true"]) for r in recs], device=device)
        urg_ok += (um.argmax(-1) == utruth).sum().item()
    n = len(val_recs)
    out.update(val_answer_loss=tot_loss / max(tot_tok, 1), val_dept_acc=dept_ok / n, val_urgency_acc=urg_ok / n,
               val_mouth_entropy_bits=ent / n / math.log(2), val_kl_differential_to_mouth=kl / n)
    return _geometry_and_canary(model, tok, probe_recs, eos_id, device, canary_ids, batch, out)


def _validate_translator(model, tok, val_recs, eos_id, device, batch):
    """Teacher-forced: answer loss, share of answer tokens predicted right, and share of patients whose WHOLE field list is
    predicted right token by token (a strict exact-match; the free-running field F1 is scored by the eval script)."""
    tot_loss, tot_tok, tok_ok, exact = 0.0, 0, 0, 0
    for i in range(0, len(val_recs), batch):
        recs = val_recs[i:i + batch]
        ids, att, lab, _ = encode_batch(tok, recs, eos_id, device)
        h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
        tgt = lab[:, 1:]
        m = tgt != -100
        logits = model.lm_head(h[:, :-1][m]).float()
        tot_loss += torch.nn.functional.cross_entropy(logits, tgt[m], reduction="sum").item()
        tot_tok += m.sum().item()
        right = (logits.argmax(-1) == tgt[m])
        tok_ok += right.sum().item()
        per_row = torch.zeros_like(tgt, dtype=torch.bool)
        per_row[m] = ~right
        exact += (~per_row.any(1)).sum().item()
    n = len(val_recs)
    return {"val_answer_loss": tot_loss / max(tot_tok, 1), "val_token_acc": tok_ok / max(tot_tok, 1), "val_exact": exact / n}


def _geometry_and_canary(model, tok, probe_recs, eos_id, device, canary_ids, batch, out):
    # geometry at the cue position: dark-matter ratio + participation ratio per layer (raw block outputs).
    # RAW and DE-ROGUED. A few huge dimensions swallow the top components of a raw GPT-2 state, so the raw ratio mostly
    # measures them. De-rogued = each dimension z-scored with mean/sd from a SEPARATE calibration sample (fold-safe:
    # normalising with the rows being measured turns noise into signal), recomputed at every measurement.
    probe, calib = probe_recs[0], probe_recs[1]
    tap = BlockTap(model)

    def collect(recs_all):
        states = {}
        for i in range(0, len(recs_all), batch):
            recs = recs_all[i:i + batch]
            ids, att, _, cue = encode_batch(tok, recs, eos_id, device)
            model(input_ids=ids, attention_mask=att)
            rows = torch.arange(len(recs), device=device)
            for layer, h in tap.out.items():
                states.setdefault(layer, []).append(h[rows, cue].float().cpu())
        return {k: torch.cat(v) for k, v in states.items()}

    sp, sc = collect(probe), collect(calib)
    tap.remove()
    for layer in sorted(sp):
        dark, pr = spectrum_readings(sp[layer])
        mu, sd = sc[layer].mean(0, keepdim=True), sc[layer].std(0, keepdim=True) + 1e-6
        darkz, prz = spectrum_readings((sp[layer] - mu) / sd)
        out[f"dark_L{layer}"] = round(dark, 5)
        out[f"pr_L{layer}"] = round(pr, 3)
        out[f"darkz_L{layer}"] = round(darkz, 5)
        out[f"prz_L{layer}"] = round(prz, 3)

    # English canary (GPT-2 arms): plain LM loss on passages it never trains on
    if canary_ids is not None:
        losses = [model(input_ids=c, labels=c).loss.item() for c in canary_ids]
        out["english_loss"] = sum(losses) / len(losses)
    model.train()
    return out


def relative_distance(model, start):
    """How far each part has moved from where it started, as a share of its starting size."""
    emb = model.get_input_embeddings().weight
    emb_name = next(n for n, p in model.named_parameters() if p is emb)
    wte = emb.detach().float().cpu()
    d_wte = (wte - start[emb_name]).norm().item() / (start[emb_name].norm().item() + 1e-12)
    blocks = []
    prefix = block_prefix(model)
    for name, p in model.named_parameters():
        if name.startswith(prefix) and name in start:
            s = start[name]
            blocks.append((p.detach().float().cpu() - s).norm().item() / (s.norm().item() + 1e-12))
    return d_wte, (sum(blocks) / len(blocks) if blocks else 0.0)

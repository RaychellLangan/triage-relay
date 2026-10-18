"""Where does the router LOOK when it decides? Attention from the cue position ("Route:"), per layer, by token type.

usage: run_env.sh probes/where_it_looks.py <model dir> <out.json> [n_patients]

Every intake is rebuilt with each character tagged:
  sink      the first token (attention sinks pile there and it carries nothing about the patient)
  header    "Age", "Sex", "Reason for visit", "Symptoms:", "History:", "- " and the line breaks
  question  the wording of each question
  yes       a bare "yes" answer
  value     a real answer value (a word or a number the patient gave)
  empty     an answer that means none/no ("N", "NA", "nowhere")
  demog     the age and sex values
  cue       "Route:" itself
For every layer: the share of the cue's attention (averaged over heads and patients) landing on each type, and that share
divided by the type's share of the tokens (above 1 = read more than its size; below 1 = mostly ignored).
Pure transformer measurement; nothing is changed in the model.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import ast
import csv
import json
import os
import random
import sys
from collections import defaultdict

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import common                       # noqa: E402
import convert_ddxplus as conv      # noqa: E402

RAW = ROOT + "/data_raw"
EMPTY = {"N", "NA", "NOWHERE"}
TYPES = ("sink", "header", "question", "yes", "value", "empty", "demog", "cue")


def tagged_text(row, evid):
    """The same text convert_ddxplus.patient_text writes, as (piece, type) parts."""
    groups = {}
    for item in ast.literal_eval(row["EVIDENCES"]):
        name, _, value = item.partition("_@_")
        groups.setdefault(name, [])
        if value:
            groups[name].append(value)

    def line(name, lead):
        ev = evid[name]
        parts = [(lead, "header"), (ev["question_en"].strip(), "question"), (" ", "header")]
        if ev["data_type"] == "B" or not groups[name]:
            return parts + [("yes", "yes")]
        for i, v in enumerate(groups[name]):
            word = conv.answer_text(ev, v)
            parts += ([(", ", "header")] if i else []) + [(word, "empty" if word.strip().upper() in EMPTY else "value")]
        return parts

    init = row["INITIAL_EVIDENCE"]
    parts = [("Age ", "header"), (row["AGE"], "demog"), (". Sex ", "header"), (row["SEX"], "demog"), (".", "header"),
             ("\nReason for visit: ", "header")]
    parts += line(init, "") if init in groups else [(evid[init]["question_en"].strip(), "question"), (" ", "header"), ("yes", "yes")]
    sym = [n for n in groups if n != init and not evid[n]["is_antecedent"]]
    his = [n for n in groups if n != init and evid[n]["is_antecedent"]]
    if sym:
        parts.append(("\nSymptoms:", "header"))
        for n in sym:
            parts += line(n, "\n- ")
    if his:
        parts.append(("\nHistory:", "header"))
        for n in his:
            parts += line(n, "\n- ")
    parts.append((common.ROUTE_CUE, "cue"))
    return parts


def main(model_dir, out_path, n=300):
    n = int(n)
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, attn_implementation="eager").to(dev).eval()
    evid = conv.load_evidences(f"{RAW}/ddxplus_en/release_evidences.json")
    rng, rows = random.Random(7), []
    with open(f"{RAW}/ddxplus_en_unzipped/release_test_patients", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < n:
                rows.append(row)
            elif rng.randint(0, i) < n:
                rows[rng.randint(0, n - 1)] = row
    n_layers = len(model.transformer.h)
    att_mass = [defaultdict(float) for _ in range(n_layers)]
    tok_share = defaultdict(float)
    check_equal = 0
    for row in rows:
        parts = tagged_text(row, evid)
        text = "".join(p for p, _ in parts)
        check_equal += (text == conv.patient_text(row, evid) + common.ROUTE_CUE)
        spans, pos = [], 0
        for p, t in parts:
            spans.append((pos, pos + len(p), t))
            pos += len(p)
        enc = tok(text, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        types = []
        for j, (a, b) in enumerate(offs):
            if j == 0:
                types.append("sink")
                continue
            mid = a if b <= a else (a + b - 1) // 2
            types.append(next((t for s, e, t in spans if s <= mid < e), "header"))
        types[-1] = "cue"
        for t in types:
            tok_share[t] += 1 / len(types) / n
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=dev), output_attentions=True)
        for l, a in enumerate(out.attentions):                   # (1, heads, T, T)
            row_att = a[0, :, -1, :].mean(0).float().cpu().tolist()     # from the cue, averaged over heads
            for w, t in zip(row_att, types):
                att_mass[l][t] += w / n
    res = {"model": model_dir, "patients": n, "text_matches_converter": f"{check_equal}/{n}",
           "token_share": {t: round(tok_share[t], 4) for t in TYPES},
           "attention_from_cue_by_layer": [{t: round(att_mass[l][t], 4) for t in TYPES} for l in range(n_layers)],
           "read_ratio_by_layer": [{t: (round(att_mass[l][t] / tok_share[t], 2) if tok_share[t] else None) for t in TYPES}
                                   for l in range(n_layers)]}
    print("text matches converter:", res["text_matches_converter"])
    print("token share:", res["token_share"])
    print("layer  " + "  ".join(f"{t:>8s}" for t in TYPES))
    for l in range(n_layers):
        print(f"L{l:<4d} " + "  ".join(f"{att_mass[l][t]:8.3f}" for t in TYPES))
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])

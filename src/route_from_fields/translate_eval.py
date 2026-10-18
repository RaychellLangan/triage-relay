"""Let a translator WRITE the field list on its own (greedy, no teacher forcing), score it, and route its fields.

usage: run_env.sh route_from_fields/translate_eval.py <translator model dir> <messy_split.jsonl> <counted_table.json>
                                                      <out.jsonl> [n]

Per variant: predicted fields, field precision / recall, exact match, the seesaw's department + urgency + fight from the
PREDICTED fields, and each predicted field's confidence (the lowest token probability inside that field's code).
A summary is printed and written beside the per-variant file (<out>.summary.json).
"""
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import common                                    # noqa: E402
from seesaw_router import FieldRouter            # noqa: E402

CODE = re.compile(r"E_\d+(?:_@_(?:V_\d+|\d+))?")


def tidy(s):
    """The word-level tokenizer decodes 'E_54_@_V_161' as 'E_54_ @ _V_161': put codes back together."""
    return re.sub(r"\s*@\s*", "@", s).replace("_ @", "_@")


@torch.no_grad()
def main(model_dir, data, counted, out_path, n=1000):
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(dev).eval()     # GPT-2 or any other causal LM (Gemma arm)
    router = FieldRouter(counted)
    recs = [json.loads(l) for l in open(data, encoding="utf-8")]
    recs = random.Random(3).sample(recs, min(int(n), len(recs)))
    common.TASK = "translator"
    tp = fp = fn = exact = d_ok = u_ok = fights = 0
    out = open(out_path, "w", encoding="utf-8")
    batch = 16
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        enc = tok([common.prompt_of(r) for r in chunk], return_tensors="pt", padding=True).to(dev)
        enc.pop("token_type_ids", None)   # the word-level tokenizer returns these; training never fed them (13 Oct bug)
        gen = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, output_scores=True, return_dict_in_generate=True)
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        probs = torch.stack([torch.softmax(s.float(), -1) for s in gen.scores], 1)       # (B, steps, V)
        chosen_p = probs.gather(-1, new[:, :probs.shape[1]].unsqueeze(-1)).squeeze(-1).cpu()
        for j, r in enumerate(chunk):
            ids = new[j].tolist()
            stop = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else len(ids)
            text = tidy(tok.decode(ids[:stop], skip_special_tokens=True))
            pred = CODE.findall(text)
            # field confidence: the lowest token probability among the tokens that spell it
            conf, pos = {}, 0
            pieces = [tidy(tok.decode([t])) for t in ids[:stop]]
            acc = ""
            low = 1.0
            for k, piece in enumerate(pieces):
                acc += piece
                low = min(low, float(chosen_p[j, k]))
                m = CODE.findall(tidy(acc))
                if m and (k + 1 == len(pieces) or pieces[k + 1].strip().startswith("E")):
                    conf[m[-1]] = round(low, 4)
                    acc, low = "", 1.0
            truth = set(r["codes"].split())
            p = set(pred)
            tp += len(p & truth)
            fp += len(p - truth)
            fn += len(truth - p)
            exact += p == truth
            d = router.route(sorted(p), "department")
            u = router.route(sorted(p), "urgency")
            d_ok += d[0]["label"] == r["dept_true"]
            u_ok += u[0]["label"] == r["urgency_true"]
            fights += d[0]["fight"]
            out.write(json.dumps({"id": r["id"], "v": r["v"], "keep_p": r["keep_p"], "stray_p": r["stray_p"],
                                  "pred": sorted(p), "truth": sorted(truth), "field_conf": conf,
                                  "dept_true": r["dept_true"], "dept_pred": d[0]["label"], "dept_fight": d[0]["fight"],
                                  "dept_top2": [x["label"] for x in d[:2]], "urgency_true": r["urgency_true"],
                                  "urgency_pred": u[0]["label"]}) + "\n")
    n_ = len(recs)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    summary = {"model": model_dir, "n": n_, "field_precision": round(prec, 4), "field_recall": round(rec, 4),
               "field_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "exact_field_set": round(exact / n_, 4),
               "dept_acc_via_seesaw": round(d_ok / n_, 4), "urgency_acc_via_seesaw": round(u_ok / n_, 4),
               "fight_share": round(fights / n_, 4)}
    print(json.dumps(summary, indent=1), flush=True)
    json.dump(summary, open(out_path + ".summary.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])

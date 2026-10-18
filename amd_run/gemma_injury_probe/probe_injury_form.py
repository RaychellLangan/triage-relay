"""Can a wider reader (Gemma) fill the injury form for cases the tiny translator cannot read? No training; it cannot answer
outside the form.

For each sentence and each field, the model is never asked to write anything. Every allowed answer is scored by how likely the
model finds it after the question (mean log-probability per answer token), and the most likely ALLOWED answer is taken, after calibration against the same question with the patient's words left blank. So the
output is always one of the form's own answers. "not said" is always allowed and is the honest default.
The other fields are asked only when the model says it is an injury (as the kiosk would).

usage: python probe_injury_form.py <model id or folder> <out.json> [--chat]
  --chat  wrap each question in the tokenizer's chat template (for instruction-tuned models such as gemma-3-1b-it)
The 30 sentences were written by an AI (Claude), not a clinician: a first look, not a validation.
"""
import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
YN = ["yes", "no", "not said"]
FIELDS = {
    "injury": ("Is this about an injury or accident, something that happened to the body such as a fall, a blow, a cut, a burn or a bite?", ["yes", "no"]),
    "how": ("How did the injury happen?", ["fall", "blow or kick", "cut", "burn or scald", "bite or sting", "crush", "road traffic", "other", "not said"]),
    "where": ("Which part of the body was hurt?", ["head", "face", "neck", "chest", "belly", "back", "arm", "hand or wrist", "hip or leg", "knee", "ankle or foot", "other", "not said"]),
    "knocked_out_or_memory_gap": ("Did they lose consciousness, black out, or lose their memory of what happened?", YN),
    "vomited_more_than_once": ("Have they been sick (vomited) more than once since the injury?", YN),
    "seizure_since": ("Have they had a seizure or fit since the injury?", YN),
    "dangerous_mechanism": ("Was it a dangerous way to be hurt: hit by a car as a pedestrian or cyclist, thrown out of a vehicle, or a fall from more than 1 metre or 5 stairs?", YN),
    "blood_thinners": ("Do they take blood-thinning medicine, such as warfarin, apixaban, rivaroxaban or heparin?", YN),
    "can_walk_four_steps": ("Can they walk at least four steps on the injured leg, even with a limp?", YN),
    "heavy_bleeding": ("Are they bleeding heavily, and it won't stop?", YN),
}


def norm(v):
    return "not said" if v == "unknown" else v


@torch.no_grad()
def choose(model, tok, dev, prompt, options, chat):
    if chat and getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt + "\nAnswer with one of: " + ", ".join(options) + "."}],
                                         tokenize=False, add_generation_prompt=True)
        lead = ""
    else:
        prompt, lead = prompt + "\nAnswer:", " "
    p_ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
    scores = []
    for o in options:
        o_ids = tok(lead + o, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        ids = torch.cat([p_ids, o_ids], 1)
        logp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        lp = logp[torch.arange(len(tgt)), tgt][-o_ids.shape[1]:]
        scores.append(lp.mean().item())
    return scores


@torch.no_grad()
def pick(model, tok, dev, text, question, options, chat):
    """contextual calibration (Zhao et al. 2021, 'Calibrate Before Use'): each answer is scored against how likely the model
    finds it when the patient's words are blank, so a model's habit of saying 'no' is not mistaken for reading the case"""
    real = choose(model, tok, dev, f'A patient at the hospital front desk wrote: "{text}"\nQuestion: ' + question, options, chat)
    blank = choose(model, tok, dev, 'A patient at the hospital front desk wrote: ""\nQuestion: ' + question, options, chat)
    cal = [a - b for a, b in zip(real, blank)]
    return options[max(range(len(options)), key=lambda i: cal[i])]


def main(model_id, out_path, chat=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dev).eval()
    rows = [json.loads(l) for l in open(os.path.join(HERE, "probe_sentences.jsonl"), encoding="utf-8")]
    per_field, results = {}, []
    for r in rows:
        got = {}
        inj = pick(model, tok, dev, r["text"], FIELDS["injury"][0], FIELDS["injury"][1], chat)
        got["injury"] = inj
        if inj == "yes":
            for f, (q, opts) in FIELDS.items():
                if f != "injury":
                    got[f] = pick(model, tok, dev, r["text"], q, opts, chat)
        marks = {}
        for f, want in r["expect"].items():
            want = [norm(w) for w in (want if isinstance(want, list) else [want])]
            if f == "injury" and len(want) > 1:
                continue                                          # arguable (a sting): not scored
            ok = got.get(f, "not asked") in want
            marks[f] = ok
            per_field.setdefault(f, [0, 0])
            per_field[f][0] += ok
            per_field[f][1] += 1
        results.append({"id": r["id"], "text": r["text"], "got": got, "right": marks})
        print(f"#{r['id']:2d} {'ok ' if all(marks.values()) else '-- '} {r['text'][:60]:60s} | " +
              " ".join(f"{f}={got.get(f, '-')}{'' if marks.get(f, True) else '(x)'}" for f in r["expect"]), flush=True)
    summary = {f: {"right": a, "of": b, "share": round(a / b, 3)} for f, (a, b) in per_field.items()}
    print("\nper field:", json.dumps(summary))
    json.dump({"model": model_id, "chat": chat, "device": dev, "summary": summary, "rows": results},
              open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], chat="--chat" in sys.argv)

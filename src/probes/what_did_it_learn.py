"""What did the router actually learn? Stress it and watch accuracy AND confidence (the Stuffy Room question:
a right answer for a reason that will not hold).

usage: run_env.sh probes/what_did_it_learn.py <model dir> <out.json> [n_patients]

Stresses, each on the same test patients:
  full            the intake as converted
  shuffled        the symptom and history lines in a random order (a model keyed on POSITION falls here)
  reason_only     age, sex and the reason for the visit only
  keep_p          a random share p of the answers kept (incomplete intake), p in 0.8 / 0.6 / 0.4 / 0.2
  leave_one_out   each answer dropped in turn: which ones flip the department, and are they the condition's own
                  listed symptoms (DDXPlus release_conditions.json) or history / travel items?
For each: department accuracy, mean confidence (mouth top probability), mean mouth entropy, and the share of WRONG
answers given with confidence above 0.9 (confidently wrong).
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import csv
import json
import os
import random
import sys

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import common                                   # noqa: E402
import convert_ddxplus as conv                  # noqa: E402

RAW = ROOT + "/data_raw"
TABLE = os.path.join(HERE, "..", "labels", "routing_table.json")


def load_rows(n, seed=7):
    """Reservoir sample: streams the file, holds only n rows (RAM is shared with training runs)."""
    rng, keep = random.Random(seed), []
    with open(f"{RAW}/ddxplus_en_unzipped/release_test_patients", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < n:
                keep.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    keep[j] = row
    return keep


def with_evidences(row, evidence_items):
    r = dict(row)
    r["EVIDENCES"] = repr(list(evidence_items))
    return r


def groups_of(row):
    """evidence items grouped by question, keeping the row's order: [(name, [items...]), ...]"""
    import ast
    out = {}
    for item in ast.literal_eval(row["EVIDENCES"]):
        out.setdefault(item.partition("_@_")[0], []).append(item)
    return list(out.items())


class Router:
    def __init__(self, model_dir, device):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(device).eval()
        self.dev = device
        self.dept_ids = common.first_token_ids(self.tok, common.DEPARTMENTS)
        self.eos = self.tok.eos_token_id

    @torch.no_grad()
    def mouth(self, texts, batch=64):
        """-> list of 14-way department distributions read at the cue, before anything is written."""
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ids = [self.tok(t + common.ROUTE_CUE)["input_ids"] for t in chunk]
            width = max(len(x) for x in ids)
            x = torch.full((len(ids), width), self.eos, dtype=torch.long)
            att = torch.zeros_like(x)
            for j, s in enumerate(ids):
                x[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            last = torch.tensor([len(s) - 1 for s in ids], device=self.dev)
            logits = common.logits_at(self.model, x.to(self.dev), att.to(self.dev), last)
            p = torch.softmax(logits[:, self.dept_ids].float(), -1)
            out.extend(p.cpu().tolist())
        return out


def score(dists, truths):
    n = len(truths)
    right = conf = ent = cw = 0.0
    import math
    for p, t in zip(dists, truths):
        k = max(range(len(p)), key=p.__getitem__)
        right += (k == t)
        conf += p[k]
        ent += -sum(q * math.log2(q) for q in p if q > 0)
        cw += (k != t and p[k] > 0.9)
    return {"dept_acc": round(right / n, 4), "mean_confidence": round(conf / n, 4),
            "mean_entropy_bits": round(ent / n, 4), "confidently_wrong_share": round(cw / n, 4)}


def main(model_dir, out_path, n=1000):
    n = int(n)
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    router = Router(model_dir, dev)
    evid = conv.load_evidences(f"{RAW}/ddxplus_en/release_evidences.json")
    conds = json.load(open(f"{RAW}/ddxplus_en/release_conditions.json", encoding="utf-8"))
    table = json.load(open(TABLE, encoding="utf-8"))
    rows = load_rows(n)
    truths = [common.DEPARTMENTS.index(table[r["PATHOLOGY"]]["department"]) for r in rows]
    rng = random.Random(11)
    result = {"model": model_dir, "patients": n}

    def run(name, make_row):
        texts = [conv.patient_text(make_row(r), evid) for r in rows]
        result[name] = score(router.mouth(texts), truths)
        print(name, result[name], flush=True)

    run("full", lambda r: r)

    def shuffled(r):
        g = groups_of(r)
        init = [x for x in g if x[0] == r["INITIAL_EVIDENCE"]]
        rest = [x for x in g if x[0] != r["INITIAL_EVIDENCE"]]
        rng.shuffle(rest)
        return with_evidences(r, [i for _, items in init + rest for i in items])
    run("shuffled", shuffled)

    run("reason_only", lambda r: with_evidences(r, [i for name, items in groups_of(r)
                                                    if name == r["INITIAL_EVIDENCE"] for i in items]))
    for p in (0.8, 0.6, 0.4, 0.2):
        def keep(r, p=p):
            g = groups_of(r)
            kept = [x for x in g if x[0] == r["INITIAL_EVIDENCE"] or rng.random() < p]
            return with_evidences(r, [i for _, items in kept for i in items])
        run(f"keep_{p}", keep)

    # STRAY SYMPTOMS (real patients carry unrelated ones; DDXPlus's generator never adds one).
    # Add k yes/no symptoms that belong to a condition in a DIFFERENT department and NOT to this patient's condition.
    by_dept = {}
    for name, c in conds.items():
        by_dept.setdefault(table[name]["department"], []).append(name)

    def strays(r, k):
        own = conds[r["PATHOLOGY"]]
        own_set = set(own.get("symptoms", {})) | set(own.get("antecedents", {}))
        dept = table[r["PATHOLOGY"]]["department"]
        other = rng.choice([n for n in conds if table[n]["department"] != dept])
        pool = [e for e in conds[other].get("symptoms", {}) if evid[e]["data_type"] == "B" and e not in own_set]
        rng.shuffle(pool)
        return [i for _, items in groups_of(r) for i in items] + pool[:k]
    for k in (1, 2, 3, 5):
        run(f"stray_{k}", lambda r, k=k: with_evidences(r, strays(r, k)))

    # leave one out: which single answers flip the department?
    base = router.mouth([conv.patient_text(r, evid) for r in rows])
    flips = {"own_symptom": 0, "own_antecedent": 0, "not_listed_for_condition": 0}
    tried = {"own_symptom": 0, "own_antecedent": 0, "not_listed_for_condition": 0}
    flip_by_question = {}
    for r, b in zip(rows, base):
        g = groups_of(r)
        k0 = max(range(len(b)), key=b.__getitem__)
        cond = conds[r["PATHOLOGY"]]
        variants, kinds, names = [], [], []
        for name, _ in g:
            if name == r["INITIAL_EVIDENCE"]:
                continue
            kept = [i for nm, items in g if nm != name for i in items]
            variants.append(conv.patient_text(with_evidences(r, kept), evid))
            kind = ("own_symptom" if name in cond.get("symptoms", {}) else
                    "own_antecedent" if name in cond.get("antecedents", {}) else "not_listed_for_condition")
            kinds.append(kind)
            names.append(name)
        for p, kind, name in zip(router.mouth(variants), kinds, names):
            tried[kind] += 1
            if max(range(len(p)), key=p.__getitem__) != k0:
                flips[kind] += 1
                flip_by_question[name] = flip_by_question.get(name, 0) + 1
    result["leave_one_out"] = {
        "answers_tried": tried, "answers_that_flip": flips,
        "flip_rate": {k: round(flips[k] / tried[k], 4) if tried[k] else None for k in tried},
        "top_flipping_questions": [(q, c, evid[q]["question_en"]) for q, c in
                                   sorted(flip_by_question.items(), key=lambda kv: -kv[1])[:15]],
    }
    print("leave_one_out", json.dumps(result["leave_one_out"], ensure_ascii=False, indent=1), flush=True)
    json.dump(result, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])

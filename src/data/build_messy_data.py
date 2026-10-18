"""Messy intake language -> (a) the patient's fields as codes, (b) department + urgency. One JSON line per variant.

usage: pyd.sh -I build_messy_data.py <patients.csv> <release_evidences.json> <release_conditions.json> <routing_table.json>
                                    <out.jsonl> <n_patients> <variants_per_patient> <seed>

The input is what a person might type or say at a desk, generated from the patient's own answers with noise:
  - question wording stripped to a short phrase; answer words kept ("pain lower chest, burning, 6")
  - random order, random separators, lower case, the odd typo
  - some answers left out (the patient did not mention them)           -> NOT in the target either
  - sometimes 1-3 stray symptoms from an unrelated condition (comorbidity) -> IN the target (they said it)
The TARGET for the translator is the set of fields the text actually contains, as codes in a fixed order
("E_53 E_54_@_V_161 E_201 ..."); the router's target is department + urgency of the recorded condition.
SYNTHETIC: generated from DDXPlus synthetic patients. Not real speech; a stand-in for it.
"""
import ast
import csv
import json
import random
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import convert_ddxplus as conv      # noqa: E402
from natural import natural_bits     # noqa: E402
NATURAL_P = float(__import__("os").environ.get("MESSY_NATURAL", "0"))   # share of variants in everyday language

LEADS = ("do you have ", "have you ", "are you ", "do you ", "is your ", "did you ", "does the ", "does your ",
         "how intense is the ", "characterize your ", "how fast did the ", "how precisely is the ", "where is the ",
         "what color is the ", "do your ", "is the ", "have you had ", "do you feel ")
# STRESS SETS ONLY: env MESSY_KEEP fixes the share of answers kept, MESSY_STRAYS fixes how many stray symptoms every variant gets
FIXED_KEEP = __import__("os").environ.get("MESSY_KEEP")
FIXED_STRAYS = __import__("os").environ.get("MESSY_STRAYS")
SEPS = (". ", ", ", "; ", " and ", ". ", ", ")


def short_phrase(q):
    q = q.strip().rstrip("?").rstrip(":").lower()
    for lead in LEADS:
        if q.startswith(lead):
            return q[len(lead):]
    return q


def typo(word, rng):
    if len(word) < 4 or rng.random() > 0.03:
        return word
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def code_key(item):
    name, _, value = item.partition("_@_")
    num = int(name.split("_")[1])
    vnum = int(value.split("_")[1]) if value.startswith("V_") else (int(value) if value.isdigit() else 0)
    return (num, vnum, value)


def variant(row, evid, conds, table, rng, keep_p, stray_p):
    groups = {}
    for item in ast.literal_eval(row["EVIDENCES"]):
        groups.setdefault(item.partition("_@_")[0], []).append(item)
    init = row["INITIAL_EVIDENCE"]
    kept = {n: its for n, its in groups.items() if n == init or rng.random() < keep_p}
    if rng.random() < stray_p:                                    # comorbidity: stray yes/no symptoms they also report
        dept = table[row["PATHOLOGY"]]["department"]
        other = rng.choice([c for c in conds if table[c]["department"] != dept])
        own = set(groups)
        pool = [e for e in conds[other]["symptoms"] if evid[e]["data_type"] == "B" and e not in own]
        for e in rng.sample(pool, min(len(pool), int(FIXED_STRAYS) if FIXED_STRAYS else rng.randint(1, 3))):
            kept[e] = [e]
    bits = []
    if rng.random() < NATURAL_P:           # how people actually say it (natural.py)
        bits = [" ".join(typo(w, rng) for w in b.split()) for b in natural_bits(kept, evid, rng)]
    else:                                   # the dataset's own wording, stripped down
        for name, its in kept.items():
            ev = evid[name]
            vals = [conv.answer_text(ev, it.partition("_@_")[2]) for it in its if "_@_" in it]
            words = (short_phrase(ev["question_en"]) + (" " + ", ".join(vals) if vals else "")).split()
            bits.append(" ".join(typo(w, rng) for w in words))
    rng.shuffle(bits)
    text = f"{row['AGE']}{row['SEX'].lower()} " + "".join(b + rng.choice(SEPS) for b in bits).rstrip(" ,;.") + "."
    codes = sorted((it for its in kept.values() for it in its), key=code_key)
    return text, codes


def main(patients, evid_path, cond_path, table_path, out_path, n, k, seed):
    n, k, rng = int(n), int(k), random.Random(int(seed))
    evid = conv.load_evidences(evid_path)
    conds = json.load(open(cond_path, encoding="utf-8"))
    table = json.load(open(table_path, encoding="utf-8"))
    written = 0
    with open(patients, encoding="utf-8", newline="") as f, open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            lab = conv.labels(row, table)
            for v in range(k):
                keep_p = float(FIXED_KEEP) if FIXED_KEEP else rng.choice((1.0, 1.0, 0.85, 0.7, 0.5))
                stray_p = 1.0 if FIXED_STRAYS else rng.choice((0.0, 0.0, 0.3, 0.6))
                text, codes = variant(row, evid, conds, table, rng, keep_p, stray_p)
                rec = {"id": i, "v": v, "text": text, "codes": " ".join(codes), "keep_p": keep_p, "stray_p": stray_p}
                rec.update(lab)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} variants of {min(i + 1, n)} patients -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:9])

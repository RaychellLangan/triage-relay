"""WHAT association did the router learn? (the CO2-vs-people-in-the-room hunt)

usage: run_env.sh probes/what_is_it_keyed_on.py <model dir> <out.json> [n_patients]

Suspect: in DDXPlus a question is only ASKED when the patient's condition involves it, so the PRESENCE of a question
carries the condition even when the answer is "no" (travel abroad: "N"). A router can be right by reading which questions
were asked, never what the patient said.

A  content_blind   keep every question, replace every multi-choice / scale / categorical ANSWER with another valid
                   value of the same question. Accuracy that barely moves = it never read the answers.
B  planted_no      add a question at its default "no / none" value to patients who were never asked it; record how often
                   the department changes and toward where. A medically empty answer that steers = presence is the signal.
C  single         each question ALONE (age 40, sex F, the question as the reason for the visit, answered yes or with its
                   first non-default value). The mouth's pick + confidence per question, against how often that question
                   appears per department in TRAINING data (P(department | question asked)). Agreement = it learned the
                   dataset's question statistics.
D  demographics    the full intake with age set to 2 / 40 / 80 and sex swapped; shift rate per true department.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import ast
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
import common                       # noqa: E402
import convert_ddxplus as conv      # noqa: E402
from stresses import groups_of, with_items   # noqa: E402
from what_did_it_learn import Router          # noqa: E402

RAW = ROOT + "/data_raw"
TABLE = os.path.join(HERE, "..", "labels", "routing_table.json")


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


def picks(router, texts):
    d = router.mouth(texts)
    return [max(range(14), key=p.__getitem__) for p in d], [max(p) for p in d]


def main(model_dir, out_path, n=600):
    n = int(n)
    dev = os.environ.get("PROBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    router = Router(model_dir, dev)
    evid = conv.load_evidences(f"{RAW}/ddxplus_en/release_evidences.json")
    table = json.load(open(TABLE, encoding="utf-8"))
    rng = random.Random(5)
    rows = reservoir(f"{RAW}/ddxplus_en_unzipped/release_test_patients", n, seed=7)
    truth = [common.DEPARTMENTS.index(table[r["PATHOLOGY"]]["department"]) for r in rows]
    text = lambda r: conv.patient_text(r, evid)          # noqa: E731
    base_pick, base_conf = picks(router, [text(r) for r in rows])
    acc = lambda p: round(sum(a == b for a, b in zip(p, truth)) / len(truth), 4)   # noqa: E731
    out = {"model": model_dir, "patients": n, "full_acc": acc(base_pick)}

    # A — content blind
    def blind(r):
        items = []
        for name, its in groups_of(r):
            ev = evid[name]
            vals = [v for v in ev.get("possible-values", []) if v != ev.get("default_value")]
            if ev["data_type"] == "B" or not vals:
                items += its
                continue
            k = max(1, sum("_@_" in i for i in its))
            items += [f"{name}_@_{v}" for v in rng.sample(vals, min(k, len(vals)))]
        return with_items(r, items)
    bp, bc = picks(router, [text(blind(r)) for r in rows])
    out["A_content_blind"] = {"acc": acc(bp), "changed_vs_full": round(sum(a != b for a, b in zip(bp, base_pick)) / n, 4),
                              "mean_conf": round(sum(bc) / n, 4), "full_mean_conf": round(sum(base_conf) / n, 4)}
    print("A content_blind", out["A_content_blind"], flush=True)

    # B — planted "no": every categorical question whose default value means none/no
    defaults = [q for q, ev in evid.items() if ev["data_type"] in ("C", "M") and isinstance(ev.get("default_value"), str)
                and ev["value_meaning"].get(ev["default_value"], {}).get("en", "").strip().upper() in ("N", "NA", "NO", "NOWHERE")]
    out["B_planted_no"] = {}
    for q in defaults:
        targets = [i for i, r in enumerate(rows) if q not in dict(groups_of(r))]
        if not targets:
            continue
        planted = [with_items(rows[i], [x for _, its in groups_of(rows[i]) for x in its] + [f"{q}_@_{evid[q]['default_value']}"])
                   for i in targets]
        pp, _ = picks(router, [text(r) for r in planted])
        changed = [(base_pick[i], p) for i, p in zip(targets, pp) if p != base_pick[i]]
        toward = Counter(common.DEPARTMENTS[p] for _, p in changed)
        out["B_planted_no"][q] = {"question": evid[q]["question_en"],
                                  "answer_planted": evid[q]["value_meaning"][evid[q]["default_value"]]["en"],
                                  "patients": len(targets), "department_changed": len(changed),
                                  "changed_share": round(len(changed) / len(targets), 4), "toward": dict(toward.most_common(4))}
        print("B planted_no", q, out["B_planted_no"][q], flush=True)

    # C — single questions vs the training data's own question statistics
    train = reservoir(f"{RAW}/ddxplus_en_unzipped/release_train_patients", 30000, seed=9)
    asked = defaultdict(Counter)
    for r in train:
        d = table[r["PATHOLOGY"]]["department"]
        for q in {i.partition("_@_")[0] for i in ast.literal_eval(r["EVIDENCES"])}:
            asked[q][d] += 1
    singles, names = [], []
    for q, ev in evid.items():
        vals = [v for v in ev.get("possible-values", []) if v != ev.get("default_value")]
        item = q if (ev["data_type"] == "B" or not vals) else f"{q}_@_{vals[0]}"
        r = {"AGE": "40", "SEX": "F", "EVIDENCES": repr([item]), "INITIAL_EVIDENCE": q}
        singles.append(conv.patient_text(r, evid))
        names.append(q)
    d = router.mouth(singles)
    rows_c, agree, have = [], 0, 0
    for q, p in zip(names, d):
        k = max(range(14), key=p.__getitem__)
        stat = asked.get(q)
        data_top = stat.most_common(1)[0][0] if stat else None
        data_share = round(stat.most_common(1)[0][1] / sum(stat.values()), 3) if stat else None
        if data_top:
            have += 1
            agree += (common.DEPARTMENTS[k] == data_top)
        rows_c.append({"q": q, "question": evid[q]["question_en"], "model_pick": common.DEPARTMENTS[k],
                       "model_conf": round(max(p), 4), "data_top_department_when_asked": data_top,
                       "data_share": data_share, "times_asked_in_30k": sum(stat.values()) if stat else 0})
    rows_c.sort(key=lambda x: -x["model_conf"])
    out["C_single_question"] = {"questions": len(rows_c), "with_training_stats": have,
                                "model_pick_equals_data_top_department": round(agree / max(have, 1), 4),
                                "most_confident_25": rows_c[:25], "all": rows_c}
    print("C single: model pick == the department that question is most often asked for, in",
          out["C_single_question"]["model_pick_equals_data_top_department"], "of", have, flush=True)
    for x in rows_c[:12]:
        print("   ", round(x["model_conf"], 3), x["model_pick"], "| data:", x["data_top_department_when_asked"],
              x["data_share"], "|", x["question"][:80], flush=True)

    # D — demographics
    out["D_demographics"] = {}
    for label, change in (("age_2", lambda r: {**r, "AGE": "2"}), ("age_40", lambda r: {**r, "AGE": "40"}),
                          ("age_80", lambda r: {**r, "AGE": "80"}),
                          ("sex_swapped", lambda r: {**r, "SEX": "M" if r["SEX"] == "F" else "F"})):
        dp, _ = picks(router, [text(change(r)) for r in rows])
        by = defaultdict(lambda: [0, 0])
        for t, a, b in zip(truth, base_pick, dp):
            by[common.DEPARTMENTS[t]][0] += 1
            by[common.DEPARTMENTS[t]][1] += (a != b)
        out["D_demographics"][label] = {"acc": acc(dp), "changed_share": round(sum(a != b for a, b in zip(dp, base_pick)) / n, 4),
                                        "changed_by_true_department": {k: f"{v[1]}/{v[0]}" for k, v in sorted(by.items()) if v[1]}}
        print("D", label, out["D_demographics"][label], flush=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])

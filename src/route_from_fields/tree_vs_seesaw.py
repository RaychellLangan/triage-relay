"""A decision tree against the counted seesaws, on the same patients: where does each sound sure, and is it right?

    run_env.sh -B route_from_fields/tree_vs_seesaw.py [n_train] [n_test] [test set]      (defaults 200000, 10000, natural2_test)

THE TREE: an ordinary scikit-learn decision tree (the kind of branching rule set a kiosk questionnaire is), trained to predict
the department from the same answer codes the seesaws read. It is trained on natural2_train, the noisy everyday-language
set (answers dropped and stray answers added), which is GENEROUS to the tree: the seesaws were counted from clean patients.
Two trees: one grown out fully (every leaf one answer, the usual rule tree), and one stopped at leaves of at least 50
patients, so it has a "how sure" (the share of its leaf that went to its answer).

THE SEESAWS: the app's own counted department planks (FieldRouter, counted_table.json), unchanged.

ON natural2_test (held out), reported:
  1. accuracy of each;
  2. "sounds sure while the evidence fights": the tree's leaf is at least 90% one department while the seesaws' winning
     plank FIGHTS (the losing side outweighs its margin), and how often the tree is right then;
  3. one answer taken away: for each patient, drop one answer at random and count how often the department changes;
  4. order: shuffle the answers and check the seesaws give the identical result (brute force; a tree has no order to
     shuffle, but its path is decided by whichever answer it asks about first);
  5. what the full tree asks first: the top splits, as question text, for a reader to judge.
Writes EVIDENCE-ready JSON to runs/tree_vs_seesaw.json.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import os
import random
import sys
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from seesaw_router import FieldRouter     # noqa: E402
import convert_ddxplus as conv            # noqa: E402

R = ROOT + ""
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
TEST_SET = sys.argv[3] if len(sys.argv) > 3 else "natural2_test"   # natural2_validate = a second, separate set of held-out patients
SURE = 0.9


def read(path, n, seed):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = random.Random(seed).sample(recs, min(n, len(recs)))
    return [(r["codes"].split(), r["dept_true"]) for r in recs]


train = read(f"{R}/data_built/natural2_train.jsonl", N_TRAIN, 1)
test = read(f"{R}/data_built/{TEST_SET}.jsonl", N_TEST, 2)
vocab = sorted({c for f, _ in train + test for c in f})
col = {c: i for i, c in enumerate(vocab)}


def matrix(rows):
    r, c = [], []
    for i, (f, _) in enumerate(rows):
        for x in set(f):
            r.append(i)
            c.append(col[x])
    return csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(rows), len(vocab)))


Xtr, ytr = matrix(train), [d for _, d in train]
Xte, yte = matrix(test), [d for _, d in test]
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
leafy = DecisionTreeClassifier(random_state=0, min_samples_leaf=50).fit(Xtr, ytr)
router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")


def seesaw(fields):
    p = router.route(fields, "department")
    return p[0]["label"], bool(p[0]["fight"])


out = {"n_train": len(train), "n_test": len(test), "sure_line": SURE,
       "tree_full": {"depth": int(full.get_depth()), "leaves": int(full.get_n_leaves())},
       "tree_leafy": {"depth": int(leafy.get_depth()), "leaves": int(leafy.get_n_leaves()), "min_samples_leaf": 50}}

full_pred = full.predict(Xte)
leafy_proba = leafy.predict_proba(Xte)
leafy_pred = leafy.classes_[leafy_proba.argmax(1)]
leafy_conf = leafy_proba.max(1)
ss = [seesaw(f) for f, _ in test]

acc = lambda pred: round(float(np.mean([p == y for p, y in zip(pred, yte)])), 4)
out["accuracy"] = {"tree_full": acc(full_pred), "tree_leafy": acc(leafy_pred), "seesaw": acc([s[0] for s in ss])}
out["full_tree_ever_unsure"] = "never: every fully grown leaf is one answer"

# 2. sounds sure while the evidence fights
fight = np.array([s[1] for s in ss])
sure = leafy_conf >= SURE
right_leafy = np.array([p == y for p, y in zip(leafy_pred, yte)])
right_ss = np.array([s[0] == y for s, y in zip(ss, yte)])
both = sure & fight
out["sounds_sure_while_fighting"] = {
    "seesaw_fights": int(fight.sum()),
    "tree_sure": int(sure.sum()),
    "tree_sure_and_seesaw_fights": int(both.sum()),
    "one_in": round(len(test) / max(1, int(both.sum())), 1),
    "tree_right_when_sure_and_calm": round(float(right_leafy[sure & ~fight].mean()), 4),
    "tree_right_when_sure_but_fighting": round(float(right_leafy[both].mean()), 4) if both.any() else None,
    "seesaw_right_when_calm": round(float(right_ss[~fight].mean()), 4),
    "seesaw_right_when_fighting": round(float(right_ss[fight].mean()), 4),
}
# where the tree's CONFIDENT mistakes are: what share of them sit in the patients the seesaws flag as fighting
sure_wrong = sure & ~right_leafy
full_wrong = np.array([p != y for p, y in zip(full_pred, yte)])
out["where_the_tree_is_wrong"] = {
    "share_of_patients_flagged_fighting": round(float(fight.mean()), 4),
    "tree_sure_mistakes": int(sure_wrong.sum()),
    "of_those_flagged_fighting": int((sure_wrong & fight).sum()),
    "share_flagged": round(float((sure_wrong & fight).sum() / max(1, sure_wrong.sum())), 4),
    "full_tree_mistakes": int(full_wrong.sum()),
    "full_tree_mistakes_flagged_fighting": int((full_wrong & fight).sum()),
    "full_tree_share_flagged": round(float((full_wrong & fight).sum() / max(1, full_wrong.sum())), 4),
}

# 3. one answer taken away
rng = random.Random(7)
flip_tree = flip_ss = n_drop = 0
for (f, _), sp, tp in zip(test, ss, full_pred):
    if len(f) < 2:
        continue
    g = list(f)
    g.pop(rng.randrange(len(g)))
    n_drop += 1
    flip_tree += full.predict(matrix([(g, "")]))[0] != tp
    flip_ss += seesaw(g)[0] != sp[0]
out["one_answer_removed"] = {"patients": n_drop, "tree_full_changed": flip_tree, "seesaw_changed": flip_ss,
                             "tree_full_share": round(flip_tree / n_drop, 4), "seesaw_share": round(flip_ss / n_drop, 4)}

# 4. order never matters for the seesaws (brute force: 5 shuffles of each of 2,000 patients)
same = total = 0
for f, _ in test[:2000]:
    base = router.route(f, "department")
    for _ in range(5):
        g = list(f)
        rng.shuffle(g)
        total += 1
        same += router.route(g, "department") == base
out["seesaw_order_shuffles"] = {"tried": total, "identical": same}

# 5. what the full tree asks first (as question text)
ev = conv.load_evidences(f"{R}/data_raw/ddxplus_en/release_evidences.json")


def say(code):
    q, _, v = code.partition("_@_")
    e = ev.get(q, {})
    text = e.get("question_en", q)
    if v:
        val = e.get("value_meaning", {}).get(v, {})
        text += f" -> {val.get('en', v) if isinstance(val, dict) else v}"
    return text


t = full.tree_
first = []
stack = [(0, 0)]
while stack:
    node, depth = stack.pop(0)
    if depth > 2 or t.feature[node] < 0:
        continue
    first.append({"depth": depth, "asks": vocab[t.feature[node]], "question": say(vocab[t.feature[node]]),
                  "patients": int(t.n_node_samples[node])})
    stack += [(t.children_left[node], depth + 1), (t.children_right[node], depth + 1)]
out["full_tree_first_questions"] = first

out["test_set"] = TEST_SET
json.dump(out, open(f"{R}/runs/tree_vs_seesaw" + ("" if TEST_SET == "natural2_test" else "_" + TEST_SET) + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "full_tree_first_questions"}, indent=1, ensure_ascii=False))
for q in first:
    print("  " * q["depth"] + f"[{q['patients']}] {q['question']}")

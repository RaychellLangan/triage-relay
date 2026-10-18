"""Why does danger-first still miss severe cases? For each missed one: the true condition's rank among the condition planks,
its force gap to the winner against the winner's contest, and what the patient did / did not report of that condition's
listed symptoms. usage: pyd.sh -I why_missed.py <messy file> [n]"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json, random, sys, collections
sys.path.insert(0, ROOT + "/src/route_from_fields")
from seesaw_router import FieldRouter
R = ROOT + ""
router = FieldRouter(f"{R}/runs/route_from_fields/counted_table.json")
table = json.load(open(f"{R}/src/labels/routing_table.json", encoding="utf-8"))
conds = json.load(open(f"{R}/data_raw/ddxplus_en/release_conditions.json", encoding="utf-8"))
sev = {c: table[c]["severity"] for c in table}
recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
recs = random.Random(3).sample(recs, min(int(sys.argv[2]) if len(sys.argv) > 2 else 1000, len(recs)))
missed, why = [], collections.Counter()
for r in recs:
    fields = r["codes"].split()
    planks = router.route(fields, "condition")
    win = planks[0]
    in_play = [p for p in planks if p["gap_to_winner"] < win["contest"]] or [win]
    pick = min(in_play, key=lambda p: (sev[p["label"]], -p["force"]))
    t = r["pathology"]
    if sev[t] <= 2 and sev[pick["label"]] > sev[t]:
        rank = [p["label"] for p in planks].index(t) + 1
        tp = next(p for p in planks if p["label"] == t)
        listed = set(conds[t]["symptoms"]) | set(conds[t]["antecedents"])
        said = {f.partition("_@_")[0] for f in fields}
        missed.append((t, rank, round(tp["gap_to_winner"], 1), round(win["contest"], 1), win["label"],
                       len(listed & said), len(listed), r["keep_p"]))
        why["true condition ranked 2nd but gap > contest" if rank == 2 else f"true condition ranked {min(rank,5)}{'+' if rank>=5 else ''}"] += 1
print("missed severe:", len(missed), "of", sum(sev[r["pathology"]] <= 2 for r in recs), dict(why))
print("by true condition:", collections.Counter(m[0] for m in missed).most_common(6))
print("won instead:", collections.Counter(m[4] for m in missed).most_common(6))
for m in missed[:10]:
    print(f"  true {m[0]} (rank {m[1]}, gap {m[2]} vs winner contest {m[3]}) won: {m[4]}; reported {m[5]} of its {m[6]} listed items (keep {m[7]})")

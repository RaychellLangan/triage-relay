"""Route from FIELDS with seesaws: one plank per department ("send here" <-> "not here"), one per urgency.

Each field the patient mentioned is a lever whose push is tanh(lean) from the counted table (counted_table.py); a field
nobody mentioned weighs nothing ("don't know", never "no"); a base-rate lever carries how common the department is.
The seesaw engine is the 28 Sep playground's (ruled publishable); it is imported from the pile until confirmed
may be copied into the repo.

Per plank the engine reports tilt (-1..+1), load (how much weight is on it), torn (how much of it cancels) and contest
(the losing side's weight). FIGHT = the losing side outweighs the margin it lost by: two measured weights, no threshold.
Summed log-ratio forces are naive Bayes in log-odds; what the seesaw adds is the reporting: empty vs loaded, torn, fight.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repo folder

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT + "/seesaw")
import seesaw as engine          # noqa: E402


class FieldRouter:
    def __init__(self, counted_path):
        c = json.load(open(counted_path, encoding="utf-8"))
        self.kinds = {}
        for kind, k in c["kinds"].items():
            beams = [{"name": lab, "high": f"send to {lab}", "low": f"not {lab}"} for lab in k["labels"]]
            levers = [{"id": f"base:{lab}", "beam": lab, "driver": "_base", "gates": [],
                       "if_yes": math.tanh(k["base_lean"][lab]), "if_no": 0.0} for lab in k["labels"]]
            no_lean = k.get("no_lean", {})
            for item, per in k["lean"].items():
                for lab, lean in per.items():
                    levers.append({"id": f"{item}:{lab}", "beam": lab, "driver": item, "gates": [],
                                   "if_yes": math.tanh(lean), "if_no": math.tanh(no_lean.get(item, {}).get(lab, 0.0))})
            self.kinds[kind] = (k["labels"], engine.Seesaws({"beams": beams, "levers": levers}))

    def route(self, fields, kind="department", per_question=False, negatives=(), extra=()):
        """fields: the codes the patient's intake contains -> ranked planks with the engine's readings.
        per_question: several answers to ONE question (three pain locations) share that question's push (each fact reads
        1/k) instead of each pushing in full: one vote per question, a first correction for counting redundant fields."""
        labels, ss = self.kinds[kind]
        if per_question:
            k = {}
            for f in fields:
                q = f.partition("_@_")[0]
                k[q] = k.get(q, 0) + 1
            facts = {f: 1.0 / k[f.partition("_@_")[0]] for f in fields}
        else:
            facts = {f: 1.0 for f in fields}
        for f in negatives:                      # a clear NO from a follow-up question (an unmentioned field stays 0)
            facts[f] = -1.0
        facts["_base"] = 1.0
        # extra: reviewed rules, each ONE labelled lever that exists only for this call (the counted table is never changed)
        for lv in extra:
            facts[lv["driver"]] = 1.0
            ss.levers_on[lv["beam"]].append(lv)
        try:
            res = ss.run(facts)
        finally:
            for lv in extra:
                ss.levers_on[lv["beam"]].remove(lv)
        force = {lab: sum(x["force"] for x in res[lab]["parts"]) for lab in labels}
        ranked = sorted(labels, key=lambda lab: -force[lab])
        top = force[ranked[0]]
        planks = [{"label": lab, "tilt": round(res[lab]["tilt"], 4), "force": round(force[lab], 3),
                   "gap_to_winner": round(top - force[lab], 3), "load": round(res[lab]["load"], 3),
                   "torn": round(res[lab]["torn"], 3), "contest": round(res[lab]["contest"], 3),
                   "fight": engine.fights(res[lab])} for lab in ranked]
        return planks

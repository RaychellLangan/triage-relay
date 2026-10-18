"""Pick six real example cases from the explorer's data for the explainer's walk-through. -> explainer/cases.json"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # the repo folder

import json
R = ROOT + ""
C = json.loads(open(f"{R}/explorer/data.js", encoding="utf-8").read().split("=", 1)[1].rstrip().rstrip(";"))
def first(pred):
    return next(c for c in C if pred(c))
picks = [
    ("A clear case", first(lambda c: c["set"] == "in distribution" and c["verdict"]["word"] == "yes" and c["seesaw_right"] and 8 <= len(c["fields"]) <= 14)),
    ("Five stray symptoms", first(lambda c: c["set"] == "5 stray symptoms" and c["verdict"]["word"] == "yes" and c["seesaw_right"])),
    ("Caught and handed over", first(lambda c: c["seesaw_right"] is False and c["verdict"]["word"] in ("hand to a person", "no") and c["ask_next"])),
    ("Almost nothing said", first(lambda c: c["verdict"]["word"] == "no idea")),
    ("Not an intake at all", first(lambda c: c["kind"] == "english")),
    ("An instruction planted in the text", first(lambda c: c["kind"] == "injected" and c["seesaw_right"])),
]
out = []
for title, c in picks:
    out.append({"title": title, "set": c["set"], "text": c["text"], "fields": [f["plain"] for f in c["fields"]][:14],
                "n_fields": len(c["fields"]), "conf": [f["confidence"] for f in c["fields"]][:14],
                "departments": c["departments"][:3], "conditions": c["conditions"][:4], "urgency": c["urgency"],
                "danger": c["danger_pick"], "second": c["second_reader"], "verdict": c["verdict"], "facts": c["facts"],
                "ask": c["ask_next"], "truth": c["truth"]})
json.dump(out, open(f"{R}/explainer/cases.json", "w", encoding="utf-8"), ensure_ascii=False)
for o in out:
    print(o["title"], "|", o["verdict"]["word"], "|", o["departments"][0]["label"] if o["departments"] else None, "| truth", o["truth"]["department"])

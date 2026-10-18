"""Run the app's engine on the hand-written sentences (written by Claude, NOT a clinician; an honest first check of real phrasing).

usage: run_env.sh app/handwritten_eval.py [jsonl]
For each: what the translator read, the route, the urgency, the verdict, whether the route matches the expected department.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # the repo folder

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine          # noqa: E402
from server import DEFAULT_CONFIG  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data_built/handwritten_by_claude.jsonl"
E = Engine()
print("models:", E.models)
ok = auto = auto_ok = 0
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
for r in rows:
    age_sex, _, words = r["text"].partition(" ")
    age, sex = age_sex[:-1], age_sex[-1]
    text, fields, conf = E.read_words(age, sex, words)
    d = E.decide(text, fields, [], conf, DEFAULT_CONFIG)
    hit = d["department"] == r["expect"]
    ok += hit
    routed = d["verdict"] in ("yes", "maybe yes")
    auto += routed
    auto_ok += routed and hit
    print(f"{'OK ' if hit else '-- '} #{r['id']:2d} {d['verdict']:16s} -> {d['department']:22s} {d['urgency']:11s} (expect {r['expect']}) "
          f"| read {len(fields)}: {'; '.join(E.plain(f) for f in fields)[:160]}")
print(f"\nroute matches expectation: {ok}/{len(rows)} | routed on its own: {auto} ({auto_ok} of them right) | the rest go to a person")

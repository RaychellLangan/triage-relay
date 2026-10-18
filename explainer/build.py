"""Inline cases.json into template.html -> triage_relay.html (the explainer) and the explorer's served copy (a build output)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
c = json.dumps(json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8")), ensure_ascii=True).replace("</", "<\\/")
page = t.replace("__CASES__", c)
for out in ("triage_relay.html",):
    open(os.path.join(HERE, out), "w", encoding="utf-8").write(page)
print("ok; non-ascii characters in the page:", sum(ord(ch) > 127 for ch in page))

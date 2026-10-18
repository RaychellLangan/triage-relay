"""Write a handful of ward cards, one of each kind, to a markdown file. usage: run_env.sh route_from_fields/demo_cards.py <out.md>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ward_card import Chain, render, ROOT     # noqa: E402


def first(path, pred):
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if pred(r):
            return r


def main(out_path):
    chain = Chain(device=os.environ.get("PROBE_DEVICE"))
    facts = {(r["id"], r["v"]): r for r in map(json.loads, open(f"{ROOT}/runs/route_from_fields/verdict/facts_test2000.jsonl"))}
    contested = next(k for k, r in facts.items() if r["department_calm"] < 0 and r["router_agrees"] < 0)
    cases = [
        ("A clear case", first(f"{ROOT}/data_built/messy_test.jsonl", lambda r: r["stray_p"] == 0 and r["keep_p"] == 1.0)),
        ("A contested case (the evidence fights and the second reader disagrees)",
         first(f"{ROOT}/data_built/messy_test.jsonl", lambda r: (r["id"], r["v"]) == contested)),
        ("Five stray symptoms from another condition", first(f"{ROOT}/data_built/messystress_strays5.jsonl", lambda r: True)),
        ("A vague one-liner", {"text": "52m just not right.", "dept_true": "none"}),
        ("Not an intake at all", first(f"{ROOT}/data_built/messystress_ood.jsonl", lambda r: r["kind"] == "english")),
        ("An instruction planted in a real intake", first(f"{ROOT}/data_built/messystress_ood.jsonl", lambda r: r["kind"] == "injected")),
    ]
    parts = ["# Ward cards: demo (13 Oct 2026)\n",
             "Every word on a card is a slot filled from data; nothing is generated. Synthetic patients (DDXPlus). Not a diagnosis.\n"]
    for title, r in cases:
        c = chain.card(r["text"])
        parts.append(f"\n## {title}\n\n**Input:** {r['text'][:600]}\n\n**Recorded department:** {r.get('dept_true')}\n\n```\n{render(c)}\n```\n")
    open(out_path, "w", encoding="utf-8").write("".join(parts))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])

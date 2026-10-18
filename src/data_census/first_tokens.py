"""Do the department and urgency names each start with a different GPT-2 token? usage: first_tokens.py <routing_table.json> <gpt2 dir>"""
import json
import sys

from transformers import GPT2TokenizerFast

table = json.load(open(sys.argv[1], encoding="utf-8"))
tok = GPT2TokenizerFast.from_pretrained(sys.argv[2])
for field in ("department", "urgency"):
    names = sorted({r[field] for r in table.values()})
    firsts = {n: tok.convert_ids_to_tokens(tok(" " + n)["input_ids"])[0] for n in names}
    print(field, "unique" if len(set(firsts.values())) == len(names) else "COLLIDE", firsts)

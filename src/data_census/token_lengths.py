"""How long are the intake texts in GPT-2 tokens? (GPT-2's window is 1,024.)

usage: python token_lengths.py <ddxplus_split.jsonl> <gpt2 tokenizer dir> [n]
"""
import json
import sys

from transformers import GPT2TokenizerFast


def main(path, tok_dir, n=20000):
    tok = GPT2TokenizerFast.from_pretrained(tok_dir)
    lengths = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= int(n):
                break
            lengths.append(len(tok(json.loads(line)["text"])["input_ids"]))
    lengths.sort()
    q = {p: lengths[int(p / 100 * (len(lengths) - 1))] for p in (5, 50, 95, 99)}
    print(f"{len(lengths)} texts; tokens p5 {q[5]}, median {q[50]}, p95 {q[95]}, p99 {q[99]}, max {lengths[-1]}; "
          f"over 1024: {sum(l > 1024 for l in lengths)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

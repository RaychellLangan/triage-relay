"""Write the novelty sensor's cases to a small file: exactly the 6,000 lines its seeded draw takes from a training file, in
order. The app then ships 6,000 lines instead of the whole training file, and reads them with pick_rate=1.

    python pick_bank.py data_built/natural2_train.jsonl data_built/natural2_train.picked.jsonl
"""
import random
import sys

src, out = sys.argv[1], sys.argv[2]
rng, kept, want = random.Random(17), [], 1000 + 4000 + 1000      # the sensor's seed and n_calib + n_bank + n_ref
for line in open(src, encoding="utf-8"):
    if rng.random() < 0.015:                                       # the sensor's own pick_rate
        kept.append(line if line.endswith("\n") else line + "\n")
    if len(kept) >= want:
        break
open(out, "w", encoding="utf-8", newline="\n").writelines(kept)
print(f"{len(kept)} cases -> {out}")

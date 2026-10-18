# Monitors: design (13 Oct 2026, written while the first models train). Nothing here is built yet.

Everything is read at ONE position: the last prompt token ("Route:"), before the model writes anything.
No reading learns at runtime. Centres are averages; bands are placed in measured gaps; lever sizes are written down.

## 1. Readings
| Reading | What it is | Where |
|---|---|---|
| MOUTH pick / margin / entropy | restricted softmax over the 14 department first-tokens | logits at the cue |
| MIDDLE pick / margin / spread | cosine of the de-rogued state to each department's CENTRE; centre = mean state of fit-fold training patients whose true department is d | each block, raw output at the cue |
| TAIL variant | the same after dropping the top-k shared directions; k chosen on a held-out fit fold, drop-0 always reported beside it (board #1131: dropping can delete the signal) | each block |
| FAMILIAR | how far the state sits from the training cloud: nearest-centre cosine, z-norm against the calibration fold | middle blocks |
| DISPLACEMENT | final-hidden vs chosen-token-embedding cosine, mouth entropy, hidden-state norm (the conductor's three; calibrate_jailbreak's method: in-domain mu/sd, band placed in the measured gap) | last block + logits |
| SEVERE WEIGHT | mouth mass on Immediate + Very urgent (read one step later) | logits at the urgency cue |
| VOCABULARY BASELINE | the same centre read on MEAN-POOLED token embeddings of the prompt (at the cue position the embedding is always the same token, so the baseline must pool) | embeddings |

## 2. The seesaw (engine: seesaw/seesaw.py, ruled publishable 28 Sep; confirm before copying into the repo)
Planks, each level = "don't know":
- `mouth_agrees_with_middle` : levers = same pick? (fact), middle margin, mouth margin
- `middle_is_decided`        : levers = middle margin, middle spread
- `familiar_case`            : levers = FAMILIAR, DISPLACEMENT readings
- goal `safe_to_route`       : driven by the three above
Verdict from `safe_to_route`: yes / maybe yes / no idea / maybe no / no; FIGHT (losing side heavier than the margin) -> a person.
Alongside (not a trust question): the HEADS-UP list = departments holding real mouth mass + SEVERE WEIGHT, so a ward hears
"possibly yours, ETA n" without being told a diagnosis.

OPEN QUESTION: lever sizes. Hand-set (MYCIN style), or measured ONCE on a fit fold (e.g. the hit rate at each
reading band) and then frozen. The second is still non-learning at runtime but is a calibration step; say which.

## 3. Tests that can fail (all on the test split, nothing tuned on it)
1. Does `safe_to_route` track correctness better than MOUTH ENTROPY ALONE (ordinary confidence)? If not, the middle adds nothing.
2. Does the middle beat the VOCABULARY BASELINE? If the pooled embeddings already route as well, the "cathedral" story is vocabulary.
3. "Middle knew": count cases where the mouth is wrong and the middle right, and the reverse, per block. Controls: the same
   read on shuffled centres (must fall to chance).
4. Ambiguity vs error: split wrong answers by the ground-truth differential (flat vs sharp). Does "no idea" land on the flat ones
   and "fight" on the sharp-but-wrong ones?
5. Out of domain: non-medical text, garbled intake, adversarial instructions in the intake, symptom sets DDXPlus never generates.
   Does `familiar_case` fall? Is the mouth still confident (the overwrite)?
6. Workload: at the accuracy a human team would accept, what share of cases can route automatically, and what share goes to
   a person? (the brief's "measurable improvement")
7. Per arm: tiny vs gpt2 vs gpt2_frozen_head vs the over-trained gpt2_nostop checkpoints (does decoupling show up in the monitors?).

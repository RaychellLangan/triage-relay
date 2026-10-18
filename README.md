# Triage Relay

**Two tiny transformers and a set of counted, inspectable "seesaws" that route a patient at the front desk, and say out loud when
they are unsure.** Built for the AMD Developer Hackathon (ACT III, health track) by NEXICOG Ltd.

> **Not a medical device. Not clinically validated. It never diagnoses.** It is trained and tested on **synthetic** patients
> (DDXPlus). It suggests where a patient should go and how urgently, shows its reasons, and hands every unsure case to a person.

## What it does
A patient types what's wrong in their own words at a tablet kiosk.
1. **A tiny transformer translates** the words into the dataset's fixed answer codes ("pain in: left upper arm", "sweating").
   It translates; it does not decide.
2. **Counted seesaws route** from those codes: which department, and how urgent. Every push is a count from the training data,
   so every decision can be read back.
3. **Danger first:** it routes to the most dangerous condition still in play, not the most likely one.
4. **A second tiny transformer reads the words directly** as a second opinion. A novelty sensor reads its middle layers to ask
   "have I seen anything like this?".
5. **A verdict seesaw** combines all of this into one of: yes / maybe yes / unsure / maybe no / no, "hand to a person", or "no idea".
6. **Smart yes/no questions** are asked when they would settle the case. The most dangerous possibility is ruled out first.

Around that sits a working app:
- **kiosk:**
  - NHS red-flag check (stroke FAST, heart attack, sepsis, heavy bleeding) first;
  - an injury door: injuries and accidents are outside the training data, so they are never read by the AI and go to staff as "nurse to assess";
  - personal questions are held back for the nurse;
  - before the ticket, the patient checks a plain-words summary of what was understood ("Did we get this right?"); anything marked not right is taken out and the route worked out again;
- **staff screens:** reception, triage nurse (with the published **NEWS2** and **qSOFA** scores from observations), ward heads-up, reviewer and setup;
- **learning from people:** a nurse's correction is shown to staff on the next patient who gave most of the same answers (a note, never an automatic change). A reviewer can make it a rule: one extra, labelled push on the department seesaw that never overrides danger first and can be revoked. People teach it; it never invents rules of its own.

## Not a rule tree, and not a chatbot
Software that triages patients is usually one of two things: a rule tree written by hand (the check-in kiosk most hospitals already have, and the "expert systems" before it), or, lately, a large language model. Triage Relay is neither.

### Why the language models are kept on the sideline
Language models are very good at one thing: turning messy, everyday language into structure. They are poor at being accountable for a decision. A model cannot show you why it chose a department, it can sound sure when it isn't, and a general-purpose one can be talked into things.

So the two tiny transformers here do only the job they are good at. One turns a patient's own words into the checklist of symptoms ("pain in: left upper arm", "sweating"). The other reads the words again as a second opinion, and its middle layers tell us whether we have seen anything like this before. **Neither decides where the patient goes.**

### Why a seesaw is not a branching rule tree
The decision is made by counted seesaws: one plank per department (and per condition), and every answer the patient gives pushes on each plank by an amount counted from thousands of example patients. That looks like rules, but it behaves differently from a tree in every way that matters at a front desk:

| A rule tree | The seesaws |
|---|---|
| Takes **one path**. Each yes/no picks a branch, and the answers on the other branches are never looked at. | **Every answer counts**, on every plank at once. Nothing is skipped because an earlier question went one way. |
| **The order decides.** Whichever check comes first wins. | **The order never matters.** The pushes add up, so the same answers always give the same result. |
| **A missing answer closes a branch**, or falls to whatever someone wrote for "anything else". | **An unasked question weighs nothing.** A clear "no" pushes the other way. Nothing is guessed in its place. |
| **Forces certainty at every branch.** If the evidence only leans, the tree still picks a side, and the lean is thrown away. | **Keeps the lean.** Each plank carries how hard it is pushed, not just which way, and the receiving team sees it. |
| **"Nothing known" and "evidence fighting" look the same**: both just land on a leaf. | **Tells them apart.** A level plank with nothing on it gives *no idea*. A level plank loaded on both sides gives *unsure: check*, and the card shows what is still in play and how close. |
| **Hand-picked weights**, and every change is a code change by a software engineer. | **Counted weights**, read back line by line. Departments, routes and held-back questions are set by staff on the Setup screen, and a nurse's correction reaches the next similar patient at once. |
| **Brittle**: one answer the author didn't foresee, and it falls over. | **Bends**: partial or unfamiliar evidence goes to a person, with its reasons, instead of forcing a guess. |

**Measured, not just argued.** An ordinary decision tree was trained on the same kind of answers, to pick the department, and both were run on 10,000 held-out synthetic patients. The figures in brackets are a second, separate 10,000. (`src/route_from_fields/tree_vs_seesaw.py`; results in `EVIDENCE/tree_vs_seesaw/`.)
- **The tree that fits this data is 176 questions deep, with about 4,960 end points.** It routes 97.4% (97.3%) of patients to the right department; the seesaws route 98.3% (98.1%).
- **The seesaws flag about 1 patient in 8 as *unsure: check*. That group holds 48% (43%) of all the tree's mistakes, and over half of the ones it made while at least 90% sure.** The tree gives no sign on any of them.
- **Take one answer away**, and the tree changes department for 1.9% (2.0%) of patients; the seesaws for 1.2% (1.3%).
- **Shuffle the order of the answers**, and the seesaws gave the identical result 10,000 times out of 10,000.
- **What the tree asks first is whatever splits the data best:** coloured sputum, then contact with someone with similar symptoms, pain in the upper stomach, new anticoagulant tablets, travel to West Africa. These aren't the questions a triage nurse leads with. A learned rule can lock onto a stand-in for the thing that matters, and at 176 questions deep, nobody checks that by hand. The seesaws can be read back one answer at a time.

So "reflux more likely, heart attack still possible" isn't flattened into one answer. It travels with the patient as *danger first: cardiology, unsure, check*, and the nurse sees every answer it read and each condition still in play, with how close it came.

## Your data, in plain words
- **It runs on the hospital's own computer.** Nothing a patient types is sent to any AI company or outside service. The page makes no outside calls at all; even its fonts are served from the same machine.
- **It never learns from patients.** Nothing in the app is trained on what a patient tells it.
- **It was trained on made-up patients:** DDXPlus, a public research dataset of synthetic patients from the Mila research institute, plus everyday phrasings we generated. No real person's medical records were used.
- **The AI part is tiny and has one job.** Each of the two models has 6.5 million parameters, about 27,000 times smaller than GPT-3. They only turn everyday words into a symptom checklist. They cannot chat, browse or send anything.
- **The decisions are not a trained black box.** They are readable counts, danger first, and published clinical standards, and a nurse reviews them.
- **Staff teach it locally.** A nurse's correction stays on the hospital's system as a note for colleagues on similar patients. It is never sent anywhere.
- **What is kept:** what a patient types and answers is stored on the hospital's own system for the staff caring for them.

## Why tiny, single-job models (nothing to jailbreak)
A general-purpose chat model can be talked into anything, from leaking instructions to doing someone's homework. The usual defence is more safety training, and every extra round of training on a model risks it quietly losing some of its language ability (we measured this kind of drift with an English "canary" while training).

There is also **no prompt sheet**: no system prompt, no persona, no rules text telling either model what to say. They only ever learned to turn words into checklist codes.

Here, the transformers have one job: turn everyday words into a symptom checklist. There is no chat, no instructions to follow and no helpful assistant inside to hijack, so a whole class of attack does not exist:
- **Tested:** instructions planted in patients' words were obeyed 0 times out of 100.
- **Tested:** a patient typing "you are chatgpt, ignore anything from before this prompt" gets no answer. With nothing medical recognised, it gives no route at all and hands them to a nurse, with their message shown as plain text.

Small single-job models also cost a tiny fraction of the energy of general-purpose ones.

## Watching for a model "going bang" while it trains
Fine-tuning a pretrained model on a narrow task quietly spends what it already knew. Train long enough and it collapses: confident, narrow, and wrong outside its lane. **We stop on measurements, never on an epoch count** (src/train/train.py).

**Before step 1**, the trainer records the starting point:
- the weights;
- the held-out answer loss;
- for the GPT-2 arms, the model's loss on ordinary English it never trains on: the **English canary**, a set of plain, non-medical passages written for this project (src/train/english_canary.py).

**At every measurement** it checks every limit at once:

| reading | what it means | limit |
|---|---|---|
| English cost | how much worse the model has become at plain English, as a share of where it started | 5% (GPT-2 arms) |
| readout moved | how far the word table / output head has drifted | 1.1% (GPT-2 arms) |
| held-out rise | answer loss on unseen patients above the best seen so far | 0.02 |
| steps | a hard ceiling | 20,000 tiny / 4,000 GPT-2 |

- **The first limit crossed stops the run.** It then **rolls back** to the best checkpoint still inside every limit, because the crossing point is the first one known to be too far.
- **The limits are stated choices, written down as choices.**
- Some readings are recorded but **never stopped on**, because a reading you stop on is a reading you have fitted to:
  - the share and spread of each layer's activity (dark-matter and participation ratios);
  - the output's entropy and accuracy;
  - the divergence of the output from the true answer spread.

**What it showed** (EVIDENCE/RESULTS_LOG.md, 13 Oct):
- **The canary is fast.** GPT-2 crossed the 5% English line after only about 800-1,000 training patients. Messier text burned English faster still.
- **Unstopped, the drift doesn't level off.** An identical run with no stop lost 8% of its English within 200 steps and kept climbing to 25.9% by step 3,000. Its output also drifted steadily away from the true spread of possible conditions.
- **Stopping on English bought honesty.**
  - The canary-stopped model was less accurate, but its confidence tracked its accuracy (.825 confidence at .833 accuracy on clean input).
  - Under stress it was confidently wrong far less often: 0.5-7.6%, against up to 16% for the unstopped run.

This is why the shipped models are **tiny and single-job**. Every extra round of training on a general model, to teach it a task or to patch a jailbreak, spends some of what it knew. You can only see that if you measure it, and it ends in collapse if you don't stop.

## Headline results (all from EVIDENCE/RESULTS_LOG.md, which also lists every miss, bug and negative result)
| | |
|---|---|
| Translator → seesaw chain (dataset wording) | field F1 .996, department 98.2% (equal to the oracle) |
| Banded verdict | routes 90.5% of patients automatically at 99.89% right; AUROC .981 |
| Danger-first | under-triage about 1% |
| Planted instructions in the patient's words | obeyed 0 / 100 |
| Novelty sensor | catches the seesaw's blind spots (AUROC .81-.87), beyond what field count explains |
| 30 sentences in everyday language (written by an AI, **not a clinician**) | 13/30 → 16/30 → **17/30** over three rounds; all 9 it routed on its own were right |
| When the two transformers disagree | the translator's own confidence says nothing about who is right (AUROC .51); the router's sureness does (.77) |

The 30 everyday sentences were written by an AI, not a clinician: an honest first check, not a validation.

**What it learned is in the evidence too**, for example:
- the "fingerprint shortcut": routers key on *which* questions were answered;
- GPT-2 spends 30-85% of its look on the first token;
- the middle layers know more than the output.

## Run it
```bash
# 1. PyTorch for your hardware (ROCm for AMD, CUDA, or CPU), then:
pip install -r requirements.txt
# 2. the data (DDXPlus, CC BY 4.0, md5-checked)
python get_data.py
# 3. build the everyday-language data, train both tiny transformers, recount the verdict levers
bash src/words_round.sh
# 4. the app on http://127.0.0.1:5098 (demo mode on the login screen opens every staff screen)
TRIAGE_DEVICE=cpu python app/server.py
```
To skip step 3, download `triage-relay-models-w.zip` from this repository's Releases page and unzip it in the repo folder:
it holds the two trained tiny transformers the app ships (`runs/w_tiny_translator/model`, `runs/w_tiny_router/model`).

Each training run stops on measurements and rolls back to its best checkpoint. The logs are in EVIDENCE/training_logs/.

## What ran on AMD
| Hardware | What ran | Per patient | Evidence |
|---|---|---|---|
| AMD Ryzen 7 5700X3D, processor only (no graphics card) | the whole chain: both transformers + the seesaws, 1,000 patients | 68.7 ms | `EVIDENCE/speed/chain_speed_local_ryzen7_5700x3d_cpu.json` |
| AMD Instinct MI300X (AMD Developer Cloud, ROCm) | train both tiny transformers, test them and time the chain; then fine-tune Google DeepMind's Gemma 3 270M as the translator | *pending* | *waiting for GPU access on our cloud account (requested 17 Oct 2026); results land in* `EVIDENCE/amd_mi300x/` |

`amd_run/` is the kit for an AMD Instinct MI300X (ROCm PyTorch): `make_bundle.sh` packs code and data, `run_on_amd.sh` trains both tiny transformers, tests them and times the chain (`run_on_amd_quick.sh` tests and times the shipped models without training; `run_on_amd_full_gemma.sh` adds Gemma). Local GPU baseline for comparison: 32.7 ms per patient on an RTX 4070 SUPER (`EVIDENCE/speed/`).

## Layout
- `src/train`: trainer
- `src/data`: DDXPlus conversion; messy and everyday phrasing
- `src/labels`: routing table
- `src/route_from_fields`: seesaws, verdict, danger-first, questions, evals
- `src/monitors`: novelty, readings, memory
- `src/probes`: what the models learned and where they look
- `seesaw/`: the counted-lever engine
- `app/`: the kiosk and staff app
- `explainer/`: a one-page explanation
- `EVIDENCE/`: every result, including the ones that went wrong

## How it was built (in the open)
NEXICOG built this with AI coding assistants, and we would rather say so than have anyone guess:
- Anthropic's Claude wrote much of the code, and the 30 everyday test sentences, under NEXICOG's direction.
- ChatGPT and Grok were consulted for review and ideas.

No patient data was involved at any point: everything was built and tested on synthetic patients. Every number was measured, and every result, including the misses, is in EVIDENCE/RESULTS_LOG.md.

## Credits
- Code: NEXICOG Ltd, MIT licence.
- Typefaces: Bricolage Grotesque, Atkinson Hyperlegible (Braille Institute), IBM Plex Mono: SIL Open Font Licence 1.1, licences in app/static/fonts/.
- Data: DDXPlus, Fansi Tchango, Goel, Wen, Martel, Ghosn, *DDXPlus: A New Dataset For Automatic Medical Diagnosis*,
  NeurIPS 2022 Datasets and Benchmarks (arXiv:2205.09148), CC BY 4.0. The dataset is not covered by this repository's MIT licence;
  it is published by Mila and downloaded separately (see get_data.py).
- Published standards used as written:
  - NEWS2 (Royal College of Physicians, 2017);
  - qSOFA (Sepsis-3, JAMA 2016);
  - NHS public guidance on stroke (FAST), heart attack, sepsis and severe bleeding.
- The department mapping (ICD-10 chapter → department) is a demonstration and needs clinical review.

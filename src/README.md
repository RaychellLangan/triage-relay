# Triage relay: a transformer that translates, seesaws that decide, and a card for the ward

Messy intake language in -> a small transformer TRANSLATES it into the patient's fields -> a transparent seesaw routes the
fields to a department and an urgency -> a second transformer reads the text as a second opinion on the cases the seesaw
falls through on -> a verdict seesaw decides whether the case routes automatically, needs a person, or is "no idea" -> a
prebaked WARD CARD (every word a slot filled from data; nothing generated). It never names a condition.

Synthetic patients (DDXPlus, CC-BY). Not a diagnosis. Not medical advice. All numbers: `../runs/RESULTS_LOG.md`.

## Why this shape (the measurements that led here)
- A router trained to answer "which department" learned a FINGERPRINT (which questions got a yes; single exclusive history
  items decide alone; age and sex ignored) and stayed ~95% sure while failing under missing answers and stray symptoms.
- At decision time GPT-2 put 30-85% of its attention on the first token (a sink) and ~1-4% on what the patient said; ~85% of
  our original intake tokens were template.
- So the transformer does what transformers are good at, TRANSLATION into a fixed vocabulary, and the decision is made by
  counted, inspectable seesaws. (NEXICOG: "the output is a reduction of the input from messy language to a range of predefined
  outputs".) Relay on the residue: NEXICOG's idea, 13 Oct.

## Layout
- `labels/routing_table.py` department from ICD-10 chapter (demo mapping, needs a clinician's review), urgency from DDXPlus severity.
- `data/convert_ddxplus.py` structured intake text; `data/build_messy_data.py` messy desk-style text + field targets
  (+ MESSY_KEEP / MESSY_STRAYS stress sets); `data/build_ood_set.py` English passages, word salad, vague, injected.
- `train/train.py` one trainer: `--arm tiny|gpt2|gpt2_frozen_head --task router|translator --set ddxplus|messy`; stops on
  measurements (English canary, readout distance, held-out rise, max steps), rolls back, records geometry it never stops on.
- `route_from_fields/counted_table.py` the counted leans (0.5 ln P(f|d)/P(f|not d), +1 smoothing), frozen.
- `route_from_fields/seesaw_router.py` department / urgency planks on the 28 Sep seesaw engine (in the pile until cleared for the repo).
- `route_from_fields/translate_eval.py` free-running translation scored; `relay_real_chain.py` the relay; `verdict.py` facts ->
  counted levers -> verdict; `ward_card.py` the card; `demo_cards.py` one of each kind.
- `probes/` what_did_it_learn, what_is_it_keyed_on, where_it_looks, stresses. `monitors/` readings, trajectory, adds_information.

## Headline numbers (13 Oct; one seed; synthetic)
- Tiny translator (6.3M): field F1 .996; seesaw on its fields 98.2% department (= perfect-translator ceiling 98.1%).
- Relay: on the seesaw's fall-through cases the router is right 94.7% vs 91.2%; agreeing, 98.0%.
- Verdict: flags wrong answers at AUROC .976 (router confidence alone .902); 87% of cases auto-route at 99.7%.
- Injected instructions: 0/100 hijacked (the translator can only write field codes). English passages 30/30 refused.
  Vague one-liners: 35/40 "no idea" with questions to ask.

## Open problems
- Department planks saturate (runner-ups at -1.00): counted evidence over correlated fields double-counts.
- Under heavy stray symptoms the counted seesaw is calm-and-wrong on 20% of its calm cases; the router discounts strays better.
- Verdict levers counted on only 30 wrong validation cases.
- The messy language is generated from the dataset's own wording: a stand-in for real speech.
- AMD rule: the shown runs must happen on AMD hardware (local dev GPU is NVIDIA).

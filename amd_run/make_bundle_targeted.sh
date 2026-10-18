#!/usr/bin/env bash
# Pack the TARGETED AMD run: the quick bundle's contents + Gemma's training data (natural2 train/validate) + the Gemma stage and
# the injury probe. No raw patient zips (nothing here retrains our tiny models).
set -e
R="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=$R/amd_run/triage_bundle_targeted.tar.gz
cd $R
tar -czf $OUT --exclude=__pycache__ \
  src \
  seesaw/seesaw.py \
  runs/w_tiny_translator/model runs/w_tiny_router/model \
  data_built/natural2_train.jsonl data_built/natural2_validate.jsonl data_built/natural2_test.jsonl \
  data_raw/ddxplus_en/release_evidences.json data_raw/ddxplus_en/release_conditions.json \
  runs/route_from_fields/counted_table.json \
  amd_run/run_on_amd_targeted.sh amd_run/run_on_amd_quick.sh amd_run/run_gemma_on_amd.sh amd_run/time_chain.py \
  amd_run/gemma_injury_probe/INJURY_FORM.md amd_run/gemma_injury_probe/probe_sentences.jsonl amd_run/gemma_injury_probe/probe_injury_form.py
ls -la $OUT

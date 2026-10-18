#!/usr/bin/env bash
# Pack the QUICK AMD run: code + the two shipped tiny transformers (already trained) + the test data only. No training data.
set -e
R="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=$R/amd_run/triage_bundle_quick.tar.gz
cd $R
tar -czf $OUT --exclude=__pycache__ \
  src \
  seesaw/seesaw.py \
  runs/w_tiny_translator/model runs/w_tiny_router/model \
  data_built/natural2_test.jsonl \
  data_raw/ddxplus_en/release_evidences.json data_raw/ddxplus_en/release_conditions.json \
  runs/route_from_fields/counted_table.json \
  amd_run/run_on_amd_quick.sh amd_run/time_chain.py
ls -la $OUT

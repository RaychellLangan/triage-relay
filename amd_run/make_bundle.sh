#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Pack what the AMD droplet needs into one tarball (code + built data + the raw DDXPlus files the scripts read).
# Paths on the droplet mirror $ROOT under /root/triage, and run_on_amd.sh rewrites the one root path.
set -e
R="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=$R/amd_run/triage_bundle.tar.gz
cd $R
tar -czf $OUT \
  src \
  seesaw/seesaw.py \
  data_built/natural2_train.jsonl data_built/natural2_validate.jsonl data_built/natural2_test.jsonl \
  data_raw/ddxplus_en/release_evidences.json data_raw/ddxplus_en/release_conditions.json \
  data_raw/ddxplus_en_unzipped/release_test_patients \
  runs/route_from_fields/counted_table.json \
  amd_run/run_on_amd.sh amd_run/run_on_amd_full_gemma.sh amd_run/run_gemma_on_amd.sh amd_run/time_chain.py amd_run/gemma_injury_probe/INJURY_FORM.md amd_run/gemma_injury_probe/probe_sentences.jsonl amd_run/gemma_injury_probe/probe_injury_form.py
ls -la $OUT

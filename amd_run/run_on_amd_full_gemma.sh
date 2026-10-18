#!/usr/bin/env bash
# FULL + GEMMA AMD run (paid, about 2 to 3 hours), one upload:
#   1. the FULL run: train our two tiny transformers on the MI300X, test them, time the whole chain per patient;
#   2. the GEMMA stage: fine-tune Google DeepMind's Gemma 3 270M as the translator (time-capped, GEMMA_MINUTES, default 40),
#      test it, time the chain with it (router: the one just trained here), then the injury-form probe on three Gemma models.
# Step 1's results are kept whatever happens in step 2. Needs HF_TOKEN (her read token; Gemma terms accepted on that account).
cd /root/triage
OUT=/root/triage/amd_results; mkdir -p $OUT
bash amd_run/run_on_amd.sh || echo "full stage STOPPED early: see run.log" | tee -a $OUT/timings.txt
bash amd_run/run_gemma_on_amd.sh || echo "gemma stage STOPPED early: see run.log" | tee -a $OUT/timings.txt
echo "=== full+gemma done" | tee -a $OUT/timings.txt

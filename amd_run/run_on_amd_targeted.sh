#!/usr/bin/env bash
# TARGETED AMD run (paid, about 1 to 1.5 hours): no retraining of our tiny models.
#   1. the QUICK run: test the two shipped tiny transformers and time the whole chain per patient on the MI300X;
#   2. the GEMMA stage: fine-tune Google DeepMind's Gemma 3 270M as the translator (time-capped, GEMMA_MINUTES, default 40),
#      test it, time the chain with it, then the injury-form probe on three Gemma models (no training).
# Step 1's results are kept whatever happens in step 2. Needs HF_TOKEN (her read token; Gemma's terms accepted on that account).
# Unpacked at /root/triage; results in /root/triage/amd_results (Gemma's in amd_results/gemma).
cd /root/triage
OUT=/root/triage/amd_results; mkdir -p $OUT
bash amd_run/run_on_amd_quick.sh || echo "quick stage STOPPED early: see run.log" | tee -a $OUT/timings.txt
bash amd_run/run_gemma_on_amd.sh || echo "gemma stage STOPPED early: see run.log" | tee -a $OUT/timings.txt
echo "=== targeted done" | tee -a $OUT/timings.txt

#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# The Gemma stage: after run_on_amd.sh on the same droplet, or inside run_on_amd_targeted.sh (the targeted bundle carries the
# training data too). Training stops at its measured limits or after GEMMA_MINUTES (default 40), whichever comes first, then
# rolls back to its best measured checkpoint and saves it (train.py --max_minutes).
# Fine-tunes Google DeepMind's Gemma 3 270M as the TRANSLATOR (words -> checklist codes), the same single job as our tiny
# translator, on the MI300X, watched by the same English canary + stop-and-roll-back. Then tests it and times the chain
# (Gemma translator + our tiny router + the seesaws).
#
# Gemma's weights are gated: NEXICOG accepts Google's Gemma terms on huggingface.co/google/gemma-3-270m with her account and
# makes a READ token. On the droplet only:   export HF_TOKEN=<her read token>   then   bash amd_run/run_gemma_on_amd.sh
set -e
cd /root/triage
OUT=/root/triage/amd_results/gemma; mkdir -p $OUT
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face read token (Gemma terms accepted on that account)}"
export HF_HOME=/root/triage/.hf TOKENIZERS_PARALLELISM=false PYTHONPATH= TRIAGE_ROOT=/root/triage
grep -rl "$ROOT" --include=*.py --include=*.sh . | xargs -r sed -i 's#$ROOT#/root/triage#g'
python -c "import transformers, tokenizers, sys; sys.exit(0 if (transformers.__version__, tokenizers.__version__) == ('4.57.1', '0.22.1') else 1)" 2>/dev/null || pip install -q "transformers==4.57.1" "tokenizers==0.22.1"   # the versions tested on the PC (an older image cannot load Gemma 3)
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-3-270m', local_dir='/root/triage/gemma-3-270m'))" | tee $OUT/download.txt
cd src
start=$(date +%s)
python train/train.py --arm hf --hf_model /root/triage/gemma-3-270m --task translator --set natural2 --max_minutes ${GEMMA_MINUTES:-40} \
  --out /root/triage/runs_amd/gemma_translator > $OUT/train_gemma_translator.log 2>&1
echo "gemma translator trained in $(( $(date +%s) - start )) s" | tee -a $OUT/timings.txt
grep -E "^(arm|stopped|rolled|peak)" $OUT/train_gemma_translator.log | tee -a $OUT/timings.txt
python route_from_fields/translate_eval.py /root/triage/runs_amd/gemma_translator/model /root/triage/data_built/natural2_test.jsonl \
  /root/triage/runs/route_from_fields/counted_table.json $OUT/translate_test.jsonl 1000 > $OUT/translate_test.log 2>&1
cat $OUT/translate_test.jsonl.summary.json | tee -a $OUT/timings.txt
ROUTER=/root/triage/runs_amd/w_tiny_router/model              # trained here by the full run ...
[ -d $ROUTER ] || ROUTER=/root/triage/runs/w_tiny_router/model    # ... or the shipped one (the targeted run)
python /root/triage/amd_run/time_chain.py /root/triage/runs_amd/gemma_translator/model $ROUTER \
  /root/triage/data_built/natural2_test.jsonl $OUT/chain_speed.json 2>&1 | tee -a $OUT/timings.txt
echo "=== gemma stage done" | tee -a $OUT/timings.txt

# ---- the injury-form probe (NEXICOG's idea): can an UNTRAINED Gemma fill the fixed injury form for cases the tiny translator
# cannot read? No training; answers are scored from the form's own list only (amd_run/gemma_injury_probe/).
# Our 124M GPT-2 stand-in got 3 of 29 "injury or not" right locally: the floor to beat.
cd /root/triage
python amd_run/gemma_injury_probe/probe_injury_form.py /root/triage/gemma-3-270m $OUT/injury_probe_270m.json | tee $OUT/injury_probe_270m.txt
for m in gemma-3-270m-it gemma-3-1b-it; do      # instruction-tuned: asked through their chat template
  python -c "from huggingface_hub import snapshot_download; snapshot_download('google/$m', local_dir='/root/triage/$m')" && \
  python amd_run/gemma_injury_probe/probe_injury_form.py /root/triage/$m $OUT/injury_probe_$m.json --chat | tee $OUT/injury_probe_$m.txt
done
echo "=== injury probe done" | tee -a $OUT/timings.txt

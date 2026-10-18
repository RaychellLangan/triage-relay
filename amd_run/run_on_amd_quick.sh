#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# QUICK AMD run (~15-20 min): no training on the droplet. Tests the two shipped tiny transformers (trained on the PC, carried in
# the bundle) and times the whole chain per patient on the MI300X. Unpacked at /root/triage; results in /root/triage/amd_results.
set -e
cd /root/triage
OUT=/root/triage/amd_results; mkdir -p $OUT
grep -rl "$ROOT" --include=*.py --include=*.sh . | xargs -r sed -i 's#$ROOT#/root/triage#g'
export HF_HOME=/root/triage/.hf TOKENIZERS_PARALLELISM=false PYTHONPATH=
python -c "import transformers, tokenizers, sys; sys.exit(0 if (transformers.__version__, tokenizers.__version__) == ('4.57.1', '0.22.1') else 1)" 2>/dev/null || pip install -q "transformers==4.57.1" "tokenizers==0.22.1"   # the versions tested on the PC (an older image cannot load Gemma 3)
{ echo "date: $(date -u)"; echo "run: quick (models trained on the PC, tested and timed here)"; rocm-smi --showproductname 2>/dev/null || true
  python -c "import torch;print('torch',torch.__version__,'| hip',torch.version.hip,'| gpu',torch.cuda.get_device_name(0),'| available',torch.cuda.is_available())"; } | tee $OUT/hardware.txt
cd src
start=$(date +%s)
python route_from_fields/translate_eval.py /root/triage/runs/w_tiny_translator/model /root/triage/data_built/natural2_test.jsonl \
  /root/triage/runs/route_from_fields/counted_table.json $OUT/translate_test.jsonl 1000 > $OUT/translate_test.log 2>&1
echo "translate + route 1000 patients: $(( $(date +%s) - start )) s" | tee -a $OUT/timings.txt
cat $OUT/translate_test.jsonl.summary.json | tee -a $OUT/timings.txt
python route_from_fields/relay_real_chain.py /root/triage/runs/w_tiny_router/model $OUT/translate_test.jsonl \
  /root/triage/data_built/natural2_test.jsonl $OUT/relay_test.json > $OUT/relay_test.log 2>&1 || true
python /root/triage/amd_run/time_chain.py /root/triage/runs/w_tiny_translator/model /root/triage/runs/w_tiny_router/model \
  /root/triage/data_built/natural2_test.jsonl $OUT/chain_speed.json 2>&1 | tee -a $OUT/timings.txt
echo "=== all done" | tee -a $OUT/timings.txt

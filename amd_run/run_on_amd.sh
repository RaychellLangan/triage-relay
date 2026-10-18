#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Runs ON the AMD droplet (Linux, ROCm PyTorch). Unpacked at /root/triage. Writes everything to /root/triage/amd_results.
set -e
cd /root/triage
OUT=/root/triage/amd_results; mkdir -p $OUT
# 1. the code finds its folder through TRIAGE_ROOT
export TRIAGE_ROOT=/root/triage
export HF_HOME=/root/triage/.hf TOKENIZERS_PARALLELISM=false PYTHONPATH=
# 2. libraries (the PyTorch image has torch for ROCm; transformers may be missing)
python -c "import transformers, tokenizers, sys; sys.exit(0 if (transformers.__version__, tokenizers.__version__) == ('4.57.1', '0.22.1') else 1)" 2>/dev/null || pip install -q "transformers==4.57.1" "tokenizers==0.22.1"   # the versions tested on the PC (an older image cannot load Gemma 3)
# 3. which chip
{ echo "date: $(date -u)"; rocm-smi --showproductname 2>/dev/null || true
  python -c "import torch;print('torch',torch.__version__,'| hip',torch.version.hip,'| gpu',torch.cuda.get_device_name(0),'| available',torch.cuda.is_available())"; } | tee $OUT/hardware.txt
cd src
# 4. train the two tiny transformers the app ships (w_: natural phrasing + everyday words), same settings as on the PC, timed
for job in "w_tiny_translator --task translator" "w_tiny_router --task router"; do
  set -- $job; name=$1; shift
  start=$(date +%s)
  python train/train.py --arm tiny --set natural2 --out /root/triage/runs_amd/$name "$@" > $OUT/train_$name.log 2>&1
  echo "$name trained in $(( $(date +%s) - start )) s" | tee -a $OUT/timings.txt
  grep -E "^(arm|stopped|rolled|peak)" $OUT/train_$name.log | tee -a $OUT/timings.txt
done
# 5. the tests, on the AMD-trained models
python route_from_fields/translate_eval.py /root/triage/runs_amd/w_tiny_translator/model /root/triage/data_built/natural2_test.jsonl \
  /root/triage/runs/route_from_fields/counted_table.json $OUT/translate_test.jsonl 1000 > $OUT/translate_test.log 2>&1
cat $OUT/translate_test.jsonl.summary.json | tee -a $OUT/timings.txt
python route_from_fields/relay_real_chain.py /root/triage/runs_amd/w_tiny_router/model $OUT/translate_test.jsonl \
  /root/triage/data_built/natural2_test.jsonl $OUT/relay_test.json > $OUT/relay_test.log 2>&1 || true
# 6. speed of the whole chain per case
python /root/triage/amd_run/time_chain.py /root/triage/runs_amd/w_tiny_translator/model /root/triage/runs_amd/w_tiny_router/model \
  /root/triage/data_built/natural2_test.jsonl $OUT/chain_speed.json 2>&1 | tee -a $OUT/timings.txt
echo "=== all done" | tee -a $OUT/timings.txt

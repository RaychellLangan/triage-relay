#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Overnight queue, 13 Oct 2026. Each arm in turn, its own folder and stdout log under $ROOT/runs/.
cd $ROOT/src
R=$ROOT/runs
run(){ name=$1; shift; mkdir -p $R/$name; echo "=== $name start $(date)"; ./run_env.sh train/train.py --out $R/$name "$@" > $R/$name/stdout.log 2>&1; echo "=== $name exit $? $(date)"; grep -E "^(arm|stopped|rolled)" $R/$name/stdout.log; }
# tiny done 13 Oct 01:54 (kept step 1000, dept acc 0.997)
run gpt2               --arm gpt2              --n_val 512
run gpt2_frozen_head   --arm gpt2_frozen_head  --n_val 512
run gpt2_nostop        --arm gpt2 --n_val 512 --english_allowance 99 --readout_ceiling 99 --val_allowance 99 --max_steps 3000 --measure_every 50
echo "=== queue done $(date)"

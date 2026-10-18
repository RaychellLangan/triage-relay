#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Messy-language round, 13 Oct: router vs translator on the same messy intake. One at a time on the GPU.
cd $ROOT/src
R=$ROOT/runs
run(){ name=$1; shift; mkdir -p $R/$name; echo "=== $name start $(date)"; ./run_env.sh train/train.py --set messy --out $R/$name "$@" > $R/$name/stdout.log 2>&1; echo "=== $name exit $? $(date)"; grep -E "^(arm|stopped|rolled|peak)" $R/$name/stdout.log; }
run m_tiny_router        --arm tiny --task router
run m_tiny_translator    --arm tiny --task translator
run m_gpt2_router        --arm gpt2 --task router --n_val 512
run m_gpt2_translator    --arm gpt2 --task translator --n_val 256 --batch 8
run m_gpt2fh_translator  --arm gpt2_frozen_head --task translator --n_val 256 --batch 8
echo "=== queue done $(date)"

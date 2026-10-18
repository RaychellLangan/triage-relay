#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Readings for every kept router, one at a time on the GPU.
cd $ROOT/src
for r in tiny gpt2 gpt2_frozen_head gpt2_nostop; do
  echo "=== $r $(date)"
  ./run_env.sh monitors/readings.py $ROOT/runs/$r/model $ROOT/runs/readings/$r.npz 500 4000 > $ROOT/runs/readings/$r.log 2>&1
  echo "exit $?"
done
echo "=== done $(date)"

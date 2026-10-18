#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Stress probe on every kept router (GPU, one at a time, nothing else running).
cd $ROOT/src
for r in tiny gpt2 gpt2_frozen_head gpt2_nostop; do
  echo "=== $r $(date)"
  ./run_env.sh probes/what_did_it_learn.py $ROOT/runs/$r/model $ROOT/runs/probes/${r}_what_did_it_learn.json 1000 > $ROOT/runs/probes/${r}_probe.log 2>&1
  echo "exit $?"; grep -E "^(full|shuffled|reason|keep|stray)" $ROOT/runs/probes/${r}_probe.log
done

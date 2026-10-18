#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# The full chain + relay on every stress set (13 Oct). One at a time on the GPU.
cd $ROOT/src
R=$ROOT; O=$R/runs/route_from_fields
for s in strays5 keep30 keep50_strays3 ood; do
  echo "=== $s $(date)"
  ./run_env.sh route_from_fields/translate_eval.py $R/runs/m_tiny_translator/model $R/data_built/messystress_$s.jsonl $O/counted_table.json $O/stress_translate_$s.jsonl 1000 2>&1 | grep -E '^ "(field|exact|dept|urgency|fight)'
  ./run_env.sh route_from_fields/relay_real_chain.py $R/runs/m_tiny_router/model $O/stress_translate_$s.jsonl $R/data_built/messystress_$s.jsonl $O/stress_relay_$s.json 2>&1 | grep -E '^(all|seesaw)'
done
echo "=== done $(date)"

#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Verdict round 2 (13 Oct): levers counted on clean + STRESSED validation; novelty fact added. One GPU job at a time.
set -e
R=$ROOT; O=$R/runs/route_from_fields; V=$O/verdict2; mkdir -p $V
cd $R/src
B="data/build_messy_data.py $R/data_raw/ddxplus_en_unzipped/release_validate_patients $R/data_raw/ddxplus_en/release_evidences.json $R/data_raw/ddxplus_en/release_conditions.json labels/routing_table.json"
MESSY_STRAYS=5 python -I $B $R/data_built/messyval_strays5.jsonl 1000 1 61
MESSY_KEEP=0.3 python -I $B $R/data_built/messyval_keep30.jsonl 1000 1 62
MESSY_KEEP=0.5 MESSY_STRAYS=3 python -I $B $R/data_built/messyval_keep50_strays3.jsonl 1000 1 63
for s in strays5 keep30 keep50_strays3; do
  ./run_env.sh route_from_fields/translate_eval.py $R/runs/m_tiny_translator/model $R/data_built/messyval_$s.jsonl $O/counted_table.json $V/translate_val_$s.jsonl 1000 2>&1 | grep -E '"field_f1"'
done
# facts (with novelty) for every validation and test set
./run_env.sh route_from_fields/verdict.py facts $R/runs/m_tiny_router/model $O/translate_validate.jsonl $R/data_built/messy_validate.jsonl $O/counted_table.json $V/facts_val_clean.jsonl 2>&1 | grep facts
for s in strays5 keep30 keep50_strays3; do
  ./run_env.sh route_from_fields/verdict.py facts $R/runs/m_tiny_router/model $V/translate_val_$s.jsonl $R/data_built/messyval_$s.jsonl $O/counted_table.json $V/facts_val_$s.jsonl 2>&1 | grep facts
done
./run_env.sh route_from_fields/verdict.py facts $R/runs/m_tiny_router/model $O/translate_test2000.jsonl $R/data_built/messy_test.jsonl $O/counted_table.json $V/facts_test2000.jsonl 2>&1 | grep facts
for s in strays5 keep30 keep50_strays3 ood; do
  ./run_env.sh route_from_fields/verdict.py facts $R/runs/m_tiny_router/model $O/stress_translate_$s.jsonl $R/data_built/messystress_$s.jsonl $O/counted_table.json $V/facts_$s.jsonl 2>&1 | grep facts
done
cat $V/facts_val_*.jsonl > $V/facts_val_all.jsonl
echo "=== done"

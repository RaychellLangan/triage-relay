#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# After the natural round: stressed natural validation sets, translator chain, facts with the new models, recount banded verdict levers.
set -e
R=$ROOT; O=$R/runs/route_from_fields; V=$O/verdict_natural; mkdir -p $V
cd $R/src
B="data/build_messy_data.py $R/data_raw/ddxplus_en_unzipped/release_validate_patients $R/data_raw/ddxplus_en/release_evidences.json $R/data_raw/ddxplus_en/release_conditions.json labels/routing_table.json"
MESSY_NATURAL=0.6 MESSY_STRAYS=5 python -I $B $R/data_built/naturalval_strays5.jsonl 1000 1 71
MESSY_NATURAL=0.6 MESSY_KEEP=0.3 python -I $B $R/data_built/naturalval_keep30.jsonl 1000 1 72
export NOVELTY_BANK=$R/data_built/natural_train.jsonl
for s in natural_validate:2000 naturalval_strays5:1000 naturalval_keep30:1000; do
  IFS=: read f n <<< "$s"
  ./run_env.sh route_from_fields/translate_eval.py $R/runs/n_tiny_translator/model $R/data_built/$f.jsonl $O/counted_table.json $V/translate_$f.jsonl $n 2>&1 | grep -E '"(field_f1|dept_acc_via_seesaw|fight_share)"'
  ./run_env.sh route_from_fields/verdict.py facts $R/runs/n_tiny_router/model $V/translate_$f.jsonl $R/data_built/$f.jsonl $O/counted_table.json $V/facts_$f.jsonl 2>&1 | grep facts
done
cat $V/facts_*.jsonl > $V/facts_val_all.jsonl
VERDICT_BANDS=1 VERDICT_FACTS=department_calm,router_agrees,translator_sure,router_sure,familiar python -I route_from_fields/verdict.py fit $V/facts_val_all.jsonl $V/levers_banded.json | grep -E '"n_(right|wrong)"'
echo "=== natural verdict done"

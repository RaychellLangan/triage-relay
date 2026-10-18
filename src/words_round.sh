#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Words round (14 Oct): natural data with the everyday words the first natural vocabulary lacked ("headache", "earache", "worried").
# Writes NEW names (data natural2_*, models w_tiny_*, levers verdict_words) so the tested n_ models stay untouched until compared.
set -e
R=$ROOT; cd $R/src
B="data/build_messy_data.py $R/data_raw/ddxplus_en/release_evidences.json $R/data_raw/ddxplus_en/release_conditions.json labels/routing_table.json"
set -- $B
for split in train:200000:2:11 validate:5000:2:12 test:5000:2:13; do
  IFS=: read s n k seed <<< "$split"
  [ -s $R/data_built/natural2_${s}.jsonl ] && continue      # already built (a rerun after an interruption)
  MESSY_NATURAL=0.6 python -I $1 $R/data_raw/ddxplus_en_unzipped/release_${s}_patients $2 $3 $4 $R/data_built/natural2_${s}.jsonl $n $k $seed
done
for job in "w_tiny_translator --task translator" "w_tiny_router --task router"; do
  set -- $job; name=$1; shift; mkdir -p $R/runs/$name
  ./run_env.sh train/train.py --arm tiny --set natural2 --out $R/runs/$name "$@" > $R/runs/$name/stdout.log 2>&1
  grep -E "^(arm|stopped|rolled|peak)" $R/runs/$name/stdout.log
done
O=$R/runs/route_from_fields; V=$O/verdict_words; mkdir -p $V
B="data/build_messy_data.py $R/data_raw/ddxplus_en_unzipped/release_validate_patients $R/data_raw/ddxplus_en/release_evidences.json $R/data_raw/ddxplus_en/release_conditions.json labels/routing_table.json"
MESSY_NATURAL=0.6 MESSY_STRAYS=5 python -I $B $R/data_built/natural2val_strays5.jsonl 1000 1 71
MESSY_NATURAL=0.6 MESSY_KEEP=0.3 python -I $B $R/data_built/natural2val_keep30.jsonl 1000 1 72
export NOVELTY_BANK=$R/data_built/natural2_train.jsonl
for s in natural2_validate:2000 natural2val_strays5:1000 natural2val_keep30:1000; do
  IFS=: read f n <<< "$s"
  ./run_env.sh route_from_fields/translate_eval.py $R/runs/w_tiny_translator/model $R/data_built/$f.jsonl $O/counted_table.json $V/translate_$f.jsonl $n 2>&1 | grep -E '"(field_f1|dept_acc_via_seesaw|fight_share)"'
  ./run_env.sh route_from_fields/verdict.py facts $R/runs/w_tiny_router/model $V/translate_$f.jsonl $R/data_built/$f.jsonl $O/counted_table.json $V/facts_$f.jsonl 2>&1 | grep facts
done
cat $V/facts_natural2*.jsonl > $V/facts_val_all.jsonl
VERDICT_BANDS=1 VERDICT_FACTS=department_calm,router_agrees,translator_sure,router_sure,familiar python -I route_from_fields/verdict.py fit $V/facts_val_all.jsonl $V/levers_banded.json | grep -E '"n_(right|wrong)"'
echo "=== words round done"

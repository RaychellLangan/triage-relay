#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
# Natural-language round (14 Oct): data 60% everyday phrasing, retrain both tiny transformers on it.
set -e
R=$ROOT; cd $R/src
B="data/build_messy_data.py $R/data_raw/ddxplus_en/release_evidences.json $R/data_raw/ddxplus_en/release_conditions.json labels/routing_table.json"
set -- $B
for split in train:200000:2:11 validate:5000:2:12 test:5000:2:13; do
  IFS=: read s n k seed <<< "$split"
  MESSY_NATURAL=0.6 python -I $1 $R/data_raw/ddxplus_en_unzipped/release_${s}_patients $2 $3 $4 $R/data_built/natural_${s}.jsonl $n $k $seed
done
for job in "n_tiny_translator --task translator" "n_tiny_router --task router"; do
  set -- $job; name=$1; shift; mkdir -p $R/runs/$name
  ./run_env.sh train/train.py --arm tiny --set natural --out $R/runs/$name "$@" > $R/runs/$name/stdout.log 2>&1
  grep -E "^(arm|stopped|rolled|peak)" $R/runs/$name/stdout.log
done
echo "=== natural round done"

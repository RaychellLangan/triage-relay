#!/usr/bin/env bash
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # the repo folder
R=$ROOT
B="$R/src/data/build_messy_data.py $R/data_raw/ddxplus_en/release_evidences.json $R/data_raw/ddxplus_en/release_conditions.json $R/src/labels/routing_table.json"
set -- $B
for split in train:200000:2:1 validate:5000:2:2 test:5000:2:3; do
  IFS=: read s n k seed <<< "$split"
  python -I $1 $R/data_raw/ddxplus_en_unzipped/release_${s}_patients $2 $3 $4 $R/data_built/messy_${s}.jsonl $n $k $seed
done

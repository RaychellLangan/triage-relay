#!/usr/bin/env bash
# Run a project script with the repo's settings.
#   src/run_env.sh path/to/script.py [args...]      (paths relative to src/, e.g. train/train.py)
# Set PYTHON to choose the interpreter (default: python). TRIAGE_ROOT defaults to the repo folder.
ROOT="${TRIAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export TRIAGE_ROOT="$ROOT" HF_HOME="${HF_HOME:-$ROOT/_cache/hf}" TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -X utf8 "$@"

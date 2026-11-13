#!/bin/bash
# Second pass of the CPU analysis of the first Kaggle run: the steps that failed in queue_ingest.sh
# (Gemma CLadder steering analysis; Qwen 3B probes on the original pool and CLadder), then the summaries.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run () { echo "=== $(date +%H:%M:%S) $*"; "$@" || echo "!!! FAILED ($?): $*"; }
for S in "" _d0.5 _d0.85; do run $PY scripts/07b_steering_analysis.py --model gemma2b --tag cladder --suffix "$S"; done
run ./scripts/run_tier1.sh qwen3b 2
run $PY scripts/15_answer_bias.py
run $PY scripts/09_cross_model.py
run $PY scripts/12_tables.py
echo "=== $(date +%H:%M:%S) queue finished"

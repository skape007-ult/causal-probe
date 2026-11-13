#!/bin/bash
# CPU analysis of the first Kaggle run's caches (Qwen 3B: all probe sets; Gemma: debiased pool), then
# the steering re-analysis for every new steering file. Run under scripts/guarded.sh.
#   ./scripts/guarded.sh queue_ingest ./scripts/queue_ingest.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run () { echo "=== $(date +%H:%M:%S) $*"; "$@" || echo "!!! FAILED ($?): $*"; }
# the debiased pool first: it decides the headline
run ./scripts/run_debiased.sh qwen3b 2
run ./scripts/run_debiased.sh gemma2b 2
run $PY scripts/05b_probe_content.py --model qwen3b
run $PY scripts/05b_probe_content.py --model gemma2b
for M in gemma2b qwen3b; do
  for T in c2c_probe cladder c2c_debiased; do
    [ -e results/steering_${M}_${T}.json ] && run $PY scripts/07b_steering_analysis.py --model $M --tag $T
    for D in 0.5 0.85; do
      [ -e results/steering_${M}_${T}_d$D.json ] && run $PY scripts/07b_steering_analysis.py --model $M --tag $T --suffix _d$D
    done
  done
done
run ./scripts/run_tier1.sh qwen3b 2
run $PY scripts/15_answer_bias.py
run $PY scripts/09_cross_model.py
echo "=== $(date +%H:%M:%S) queue finished"

#!/bin/bash
# Probes and the audited comparison on the debiased Corr2Cause pool, for one model (CPU only).
# usage: scripts/run_debiased.sh <model> [jobs]     (needs cache/<model>/c2c_debiased)
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M="$1"; J="${2:-2}"
for C in 0.01 0.1 1; do
  $PY scripts/04_probes.py --model "$M" --tag c2c_debiased --jobs "$J" --C "$C"
done
$PY scripts/05_confound.py --model "$M" --tag c2c_debiased

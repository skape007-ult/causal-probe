#!/bin/bash
# Tier 1 analyses for one model, from the cache (CPU only).  usage: scripts/run_tier1.sh <model> [jobs]
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M="$1"; J="${2:-6}"
$PY scripts/02_behaviour.py --model "$M"
$PY scripts/03_logit_lens.py --model "$M"
for T in c2c_probe cladder; do
  for C in 0.01 0.1 1; do
    $PY scripts/04_probes.py --model "$M" --tag "$T" --jobs "$J" --C "$C"
  done
  $PY scripts/05_confound.py --model "$M" --tag "$T"            # (layer, C) chosen on the selection half
  $PY scripts/05_confound.py --model "$M" --tag "$T" --fixed-c  # original pre-registered C = 0.01 (sensitivity)
done
$PY scripts/04b_probe_sensitivity.py --model "$M"

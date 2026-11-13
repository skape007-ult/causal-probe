#!/bin/bash
# Tier 1 analyses for one model, from the cache (CPU only).  usage: scripts/run_tier1.sh <model> [jobs]
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M="$1"; J="${2:-6}"
$PY scripts/02_behaviour.py --model "$M"
# models cached on Kaggle come back with final-token activations for the probe sets only; their
# logit lens was computed there (results/lens_<model>.json is in results.zip)
if [ -e "${CAUSAL_PROBE_CACHE:-cache}/$M/c2c_test/resid.npy" ]; then $PY scripts/03_logit_lens.py --model "$M"; fi
for T in c2c_probe cladder c2c_debiased; do
  [ -e "${CAUSAL_PROBE_CACHE:-cache}/$M/$T/resid.npy" ] || { echo "no cache for $M/$T, skipped"; continue; }
  for C in 0.01 0.1 1; do
    $PY scripts/04_probes.py --model "$M" --tag "$T" --jobs "$J" --C "$C"
  done
  $PY scripts/05_confound.py --model "$M" --tag "$T"            # (layer, C) chosen on the selection half
  $PY scripts/05_confound.py --model "$M" --tag "$T" --fixed-c  # original pre-registered C = 0.01 (sensitivity)
done
$PY scripts/04b_probe_sensitivity.py --model "$M"

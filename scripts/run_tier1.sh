#!/bin/bash
# Tier 1 analyses for one model, from the cache (CPU only).  usage: scripts/run_tier1.sh <model> [jobs]
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python; M="$1"; J="${2:-6}"
$PY scripts/02_behaviour.py --model "$M"
$PY scripts/03_logit_lens.py --model "$M"
for T in c2c_probe cladder; do
  $PY scripts/04_probes.py --model "$M" --tag "$T" --jobs "$J"
  $PY scripts/05_confound.py --model "$M" --tag "$T"
done

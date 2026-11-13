#!/bin/bash
# Sequential overnight queue (one heavy job at a time on an 8 GB machine). Safe to re-run:
# every step skips itself if its output already exists.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
step () {  # step <output-file-that-marks-done> <log> <cmd...>
  local out="$1" log="$2"; shift 2
  if [ -e "$out" ]; then echo "[queue] skip $out"; return 0; fi
  echo "[queue] $(TZ=America/Toronto date '+%a %H:%M') start: $*"
  "$@" >> "logs/$log" 2>&1 || echo "[queue] FAILED: $*"
}
wait_for () { while pgrep -f "$1" > /dev/null; do sleep 20; done; }

wait_for "08_patching.py --model qwen0.5b"
wait_for "run_tier1.sh qwen1.5b"

# --- Tier 2 for qwen1.5b
step results/steering_qwen1.5b_c2c_probe.json tier2_qwen1.5b.log $PY scripts/07_steering.py --model qwen1.5b --tag c2c_probe
step results/steering_qwen1.5b_cladder.json   tier2_qwen1.5b.log $PY scripts/07_steering.py --model qwen1.5b --tag cladder
step results/patching_qwen1.5b.json           tier2_qwen1.5b.log $PY scripts/08_patching.py --model qwen1.5b --n-pairs 40 --n-comp-pairs 20 --layer-stride 2 --batch 8

# --- Gemma: cache (trained, then random-init), alone on the machine
MODELS=gemma2b ./scripts/build_cache.sh >> logs/queue_gemma_cache.log 2>&1

# --- Gemma Tier 1 (CPU) and Tier 2
step results/probe_sensitivity_gemma2b.json   tier1_gemma2b.log ./scripts/run_tier1.sh gemma2b 4
step results/steering_gemma2b_c2c_probe.json  tier2_gemma2b.log $PY scripts/07_steering.py --model gemma2b --tag c2c_probe
step results/steering_gemma2b_cladder.json    tier2_gemma2b.log $PY scripts/07_steering.py --model gemma2b --tag cladder
step results/patching_gemma2b.json            tier2_gemma2b.log $PY scripts/08_patching.py --model gemma2b --n-pairs 24 --n-comp-pairs 12 --layer-stride 2 --batch 1
echo "[queue] $(TZ=America/Toronto date '+%a %H:%M') all done"

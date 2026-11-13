#!/bin/bash
# Build a cache on the 8 GB laptop under the memory rules: capped GPU memory, small batches, and a
# watchdog that stops the job if swap grows by more than SWAP_LIMIT_MB while it runs.
#   scripts/guarded_cache.sh <args for 01_build_cache.py>
# Resumable: finished chunks are kept, so re-running continues.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
export CAUSAL_PROBE_TOKEN_BUDGET="${CAUSAL_PROBE_TOKEN_BUDGET:-900}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.82}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.72}"
LIMIT="${SWAP_LIMIT_MB:-1500}"
swap_mb () { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
mkdir -p logs
LOG="logs/guarded_$(echo "$@" | tr ' /,' '___').log"
nice .venv/bin/python scripts/01_build_cache.py "$@" > "$LOG" 2>&1 &
PID=$!
START=$(swap_mb)
echo "[guard] pid $PID, swap at start ${START} MB, limit +${LIMIT} MB, log $LOG"
while kill -0 $PID 2>/dev/null; do
  sleep 30
  NOW=$(swap_mb)
  echo "[guard] $(date +%H:%M:%S) swap ${NOW} MB (+$((NOW - START)))  $(grep -ao 'chunk [0-9]*/[0-9]* [0-9.]* items/s' "$LOG" | tail -1)"
  if [ $((NOW - START)) -gt "$LIMIT" ]; then
    echo "[guard] swap grew by more than ${LIMIT} MB: stopping the job (finished chunks are kept)"
    kill $PID; wait $PID 2>/dev/null; exit 3
  fi
done
wait $PID; RC=$?
echo "[guard] job exited with code $RC"; grep -a "Error" "$LOG" | tail -1 | cut -c1-300
exit $RC

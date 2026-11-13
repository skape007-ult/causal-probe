#!/bin/bash
# Run one job on the 8 GB laptop under the memory rules: two CPU threads, niced, capped GPU memory,
# small batches, and a watchdog that stops the job if swap grows by more than SWAP_LIMIT_MB while
# it runs (exit code 3). Cache builds are resumable, so a stopped build can simply be re-run.
#   scripts/guarded.sh <log name> <command...>
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
export CAUSAL_PROBE_JOBS="${CAUSAL_PROBE_JOBS:-2}"
export CAUSAL_PROBE_TOKEN_BUDGET="${CAUSAL_PROBE_TOKEN_BUDGET:-900}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.82}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.72}"
LIMIT="${SWAP_LIMIT_MB:-1500}"
swap_mb () { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
mkdir -p logs
LOG="logs/$1.log"; shift
nice "$@" > "$LOG" 2>&1 &
PID=$!
START=$(swap_mb)
echo "[guard] $(date +%H:%M:%S) start: $*  (swap ${START} MB, limit +${LIMIT} MB, log $LOG)"
while kill -0 $PID 2>/dev/null; do
  sleep 30
  NOW=$(swap_mb)
  if [ $((NOW - START)) -gt "$LIMIT" ]; then
    echo "[guard] $(date +%H:%M:%S) swap ${NOW} MB (+$((NOW - START))): stopping the job"
    kill $PID; wait $PID 2>/dev/null; exit 3
  fi
done
wait $PID; RC=$?
echo "[guard] $(date +%H:%M:%S) exit code $RC, swap $(swap_mb) MB; last line: $(grep -av Warning "$LOG" | tail -1 | cut -c1-200)"
exit $RC

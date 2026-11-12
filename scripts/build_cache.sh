#!/bin/bash
# Rebuild the whole activation cache (resumable: finished pieces are skipped).
# Each python invocation stops itself after ~85 minutes and is simply re-run until done.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODELS="${MODELS:-qwen0.5b qwen1.5b gemma2b}"
mkdir -p logs
run_until_done () {   # $@ = args for 01_build_cache.py
  for attempt in 1 2 3 4 5 6 7 8; do
    $PY scripts/01_build_cache.py "$@" >> "logs/cache_$(echo "$@" | tr ' /' '__').log" 2>&1 && return 0
    echo "[build_cache] attempt $attempt for '$*' stopped (time budget or error); retrying" >&2
    sleep 5
  done
  return 1
}
for m in $MODELS; do
  run_until_done --model "$m" || echo "[build_cache] FAILED trained $m" >&2
  run_until_done --model "$m" --random || echo "[build_cache] FAILED random $m" >&2
done
echo "[build_cache] all done"

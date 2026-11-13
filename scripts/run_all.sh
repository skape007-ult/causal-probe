#!/bin/bash
# Regenerate every number, table and figure that can be computed WITHOUT running a language model:
# everything from the activation cache (cache/) plus the text-only benchmark analysis.
#
#   ./scripts/run_all.sh            # all models that have a cache
#   JOBS=2 ./scripts/run_all.sh     # probe workers (default 2; this is an 8 GB machine)
#
# Needs: data/ (scripts/download_data.py) and cache/ (scripts/build_cache.sh for the Qwen models;
# KAGGLE.md + scripts/kaggle_ingest.py for Gemma's debiased pool and the larger model).
# Not regenerated here, because they need model forward passes: steering (07), patching (08) and
# the SAE tier (11). Their result files in results/ are read as they are.
# Runs one model at a time; expect several hours on the laptop.
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
PY=.venv/bin/python; J="${JOBS:-2}"
CACHE_DIR="${CAUSAL_PROBE_CACHE:-cache}"

$PY scripts/download_data.py
$PY scripts/14_benchmark_baselines.py                      # text-only baselines on the benchmarks
for M in qwen0.5b qwen1.5b gemma2b qwen3b qwen7b-4bit; do
  if [ -e "$CACHE_DIR/$M/c2c_probe/meta.json" ]; then
    echo "=== $M ==="
    nice ./scripts/run_tier1.sh "$M" "$J"                  # behaviour, lens, probes, confound (all cached item sets)
  else
    echo "=== $M: no cache, skipped ==="
  fi
done
$PY scripts/15_answer_bias.py
$PY scripts/09_cross_model.py
$PY scripts/12_tables.py
$PY scripts/06_figures.py
$PY scripts/13_build_docs.py
echo "run_all: done. Tables in report/tables/, figures in figures/, documents rebuilt."

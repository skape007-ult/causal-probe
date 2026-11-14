#!/bin/bash
# Regenerate every number, table and figure that can be computed WITHOUT running a language model:
# everything from the activation cache (cache/) plus the text-only benchmark analysis.
#
#   ./scripts/run_all.sh            # all models that have a cache
#   JOBS=2 ./scripts/run_all.sh     # probe workers (default 2; this is an 8 GB machine)
#
# Needs: data/ (scripts/download_data.py) and cache/ (scripts/build_cache.sh for the Qwen models;
# KAGGLE.md + scripts/kaggle_ingest.py for Gemma's debiased pool and the larger model).
# Not regenerated here, because they need model forward passes: steering (07, 07c), patching (08),
# the SAE tier (11), the adjacency test (17) and the held-out steering test (07d). Their result files in
# results/ are read as they are (the adjacency files of the GPU runs must not be re-analysed here: the
# laptop's activations are not the GPU run's); the statistics of 07 are recomputed from the saved raw
# logit differences where those are present.
# Runs one model at a time; expect several hours on the laptop.
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
PY=.venv/bin/python; J="${JOBS:-2}"
CACHE_DIR="${CAUSAL_PROBE_CACHE:-cache}"

$PY scripts/download_data.py
$PY scripts/14_benchmark_baselines.py                      # text-only baselines on the benchmarks
for M in qwen0.5b qwen1.5b gemma2b qwen3b qwen7b-4bit gemma9b; do
  if [ -e "$CACHE_DIR/$M/c2c_probe/meta.json" ]; then
    echo "=== $M ==="
    nice ./scripts/run_tier1.sh "$M" "$J"                  # behaviour, lens, probes, confound (all cached item sets)
  else
    echo "=== $M: no cache, skipped ==="
  fi
done
# analyses that re-read saved intervention outputs or the debiased-pool cache (CPU only)
for M in qwen0.5b qwen1.5b gemma2b qwen3b qwen7b-4bit gemma9b; do
  D="$CACHE_DIR/$M/c2c_debiased"
  if [ -e "$D/resid.npy" ]; then
    nice $PY scripts/05b_probe_content.py --model "$M" --jobs "$J"          # ladder, incremental value, property probes
  fi
  for T in c2c_probe cladder c2c_debiased; do
    for SUF in "" _d0.5 _d0.85; do
      if [ -e "results/steering_${M}_${T}${SUF}.json" ] && [ -e "$CACHE_DIR/$M/$T/steering${SUF}.npz" ] && [ -e "$CACHE_DIR/$M/$T/resid.npy" ]; then
        $PY scripts/07b_steering_analysis.py --model "$M" --tag "$T" --suffix "$SUF"
      fi
    done
  done
done
for M in qwen0.5b qwen1.5b gemma2b qwen3b; do          # label-changing vs label-preserving patching pairs, within template
  if [ -e "$CACHE_DIR/$M/patching.npz" ] && [ -e "$CACHE_DIR/$M/patching_same.npz" ]; then $PY scripts/08b_patching_by_template.py --model "$M"; fi
done
$PY scripts/07e_steering_multiplicity.py                    # how many steering comparisons, how many first ranks
if [ -e "$CACHE_DIR/qwen3b/c2c_debiased/resid.npy" ]; then $PY scripts/07f_direction_geometry.py --model qwen3b; fi
$PY scripts/15_answer_bias.py
$PY scripts/09_cross_model.py
$PY scripts/12_tables.py
$PY scripts/06_figures.py
$PY scripts/13_build_docs.py
echo "run_all: done. Tables in report/tables/, figures in figures/, documents rebuilt."

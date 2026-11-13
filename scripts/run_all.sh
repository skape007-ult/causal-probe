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
# the SAE tier (11) and the intervention pass of the adjacency test (16). Their result files in
# results/ are read as they are; the statistics of 07 and 16 are recomputed from the saved raw
# logit differences where those are present.
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
# analyses that re-read saved intervention outputs or the debiased-pool cache (CPU only)
for M in qwen0.5b qwen1.5b gemma2b qwen3b qwen7b-4bit; do
  D="$CACHE_DIR/$M/c2c_debiased"
  if [ -e "$D/resid.npy" ]; then
    nice $PY scripts/05b_probe_content.py --model "$M" --jobs "$J"          # ladder, incremental value, property probes
    for F in "$D"/steering_feature_*.npz; do                              # adjacency interventions: recompute the statistics
      [ -e "$F" ] || continue
      case "$F" in *_partial.npz) continue;; esac
      S=$(basename "$F" .npz); S=${S#steering_feature_}                   # e.g. adj_hyp_best
      P=final; case "$S" in *_hyp_*) P=content_end;; esac
      $PY scripts/16_feature_causal.py --model "$M" --position "$P" --layer-spec "${S##*_}" --analyse-only \
        || echo "run_all: 16 --analyse-only skipped for $M $S (needs the cache at that token position)"
    done
  fi
  for T in c2c_probe cladder c2c_debiased; do
    for SUF in "" _d0.5 _d0.85; do
      if [ -e "results/steering_${M}_${T}${SUF}.json" ] && [ -e "$CACHE_DIR/$M/$T/steering${SUF}.npz" ] && [ -e "$CACHE_DIR/$M/$T/resid.npy" ]; then
        $PY scripts/07b_steering_analysis.py --model "$M" --tag "$T" --suffix "$SUF"
      fi
    done
  done
done
$PY scripts/15_answer_bias.py
$PY scripts/09_cross_model.py
$PY scripts/12_tables.py
$PY scripts/06_figures.py
$PY scripts/13_build_docs.py
echo "run_all: done. Tables in report/tables/, figures in figures/, documents rebuilt."

#!/bin/bash
# Local queue for Friday 13 Nov (one job at a time, each under scripts/guarded.sh). Re-runnable:
# a step is skipped if its output exists. A cache step stopped by the swap watchdog is retried
# up to three times (it resumes from finished chunks) after memory has settled.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
step () {  # step <output file> <log name> <command...>
  local out="$1" log="$2"; shift 2
  if [ -e "$out" ]; then echo "[queue] skip (exists): $out"; return 0; fi
  for attempt in 1 2 3; do
    ./scripts/guarded.sh "$log" "$@"; rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -ne 3 ] && { echo "[queue] FAILED ($rc): $*"; return $rc; }
    echo "[queue] stopped by the swap watchdog (attempt $attempt); waiting 90 s"; sleep 90
  done
  echo "[queue] GAVE UP after three watchdog stops: $*"; return 3
}
R=results; C=cache
# --- Step 3 (Tier 1 deadline): probes on the debiased pool
step $R/confound_qwen0.5b_c2c_debiased.json debiased_probes_qwen0.5b ./scripts/run_debiased.sh qwen0.5b 2
step $C/qwen1.5b-rand/c2c_debiased/meta.json cache_qwen1.5b_rand_debiased $PY scripts/01_build_cache.py --model qwen1.5b --random --tags c2c_debiased
step $R/confound_qwen1.5b_c2c_debiased.json debiased_probes_qwen1.5b ./scripts/run_debiased.sh qwen1.5b 2
# --- Tier 2 audit fixes, Qwen 0.5B: rerun at the current layer/C selection, debiased pool, two depths, matched controls
for T in c2c_probe cladder c2c_debiased; do
  step logs/steering_qwen0.5b_${T}_v2.marker steer_qwen0.5b_$T sh -c "$PY scripts/07_steering.py --model qwen0.5b --tag $T --sub-batch 16 && touch logs/steering_qwen0.5b_${T}_v2.marker"
  step $R/steering_analysis_qwen0.5b_$T.json steer_an_qwen0.5b_$T $PY scripts/07b_steering_analysis.py --model qwen0.5b --tag $T
done
for D in 0.5 0.85; do
  step $R/steering_qwen0.5b_c2c_probe_d$D.json steer_qwen0.5b_c2c_probe_d$D $PY scripts/07_steering.py --model qwen0.5b --tag c2c_probe --sub-batch 16 --rel-depth $D --suffix _d$D
  step $R/steering_analysis_qwen0.5b_c2c_probe_d$D.json steer_an_qwen0.5b_c2c_probe_d$D $PY scripts/07b_steering_analysis.py --model qwen0.5b --tag c2c_probe --suffix _d$D
done
for T in c2c_probe c2c_debiased; do
  step $R/steering_controls_qwen0.5b_$T.json steer_ctl_qwen0.5b_$T $PY scripts/07c_steering_controls.py --model qwen0.5b --tag $T --sub-batch 32
done
# --- Qwen 1.5B (3 GB of weights: small batches)
step $R/steering_controls_qwen1.5b_c2c_probe.json steer_ctl_qwen1.5b_c2c_probe $PY scripts/07c_steering_controls.py --model qwen1.5b --tag c2c_probe --sub-batch 8 --n 100
step $R/steering_qwen1.5b_c2c_debiased.json steer_qwen1.5b_c2c_debiased $PY scripts/07_steering.py --model qwen1.5b --tag c2c_debiased --sub-batch 8 --select-stride 2
step $R/steering_analysis_qwen1.5b_c2c_debiased.json steer_an_qwen1.5b_c2c_debiased $PY scripts/07b_steering_analysis.py --model qwen1.5b --tag c2c_debiased
echo "[queue] $(date '+%a %H:%M') all done"

# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Fri 2026-11-13 11:15 (Toronto).

## Where things stand

| Piece | State |
|---|---|
| Tier 1, three local models, original item sets | Done, audited (Qwen) / reviewed (Gemma), in tables, figures and `report/report.md` |
| Step 2: text baselines on the benchmarks | Done: `scripts/14_benchmark_baselines.py` → `results/benchmark_baselines.json`; note in `report/benchmark_text_baselines.md`; literature check in `notes/literature_check.md` |
| Step 3: debiased pool `c2c_debiased` | Pool defined and tested (block design, D21). Text baselines done. Cached for qwen0.5b (trained + random-init). **qwen1.5b cache in progress** (see below). Probes not yet run for any model |
| Step 3: answer bias | Done: `scripts/15_answer_bias.py` → `results/answer_bias.json`, table `answer_bias` |
| Step 4: Kaggle notebook | Pushed and tested locally: `notebooks/kaggle_gpu.ipynb` pinned to code commit `406d994`. Waiting for the user to run it and bring back the zips |
| Tier 2, Qwen models | Results exist. Independent audit launched 11:08 (subagent); write-up waits for it |
| EXPLAINER | Start here rewritten; primer and decisions D20–D23 added; Tier 2/3 Q&A still to write |

## Running now / to restart if interrupted

1. `CAUSAL_PROBE_TOKEN_BUDGET=1500 nice .venv/bin/python scripts/01_build_cache.py --model qwen1.5b --tags c2c_debiased --no-gen` (resumable; log `logs/cache_qwen1.5b_debiased.log`). The first attempt at the default batch size stalled in swap next to other processes and was stopped.
2. Then the same with `--random`.
3. Then, one at a time: `nice ./scripts/run_debiased.sh qwen0.5b 2` and `nice ./scripts/run_debiased.sh qwen1.5b 2` (probes at three regularisation strengths + audited comparison on the debiased pool).
4. Then steering on the debiased pool for both Qwen models: `scripts/07_steering.py --model <m> --tag c2c_debiased --sub-batch 8`.
5. Then `scripts/12_tables.py`, `scripts/06_figures.py`, `scripts/13_build_docs.py`; fill `<!-- T1-DEBIASED-TEXT -->` in `report/report_template.md`; step 6 of "Start here" in EXPLAINER.md.

## When the Kaggle zips come back

1. Unpack as in `KAGGLE.md` step 9. Check `results/kaggle_run_status.json` and `results/cache_agreement_gemma2b.json`.
2. `OMP_NUM_THREADS=2 nice ./scripts/run_tier1.sh qwen3b 2` (behaviour from exported logits; probes on c2c_probe, cladder, c2c_debiased; skips the lens, which was run on Kaggle).
3. `nice ./scripts/run_debiased.sh gemma2b 2`.
4. `scripts/15_answer_bias.py`, `09_cross_model.py` (needs `qwen3b` added to its loop if wanted), `12_tables.py`, `06_figures.py`, `13_build_docs.py`.
5. Compare the larger model's `steer_select_*` layer with the layer `05_confound.py` picks (D22).

## Deadlines

- Tonight 23:00: Tier 1 write-up with three local models, Steps 2–3. (Steps 2 and the Tier 1 prose are pushed; Step 3 needs the probe runs above.)
- Saturday 10:00: Kaggle notebook ready (done Friday 10:50).
- Sunday 10:00: packaging starts regardless. Sunday 14:00: final push.

## Still to do after that

- Tier 2 write-up after the audit: steering, patching, cross-model; Tier 2 Q&A.
- Tier 3 (SAE) write-up from Kaggle results; LoRA dropped unless time remains.
- README (`report/README_template.md` does not exist yet), report Audits and Limitations sections, `scripts/run_all.sh`, at least 25 professor questions, finalise YOUR_TURN.md (its helper names must be re-checked: the selection split is now `data.selection_mask`).
- Upstream issue U8 (random re-initialisation no-op): re-run its reproduction and mark verified/unverified.

# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Fri 2026-11-13 17:15 (Toronto).

**Scope (user, Fri 16:25):** the scope freeze is lifted; push the science until Sunday 10:00, then package (decision D27). Heavy runs on Kaggle and Colab, launched by the user.

## Where things stand

| Piece | State |
|---|---|
| Tier 1 and Steps 2–3 for Qwen 0.5B, 1.5B (and Gemma on the original sets) | Done, audited, written up |
| First Kaggle run (commit `406d994`) | Back and ingested Fri 16:40: all 25 stages ok. Gemma steering/patching/SAE; Qwen 3B cache, behaviour, lens, steering. Results committed |
| CLadder sample | Bug found and fixed (decision D28): the sample is pinned to a committed id list; Qwen 3B's CLadder cache is a different draw and is loaded by id |
| Qwen 3B and Gemma: probes, baselines, debiased-pool analysis, steering re-analysis | **Running**: `scripts/queue_ingest.sh` under `scripts/guarded.sh` (log `logs/queue_ingest.log`, started Fri 16:41, several hours of CPU) |
| Round-2 GPU code | Written and smoke-tested on Qwen 0.5B: `scripts/gpu_round2.py`, `07c` (end-of-hypothesis token), `08 --label-mode same`, `16_feature_causal.py` |
| Round-2 notebooks | `notebooks/kaggle_r2_qwen3b.ipynb`, `kaggle_r2_qwen7b.ipynb`, `kaggle_r2_gemma.ipynb`, `colab_r2_qwen1.5b.ipynb`; regenerate with `scripts/make_round2_notebooks.py <sha>`; guide in `KAGGLE.md` ("Round 2") |
| Prior-work check | `notes/prior_work.md` (agent, Fri); folded into the benchmark note |
| study/ | Draft being written by an agent into `study/templates/` (built by `13_build_docs.py`); round-2 slots to fill |
| Tier 3 (SAE) | Results in (`results/sae_*.json`), tables built (`sae_*`), text not written |

## If the laptop restarts

1. Re-run the CPU queue; every step skips nothing but is idempotent: `./scripts/guarded.sh queue_ingest ./scripts/queue_ingest.sh` (or run the remaining lines of that script by hand; check `logs/queue_ingest.log` for the last `===` line).
2. Qwen 0.5B round-2 jobs on the laptop (not started yet at 17:15):
   `./scripts/guarded.sh r2_local ./.venv/bin/python -u scripts/gpu_round2.py --run r2_qwen05b --models qwen0.5b --jobs controls_hyp,feature,patch_same --out kaggle_out`
3. Then `scripts/12_tables.py`, `06_figures.py`, `13_build_docs.py`.

## When a round-2 download arrives

`kaggle_in/<run>/` then `.venv/bin/python scripts/kaggle_ingest.py --run <run>` (runs: `r2_qwen3b`, `r2_qwen7b`, `r2_gemma`, `r2_qwen15b`). For `r2_qwen3b`, `r2_gemma`, `r2_qwen15b` the debiased-pool cache gains the end-of-hypothesis token: re-run `scripts/05b_probe_content.py --model <m>` afterwards (property probes by position). For `r2_qwen7b`: report behaviour first; if above chance on the debiased pool, run `./scripts/run_debiased.sh qwen7b-4bit 2` and `05b`.

## Still to do (in order)

1. Audit 4 (independent agent) on: the Kaggle results as ingested, the CLadder-sample handling, and each round-2 result as it arrives. The audit must attack the adjacency causal claim specifically.
2. Report: section 3 rows for Gemma and Qwen 3B on the debiased pool; section 4 (matched controls at two tokens, label-preserving pairs, adjacency test); section 5 (SAE); README; EXPLAINER "Start here", primer entries, figure guide, Tier 3 questions.
3. Figures for the round-2 results (check each visually).
4. study/: fill the round-2 slots.
5. Packaging from Sunday 10:00: `run_all.sh` dry check, final chat post (headline per tier, surprises against logged expectations, what audits caught, five things to verify by hand).

## Deadlines

- Sunday 10:00: packaging starts regardless. Sunday 14:00: final push.

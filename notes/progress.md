# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Fri 2026-11-13 15:10 (Toronto).

## Where things stand

| Piece | State |
|---|---|
| Tier 1, three local models, original item sets | Done, audited (Qwen) / reviewed (Gemma); in tables, figures, `report/report.md` |
| Step 2: text baselines on the benchmarks | Done: `scripts/14_benchmark_baselines.py`, `report/benchmark_text_baselines.md`, `notes/literature_check.md` |
| Step 3: debiased pool | Done and audited for Qwen 0.5B and 1.5B: caches, probes, audited comparison, post-hoc ladder analysis (`scripts/05b_probe_content.py`), report section 3.6. Gemma and the larger model: waiting for Kaggle |
| Step 4: Kaggle | Notebook launched by the user Friday ~11:20 on code commit `406d994`. Ingest: `notes/kaggle_ingest.md`, `scripts/kaggle_ingest.py` |
| Tier 2, Qwen models | Audited twice (`notes/audit_tier2_raw.md`, `notes/audit_debiased_raw.md`); steering re-analysed with signed slopes and matched controls; report section 4 rewritten |
| Audits | Three done and logged. A fourth is due on the Kaggle results (Gemma interventions, SAE, Qwen 3B) |
| EXPLAINER | Story, primer (incl. SAE), decisions D1–D26, figure guide, 37 questions. Tier 3 questions still to write |

## Nothing is running locally

All local jobs finished Friday 15:01. Every local model job goes through `scripts/guarded.sh` (decision D26).

## When the Kaggle zips come back (user will say so)

1. `.venv/bin/python scripts/kaggle_ingest.py` (validates, extracts, regenerates tables/figures/documents, prints the larger model's behaviour). Details: `notes/kaggle_ingest.md`.
2. Tell the user in a few lines whether Qwen 3B is above chance on the original and debiased pools (table `answer_bias`; also the per-template view once step 4 has run). That decides whether Kaggle quota goes to a 7B model.
3. `OMP_NUM_THREADS=2 ./scripts/guarded.sh tier1_qwen3b ./scripts/run_tier1.sh qwen3b 2` (probes on c2c_probe, cladder, c2c_debiased; skips the lens, which ran on Kaggle).
4. `./scripts/guarded.sh debiased_gemma2b ./scripts/run_debiased.sh gemma2b 2`, then `scripts/05b_probe_content.py --model gemma2b` and `--model qwen3b` (the end-of-premise check is skipped automatically for final-token-only caches).
5. `scripts/07b_steering_analysis.py` for every new `steering_<model>_<tag>[suffix].json` (Gemma, Qwen 3B; suffixes `_d0.5`, `_d0.85`). The Kaggle run has three random directions only: no matched controls for those models unless a second session runs `07c_steering_controls.py`.
6. `scripts/15_answer_bias.py`, `09_cross_model.py`, `12_tables.py`, `06_figures.py`, `13_build_docs.py`.
7. Fourth independent audit on the new results; then update README, report (sections 3.6, 4, 5) and EXPLAINER.
8. Compare the larger model's `steer_select_*` layer with the layer `05_confound.py` picks (decision D22).

## Deadlines

- Tonight 23:00: Tier 1 write-up with three local models, Steps 2–3. (Steps 2 and the Tier 1 prose are pushed; Step 3 needs the probe runs above.)
- Saturday 10:00: Kaggle notebook ready (done Friday 10:50).
- Sunday 10:00: packaging starts regardless. Sunday 14:00: final push.

## Still to do after that

- Tier 3 (SAE) write-up from the Kaggle results; Tier 3 questions in EXPLAINER.
- Packaging (Sunday 10:00): README final pass (limitations section is a stub), `scripts/run_all.sh` dry check, final chat post (headline results per tier, surprises against logged expectations, what the audits caught, five things to verify by hand).
- Not done and listed as such: label-preserving minimal pairs for patching; steering at the end-of-hypothesis position; matched steering controls for Gemma and Qwen 3B.

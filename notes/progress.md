# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Fri 2026-11-13 14:15 (Toronto).

## Where things stand

| Piece | State |
|---|---|
| Tier 1, three local models, original item sets | Done, audited (Qwen) / reviewed (Gemma); in tables, figures, `report/report.md` |
| Step 2: text baselines on the benchmarks | Done: `scripts/14_benchmark_baselines.py`, `report/benchmark_text_baselines.md`, `notes/literature_check.md` |
| Step 3: debiased pool | Done for Qwen 0.5B and 1.5B: caches (trained + random-init), probes, audited comparison, post-hoc analysis (`scripts/05b_probe_content.py`), report section 3.6. Gemma and the larger model: waiting for Kaggle |
| Step 3: answer bias | Done (`scripts/15_answer_bias.py`, table `answer_bias`) |
| Step 4: Kaggle | Notebook launched by the user Friday ~11:20 on code commit `406d994`. Ingest: `notes/kaggle_ingest.md`, `scripts/kaggle_ingest.py` |
| Tier 2, Qwen models | Audited (report: `notes/audit_tier2_raw.md`); fixes applied; steering re-analysed with signed slopes and matched controls; report section 4 rewritten |
| Third audit | Launched Friday 14:10 on the debiased-pool claim and the steering wording. Not back yet |
| EXPLAINER | Story, primer, decisions D1–D26, figure guide, 36 questions. Tier 3 primer/questions still to write |

## Nothing is running locally

All local queues finished Friday 14:08. Every local job goes through `scripts/guarded.sh` (see D26).

## To do next, in order

1. Act on the third audit's findings; log them; adjust section 3.6 / 4.1 wording.
2. When the Kaggle zips arrive: section below.
3. Tier 3 write-up (SAE) from the Kaggle results; Tier 3 primer entries and questions.
4. Packaging (Sunday 10:00): README final pass, report summary (`<!-- SUMMARY -->` is still empty), `scripts/run_all.sh` dry check, final chat post.

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

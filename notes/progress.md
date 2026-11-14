# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Fri 2026-11-13 20:30 (Toronto).

**Scope (user, Fri 16:25):** the scope freeze is lifted; push the science until Sunday 10:00, then package (decision D27). Heavy runs on Kaggle and Colab, launched by the user.

## Where things stand (Fri 21:15)

| Piece | State |
|---|---|
| GPU runs launched by the user | **A (`r2_qwen3b`) and C (`r2_qwen15b`) from pin `5c31791`**, Fri evening. B (7B behaviour) and D (Gemma): user told to launch from the current pin `cdb08d7`. Follow-ups A2/C2/D2 (`*_r2b_*`, pin `cdb08d7`: cache, controls, controls_hyp, adjacency, export) after the main run of each model |
| Consolidation (D30) | `RESULTS.md` (from `report/RESULTS_template.md`) = top of README; report appendix A–D; fig13. Audited (sixth audit, `notes/audit_consolidation_raw.md`) and corrected |
| Laptop | **Nothing running.** All Qwen 0.5B round-2 work done; Qwen 3B and Gemma analysis done |
| Audits | Six done and logged. Not audited: the wording changes made in response to the sixth |
| Ingest | Fixed after the sixth audit: compares with the commit the run used; refuses smoke-test archives; conflicts saved as `<file>.kaggle` |

## The user's instructions for arriving runs (Fri 20:40)

1. Ingest and validate (`scripts/kaggle_ingest.py --run <run>`; runs from earlier pins are accepted with a note), then for that model: `17_adjacency_test.py --analyse-only` per site (positions final and content_end; layer-spec best and first50), `08b_patching_by_template.py`, `05b_probe_content.py` if the cache gained the end-of-hypothesis token, `12_tables.py`, `06_figures.py`, `13_build_docs.py`. Report in a few lines what it shows.
2. **7B rule (D30):** at chance on the debiased pool → state the scope (models up to 7B, these benchmarks, linear probes), finish the write-up, propose no further experiments. Above chance → stop, give a plan with GPU hours, use no more quota until the user answers.
3. Independent audit on anything new before it goes into `RESULTS.md`.
4. If a run fails: fix the notebook, give the new commit, say what can be reused.
5. Before going idle: a short status (done / running / needed from the user).

## If the laptop restarts

1. `./scripts/guarded.sh queue_ingest2 ./scripts/queue_ingest2.sh` (idempotent; Qwen 3B probes on c2c_probe and cladder take about an hour each).
2. Qwen 0.5B adjacency test and label-preserving patching (the runner skips finished stages):
   `./scripts/guarded.sh r2_local ./.venv/bin/python -u scripts/gpu_round2.py --run r2_qwen05b --models qwen0.5b --jobs adjacency,patch_same --out kaggle_out`
   Then, to add the transfer arm to the two `best` sites: delete `cache/qwen0.5b/c2c_debiased/steering_adjacency_final_best.npz` and `..._hyp_best.npz` and the markers `cache/_round2_markers_r2_qwen05b/r2.qwen0.5b.adjacency.final.best` and `...content_end.best`, and run the same command again.
3. `scripts/12_tables.py`, `06_figures.py` (check fig10–12 by eye), `13_build_docs.py`.

## When a round-2 download arrives

`kaggle_in/<run>/` then `.venv/bin/python scripts/kaggle_ingest.py --run <run>` (runs: `r2_qwen3b`, `r2_qwen7b`, `r2_gemma`, `r2_qwen15b`, and follow-ups `r2b_qwen3b`, `r2b_gemma`, `r2b_qwen15b`). The debiased-pool cache gains the end-of-hypothesis token for Qwen 3B and Gemma: re-run `scripts/05b_probe_content.py --model <m>` afterwards. For `r2_qwen7b`: report behaviour first; if above chance on the debiased pool, run `./scripts/run_debiased.sh qwen7b-4bit 2` and `05b`.

## Still to do (in order)

1. When `logs/r2_local3.guard` shows exit code 0: `12_tables.py`, `06_figures.py` (look at fig11, fig12), `13_build_docs.py`; add the "split by answer direction" result to report 4.4 (key `translation_split_by_answer_direction` in `results/adjacency_qwen0.5b_c2c_debiased_*.json`); log it against the tiny-run observation (answer-direction component carries the template-specific signs, remainder a uniform push towards No).
2. For each round-2 download: ingest, then `17_adjacency_test.py --analyse-only` for each site of that model (the ingested JSON may come from an older analysis), `08b_patching_by_template.py --model <m>`, tables, figures, documents; extend report 4.1 (matched controls for Gemma, Qwen 3B, 1.5B), 4.2, 4.4. The decisive case for the transfer test is Qwen 3B (class differences share a direction; observed gaps differ in sign).
3. A sixth audit on whatever the GPU runs add, before those numbers go into the README.
4. study/: fill the round-2 slots (label-preserving pairs; steering at the end of the hypothesis; the adjacency test and its two retractions; SAE; Qwen 3B and 7B).
5. Open items from the audits not yet done: SAE ablation with more random sets and per-item export (needs GPU); cross-fitted eraser; `sae_top_examples.md` announces "my reading" labels it does not contain.
6. Packaging from Sunday 10:00: `run_all.sh` dry check, final chat post (headline per tier, surprises against logged expectations, what audits caught, five things to verify by hand).

## Deadlines

- Sunday 10:00: packaging starts regardless. Sunday 14:00: final push.

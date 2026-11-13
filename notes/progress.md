# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Fri 2026-11-13 18:25 (Toronto).

**Scope (user, Fri 16:25):** the scope freeze is lifted; push the science until Sunday 10:00, then package (decision D27). Heavy runs on Kaggle and Colab, launched by the user.

## Where things stand

| Piece | State |
|---|---|
| Tier 1 and Steps 2–3, four models | Done. Qwen 3B and Gemma on the debiased pool analysed Fri evening (report 3.6 covers all four). Qwen 3B probes on the original pool and CLadder: **running** (`scripts/queue_ingest2.sh`, log `logs/queue_ingest2.log`) |
| First Kaggle run (commit `406d994`) | Ingested Fri 16:40, all 25 stages ok |
| CLadder sample | Pinned (decision D28); Qwen 3B CLadder, Gemma CLadder steering and the SAE tier are on Kaggle's draw |
| Round-2 notebooks | **Pinned to `5c31791`** (Fri 18:25): `kaggle_r2_qwen3b`, `kaggle_r2_qwen7b`, `kaggle_r2_gemma`, `colab_r2_qwen1.5b`, plus follow-ups `*_r2b_*_adjacency` for runs launched from the 17:30 version (`9fdec33`). `4d6afe5` (18:10) is also valid; it lacks only the transfer arm. Guide: `KAGGLE.md` "Round 2" |
| Fourth audit (round-2 code) | Done: `notes/audit_round2_code_raw.md`, responses in `notes/log.md` 18:10. The first adjacency test (`16_feature_causal.py`, results `feature_causal_*`) is superseded by `17_adjacency_test.py` (results `adjacency_*`) |
| Qwen 0.5B round 2 on the laptop | End-of-hypothesis steering controls done. Adjacency test **running** (`logs/r2_local.log`): `final_best` done WITHOUT the transfer arm, `hyp_best` running without it; `final_first50`, `hyp_first50` and `patch_same` follow with it |
| Tier 3 (SAE) | Written (report section 5), tables `sae_*` |
| Prior-work check | `notes/prior_work.md`, folded into the benchmark note |
| study/ | Draft built (`study/*.md` from `study/templates/`); round-2 slots to fill |
| Document review by the study agent | 15 inconsistencies; fixed so far: positive-control sentence, renaming, external check, step size, audit file list, EXPLAINER Q7, fig9 title, lens sentence, "pending" passages in sections 1, 2.2, 3.1, 3.6. Still open: EXPLAINER "Start here" (says three models), report Section 6 table wording about "the one positive verdict", the three TF-IDF values for one quantity (explain or unify), fig7 guide in EXPLAINER, which item set the 0.812 / 0.816 parse accuracy refers to |

## If the laptop restarts

1. `./scripts/guarded.sh queue_ingest2 ./scripts/queue_ingest2.sh` (idempotent; Qwen 3B probes on c2c_probe and cladder take about an hour each).
2. Qwen 0.5B adjacency test and label-preserving patching (the runner skips finished stages):
   `./scripts/guarded.sh r2_local ./.venv/bin/python -u scripts/gpu_round2.py --run r2_qwen05b --models qwen0.5b --jobs adjacency,patch_same --out kaggle_out`
   Then, to add the transfer arm to the two `best` sites: delete `cache/qwen0.5b/c2c_debiased/steering_adjacency_final_best.npz` and `..._hyp_best.npz` and the markers `cache/_round2_markers_r2_qwen05b/r2.qwen0.5b.adjacency.final.best` and `...content_end.best`, and run the same command again.
3. `scripts/12_tables.py`, `06_figures.py` (check fig10–12 by eye), `13_build_docs.py`.

## When a round-2 download arrives

`kaggle_in/<run>/` then `.venv/bin/python scripts/kaggle_ingest.py --run <run>` (runs: `r2_qwen3b`, `r2_qwen7b`, `r2_gemma`, `r2_qwen15b`, and follow-ups `r2b_qwen3b`, `r2b_gemma`, `r2b_qwen15b`). The debiased-pool cache gains the end-of-hypothesis token for Qwen 3B and Gemma: re-run `scripts/05b_probe_content.py --model <m>` afterwards. For `r2_qwen7b`: report behaviour first; if above chance on the debiased pool, run `./scripts/run_debiased.sh qwen7b-4bit 2` and `05b`.

## Still to do (in order)

1. Report section 4.4 (adjacency test): write from `results/adjacency_*.json` once Qwen 0.5B's four sites are in; extend as models arrive. Tables `adjacency_sites`, `adjacency_effects`, `adjacency_contrast` exist; a transfer-matrix table and figure are still to write. Label-preserving patching: table `patching_label_preserving`, figure `fig10`.
2. Fifth audit (independent agent) on the round-2 RESULTS and the claims drawn from them, the adjacency claim in particular (localisation vs feature).
3. README; EXPLAINER "Start here", figure guide (fig10–12), Tier 3 and round-2 questions; study/ round-2 slots; the open items of the document review above.
4. Packaging from Sunday 10:00: `run_all.sh` dry check, final chat post (headline per tier, surprises against logged expectations, what audits caught, five things to verify by hand).

## Deadlines

- Sunday 10:00: packaging starts regardless. Sunday 14:00: final push.

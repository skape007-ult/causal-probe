# Progress (read this first after a restart)

## MORNING STATUS, Saturday 14 November (written 01:45, updated 02:45; nothing was started that needs your input)

**In one line.** Qwen 3B and Qwen 1.5B are ingested and audited. One steering result beats its 20 controls and is undecided. `RESULTS.md` is unchanged. A run designed to decide it is ready for you to launch or not.

### 1. What the audit of the Qwen 3B and 1.5B results found (eighth audit, `notes/audit_round2_runs_raw.md`)

Every new number was recomputed from the raw outputs and agrees. On the Qwen 3B steering result (gold difference of class means at the end of the hypothesis, +0.74 logits per unit of s, larger than all 20 controls), your four questions:

| Question | Answer |
|---|---|
| Is it real? | As computed, yes. All 150 items, 13 blocks and 48 hypothesis strings have the sign. |
| One arm among many? | Yes: 1 of 12 comparisons on the debiased pool (24 in all). 0.57 first ranks expected by chance, 2 observed (probability 0.11). No arm survives correction. |
| Does it replicate? | Not at rank 1 anywhere: 5th of 21 at the final token of the same model; 10th and 5th at the same token in the other two models; 4th and 15th on the original pool. The sign repeats in all 12 arms. |
| Does it survive the within-string shuffle? | Yes; the auditor regenerated the controls and confirmed they are within-string and matched. |

Further: the direction is loaded on adjacency (cosine 0.69 with the adjacency contrast of direct-causation items), not on the model's answer, and is worth 0.02 logits at natural scale (the answer logit has mean −15). **Verdict: undecided between chance and a modest real effect of an adjacency-loaded direction; not evidence about validity. 100 controls would not settle it.**

Elsewhere: the adjacency test repeats the Qwen 0.5B picture in both models (a located answer shift at the final token, beyond all controls in Qwen 3B; nothing from translation or erasure at the end of the hypothesis). The audit corrected two readings and four table labels; those are applied.

A claims pass on the report and README then found 43 + 9 sentences that were stale or too wide now that three models have rows (8 contradicted by a result file). All fixed. Two things it found that nobody had reported: the "sign follows position, not validity" pattern of the label-preserving patching pairs holds for Qwen 0.5B only; and in Qwen 1.5B one arm of the adjacency test at the end of the hypothesis is borderline outside its controls.

### 2. Not changed, waiting for you

- **`RESULTS.md`.** Claim 3 is still worded for one model. Proposed rewording, evidence for and against, three options: **`notes/claim3_proposed_rewording.md`**. Nine smaller stale items in `RESULTS.md` are listed in `notes/claims_pass_14nov_raw.md`, list C.
- **Report Section 8** is still empty.

### 3. Ready to launch (all on repo `main`; first code cell shows `COMMIT = "e336431…"`)

| Notebook | Hours | HF secret | My recommendation |
|---|---|---|---|
| `notebooks/kaggle_r3_settle_qwen3b.ipynb` | about 1.5 | no | **Your call; read the caveat below first.** It has about one chance in three to one in two of an informative outcome. |
| `notebooks/kaggle_r3_settle_qwen1.5b.ipynb` | about 1 | no | The more informative of the two (a little better than even). Same test for the Qwen 1.5B arm. |
| `notebooks/kaggle_r3_gemma9b.ipynb` | 2 – 3.5 (ends in about 15 min if it cannot run) | **yes**, with the licence of `google/gemma-2-9b-it` accepted | Yes. |
| `notebooks/kaggle_r3_patch240_qwen3b.ipynb` | about 2 | no | Yes. |
| `notebooks/kaggle_r3_patch240_gemma.ipynb` | 1 – 2.5 | **yes** | Yes. |
| `notebooks/kaggle_r3_controls100_qwen3b.ipynb` (and `_qwen1.5b`, `_qwen0.5b`, `_gemma`) | 1 – 3 | Gemma only | **No longer recommended**: the audit showed 100 controls on the same split decide nothing. Ready if you want them anyway. |

**Caveat on the settle run: what it can and cannot show.** Its test and threshold are written down before any run (`notes/log.md`, the two "amendment" entries of 14 November). The script was reviewed twice before running (ninth and tenth audits). The first review showed my first design would have announced a replication in 57% of cases with nothing real behind it; the second confirmed the rewrite is correct and put numbers on its power:

| What is true | Qwen 3B: beyond all 500 controls / in between / p > 0.05 | Qwen 1.5B |
|---|---|---|
| Nothing label-specific | 3% / 51% / 46% | 2% / 26% / 72% |
| A real effect of the size observed | 25% / 67% / 9% | 48% / 44% / 8% |

"Beyond all 500" would be moderate evidence for a real effect; "p > 0.05" moderate evidence against; **the in-between outcome is evidence for neither**, and for Qwen 3B it is the most likely outcome whatever is true. So the run may well leave claim 3 exactly where it is. It costs about 2.5 GPU hours for both models. Not run on a GPU; the last five small changes after the second review have been seen by me alone.

### 4. What I need from you

1. **B (Qwen 7B) and D (Gemma) zips** in `kaggle_in/r2_qwen7b/` and `kaggle_in/r2_gemma/` (or `kaggle_out/<run>/` as you did last night; either works). Both were launched from pin `6d9ce8d`: they write no `STATUS.txt`, which the ingest accepts. **B has a known weak point at that pin:** if 4-bit loading failed and it fell back to two GPUs, the cache stage will show FAILED (a bug fixed one commit later). Then rerun `notebooks/kaggle_r2_qwen7b.ipynb` from the current repo; nothing from the failed run needs keeping.
2. **Your decision on claim 3**: option A, B or C in `notes/claim3_proposed_rewording.md`.
3. **Whether to launch the settle run**, knowing it may be uninformative (table above).

### 5. What I am unsure about

- Whether "No intervention shows…" is honest wording while one arm at p = 1/21 among 12 is perfectly consistent across items. I think yes; it is your call (the note says why).
- Whether the settle run is worth its GPU time (see the table). I would run the Qwen 1.5B one; for Qwen 3B it is closer to a coin flip whether it tells us anything. The error rates in that table assume the model responds to label-free directions about equally in all directions; if not, "beyond all 500" could happen by chance in up to about 12% of runs instead of 3%. The run measures that itself.
- Nothing involving two GPUs (B's fallback, Gemma-2-9B) has ever run. The Gemma-2-9B notebook checks itself and skips cleanly if it cannot run.
- The laptop could not hold a model last night without swapping (even Qwen 0.5B on CPU), so every laptop test of new GPU code used a stand-in for the model. The 7B probe ladder, when B arrives, is CPU-only analysis and should be fine.
- `scripts/run_all.sh` has not been run end to end since the second round; that is a packaging item for Sunday.

### 6. Things I got wrong overnight (all corrected, all in the log)

- My first reaction to the Qwen 3B result was to prepare 100 controls. The audit showed that would decide nothing.
- My first settle design could not have told a real effect from a reproducible artefact, and my second still had one comparison that an adjacency-only response would have triggered. Both caught by review before any run.
- A commit message claimed a log entry that was not in it (a step failed half-way). Added in the next commit.
- The figure script had been failing silently since Friday evening; two figures were stale for Qwen 3B. Fixed and checked.
- Several timestamps in my notes were guesses and ran ahead of the clock. Corrected where I found them.

---

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

**Compute rules (user instruction, Fri 13 Nov):** this 8 GB laptop does light work only: CPU analysis on existing caches, figures, writing, and Qwen 0.5B/1.5B GPU jobs one at a time. `OMP_NUM_THREADS=2`, background jobs under `nice`, stop any job that pushes memory into swap. **No Gemma model passes on this machine. Do not run `scripts/queue_night.sh`.** Gemma interventions, the SAE tier and the larger model run on Kaggle (`KAGGLE.md`), launched by the user.

Last updated: Sat 2026-11-14 01:45 (Toronto). The section above is current; the sections below are from Friday night and are kept for the restart instructions.

**Scope (user, Fri 16:25):** the scope freeze is lifted; push the science until Sunday 10:00, then package (decision D27). Heavy runs on Kaggle and Colab, launched by the user.

## Where things stand (Fri 23:30)

| Piece | State |
|---|---|
| GPU runs (launched by the user, two at a time) | **Running:** A (`r2_qwen3b`, pin `cdb08d7`; output at the top level of Kaggle's Output tab; no follow-up needed) and the Qwen 1.5B rerun (`r2_qwen15b`, pin `6d9ce8d`; replaces the lost C and C2). **Next, in the user's order:** B (`r2_qwen7b`, behaviour), D (`r2_gemma`), then round 3: E (`r3_gemma9b`), F (`r3p_qwen3b`, `r3p_gemma`: patching on all pairs), G (`r3c_*`: 100 control directions; 0.5B any time; the others only after their 20-control run is ingested and shows something beyond its controls, and only after re-pinning: a 100-control run refuses to start without the 20-control result in its checkout). All notebooks pinned to `08d99b1`. Table with files, pins and hours: `KAGGLE.md`, "Round 3" |
| Cut-offs (user, D32) | Nothing launched after **Sat 18:00**. Nothing unaudited into `RESULTS.md` after **Sun 08:00**. Packaging from Sun 10:00, final push by Sun 14:00 |
| Notebooks | Platform-chosen output directory; first cell stops on a missing secret, refused access to gated weights, or too few GPUs; last cell writes `STATUS.txt` (OK / FAILED / SKIPPED) and never raises; ingest refuses a run without OK (`--ignore-status` overrides). `scripts/notebook_cells.py`, `tests/test_notebook_cells.py` |
| Two-GPU path | `ResidCollector` fixed for models split over two GPUs (`041f11e`). **Never run on CUDA.** First real test: the preflight of E, or B's fallback |
| 7B probe ladder on the laptop | **On hold until the user brings B's zips** (user, Fri 22:45). Then `./scripts/run_debiased.sh qwen7b-4bit 2` (or `qwen7b` if the fallback ran) and `05b` |
| Analysis still to write for round 3 | `08b_patching_by_template.py` reads only the 60-pair files (needs a suffix option for `_n240`); tables and figures do not read `_c100` / `_n240` files yet. Do this when the first such run arrives, then audit |
| Seventh audit (round-3 code) | Done, applied in `08d99b1` (log Fri 23:30; raw report `notes/audit_round3_code_raw.md`). One critical (memory check would have refused real T4s), eight major. Nothing of round 3 has run on CUDA |
| Consolidation (D30) | `RESULTS.md` = top of README; audited (sixth audit) and corrected. Not audited: the rewording after the sixth audit |
| Laptop | CPU only while Kaggle runs go (user). Smoke tests of round 3 ran on CPU with Qwen 0.5B; the swap watchdog stopped one batch of them, the stages were then run one at a time |

## The user's instructions for arriving runs (Fri 20:40)

1. Ingest and validate (`scripts/kaggle_ingest.py --run <run>`; runs from earlier pins are accepted with a note). **For runs pinned at `bab7d1b` or later (all current ones) do NOT re-run `17_adjacency_test.py --analyse-only` on the laptop: the run's own JSON already has every readout and was computed on the activations the interventions used; a laptop re-analysis of Qwen 1.5B would mix the laptop's activations with Kaggle's outputs.** Only for a run from an older pin: `17_adjacency_test.py --analyse-only` per site (positions final and content_end; layer-spec best and first50), `08b_patching_by_template.py`, `05b_probe_content.py` if the cache gained the end-of-hypothesis token, `12_tables.py`, `06_figures.py`, `13_build_docs.py`. Report in a few lines what it shows.
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

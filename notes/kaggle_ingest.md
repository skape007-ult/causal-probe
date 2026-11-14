# Bringing the Kaggle results back

## 1. Where to put the downloads

Create a folder `kaggle_in/` in the repository root (it is git-ignored) and put the downloaded
files in it, with their original names:

```
causal-probe/
└── kaggle_in/
    ├── results.zip                required   (a few MB)
    ├── cache_qwen3b.zip           required   (about 1.2 GB)
    ├── cache_gemma2b.zip          required   (about 0.3 GB)
    ├── cache_qwen3b-rand.zip      optional   random-init control, about 1.2 GB
    └── cache_gemma2b-rand.zip     optional   random-init control, about 0.3 GB
```

Do **not** unzip them yourself. `results.zip` contains Kaggle's whole `results/` folder as of the
pinned commit, including old copies of files that have since been updated here, and the cache
zips contain per-item logits for item sets whose activations already exist on the laptop. A plain
`unzip -o` would overwrite newer local results and the metadata of the local Gemma cache.

## 2. The one command

```bash
.venv/bin/python scripts/kaggle_ingest.py
```

It does the following and prints each step:

1. **Files.** Tests every zip for corruption and lists size and file count.
2. **Code version.** Reads `results/kaggle_environment.json` from the zip and checks that the run
   used the commit pinned in `notebooks/kaggle_gpu.ipynb` (`406d994`). Prints the torch and
   transformers versions and the GPU the run had.
3. **Stages.** Compares `results/kaggle_run_status.json` with the list of 25 expected stages and
   prints `ok` / `FAILED` (with the last log lines) / `MISSING` for each, whether the run reached
   the export step, and which expected zips are absent. **If anything required is missing or
   failed it stops here, changes nothing and exits with code 1.**
4. **Results.** Extracts from `results.zip` only the files the run created or changed (each file
   is compared with its version at the pinned commit). A file changed both on Kaggle and locally
   is saved as `<name>.kaggle` and reported, never overwritten.
5. **Caches.** Extracts the cache zips into `cache/`. Folders that already hold local activations
   are left intact; Kaggle's logits for them are saved beside the local files as
   `meta_kaggle.json` / `readout_kaggle.npz`.
6. **Cache check.** Loads every extracted activation file and verifies its shape, that it is
   finite, and that its item ids match this repository's item lists in order.
7. **Fold-in.** Regenerates the answer-bias report, all tables, all figures, `report/report.md`,
   `report/benchmark_text_baselines.md` and `README.md`, then prints the larger model's behaviour
   on the original and debiased item sets (Yes-rate, accuracy against the majority rate, AUROC
   with confidence interval, accuracy at the best threshold).

It writes `results/kaggle_ingest_report.json` with what it found. It does not commit anything.

Useful variants:

```bash
.venv/bin/python scripts/kaggle_ingest.py --dry-run
```

validates and reports only (steps 1–3), changing nothing.

```bash
.venv/bin/python scripts/kaggle_ingest.py --allow-partial
```

ingests whatever is there even if stages are missing or failed (see below).

What the command does **not** do: the probe and text-baseline analysis for the larger model, which
is a few hours of CPU. That is the next step, run separately:

```bash
OMP_NUM_THREADS=2 nice ./scripts/run_tier1.sh qwen3b 2
```

## 3. What a complete run looks like

- Step 3 prints 25 lines starting with `ok` and "run reached the export step: yes".
- `kaggle_in/` holds `results.zip` plus at least `cache_qwen3b.zip` and `cache_gemma2b.zip`.
- The last line is "DONE: everything expected is present and consistent."

## 4. Partial runs (time-out, failed stage)

The runner refreshes `results.zip` after **every** stage, but it writes the `cache_*.zip` files
only at the very end. So a run that is cut short leaves:

| What you see in the notebook's Output | Meaning |
|---|---|
| `results.zip` only, no `cache_*.zip` | The run did not reach the export step (time-out or crash). JSON results of every finished stage are in the zip: Gemma steering, the larger model's behaviour, lens and steering, whatever completed. The larger model's activations are **not** available, so its probe analysis cannot run. |
| `results.zip` and all cache zips, but step 3 shows `FAILED` lines | The run finished; individual stages failed and the rest ran. The failed stage's log is in the zip under `results/kaggle_logs/<stage>.log`. |
| Nothing in Output | The version did not save its output. Check that the run was started with "Save & Run All (Commit)"; I do not know for certain whether Kaggle keeps `/kaggle/working` when a commit run hits the 12-hour limit. |

In a partial `results.zip`, `results/kaggle_run_status.json` has no `"export"` and no `"all_ok"`
key, and the stages that never started are simply absent from `"stages"`.

For a partial run:

```bash
.venv/bin/python scripts/kaggle_ingest.py --dry-run
```

shows exactly which stages are missing. Paste that output (or the notebook log) to me. Then
`--allow-partial` folds in what exists.

**Can a rerun reuse finished steps?** Not the GPU work itself: the activation cache lives on
Kaggle's temporary disk and is gone when the session ends, so a new "Save & Run All" rebuilds it
(about 10 minutes for Gemma, 25 for the larger model). What can be skipped are stages whose JSON
results are already in the partial `results.zip`: put their name prefixes into `SKIP_STAGES` in
the notebook's first cell, for example `SKIP_STAGES = "gemma.gemma2b.steer,gemma.gemma2b.patching"`.
Cache stages must not be skipped, because later stages and the export need them. In an
interactive session that is still alive, re-running the "run" cell resumes exactly where it
stopped.

## 5. If you paste errors from the Kaggle log

I need: the summary table from the last cell (or the `[FAILED ...]` lines), and the "log tail"
lines printed under the failed stage. If the install cell failed, the lines after `FAILED:`.
I will fix the code, re-pin the notebook and tell you the new commit and which stages to put in
`SKIP_STAGES`.

## 6. Round-2 runs

The four round-2 notebooks (KAGGLE.md, "Round 2") each write `results_<run>.zip` and
`cache_<run>_<model>.zip`. Put each run's downloads into `kaggle_in/<run>/` (zipped, or as the
folders a browser makes by unzipping them) and run

```bash
.venv/bin/python scripts/kaggle_ingest.py --run <run>
```

with `<run>` one of `r2_qwen3b`, `r2_qwen7b`, `r2_gemma`, `r2_qwen15b`. The checks are the same as
for the first run, with three differences: the expected commit is the one pinned in that run's
notebook; the expected stages are the ones the run itself planned (`results/round2_status_<run>.json`);
and a downloaded cache replaces the laptop's activations only when it holds strictly more token
positions for the same items (round 2 adds the end-of-hypothesis token for the debiased pool).

A partial round-2 run looks like a partial first run: the summary lists `FAILED` or `not run`
stages; `results_<run>.zip` exists from the first stage on; the cache zip exists only if the run
reached its export step. `--allow-partial` ingests what is there.

Note on the first run's downloads (13 Nov): Safari unpacked the zips into `results/`, `cache/`,
`cache-1/` ...; the ingest accepts that layout.

Runs launched from an earlier round-2 pin (the notebooks were re-pinned five times on 13 November) are accepted: the ingest
checks that the run's commit lies between the first round-2 pin and the current code and prints a note. Their result files
come from an older version of the analysis; after ingesting, the statistics are recomputed here from the raw arrays
(`17_adjacency_test.py --analyse-only` per site). Follow-up runs use the run names `r2b_qwen3b`, `r2b_qwen15b`, `r2b_gemma`.
A smoke test (`SMOKE_TEST = True`) writes under `<run>_tiny` and is refused by the ingest.

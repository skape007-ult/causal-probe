# Running the GPU work on Kaggle

The laptop has 8 GB of memory, which is too little for Gemma-2-2B interventions and for any
larger model. Those runs go to a free Kaggle GPU session. You run the notebook; the results come
back as zip files that you unpack into this repository.

## What the notebook does

`notebooks/kaggle_gpu.ipynb` clones this repository at a pinned commit, installs the pinned
requirements, and calls `scripts/kaggle_run.py`. That script runs these stages in order:

| Stage | Model | What | Output |
|---|---|---|---|
| cache | gemma-2-2b-it | Activations for the Corr2Cause pool, the debiased Corr2Cause pool and CLadder (rebuilt on the GPU; the laptop cache is not uploaded) | (kept on Kaggle; see export) |
| cache agreement | gemma-2-2b-it | Compares the rebuilt cache's answer logits with the laptop's (`results/ref_logit_diff_gemma2b.json`) | `results/cache_agreement_gemma2b.json` |
| steering | gemma-2-2b-it | Steering and ablation at the layer chosen on the laptop, on three item sets, plus two fixed depths (50% and 85%) as a layer-choice check | `results/steering_gemma2b_*.json` |
| cache, behaviour, lens | larger model | Full cache (all item sets, generated answers), accuracy tables, logit lens | `results/behaviour_*.json`, `results/lens_*.json` |
| steering | larger model | As for Gemma. The layer is chosen inside the selection half by a quick cross-validation, because the full probe analysis has not been run yet | `results/steering_*`, `results/steer_select_*` |
| random-init cache | larger model | Control activations from an untrained copy | (export) |
| patching | gemma-2-2b-it | Activation and attribution patching on 40 minimal pairs | `results/patching_gemma2b.json` |
| SAE | gemma-2-2b-it | Gemma Scope features by causal graph type, then feature ablation | `results/sae_features.json`, `results/sae_ablation.json`, `results/sae_top_examples.md` |
| export | both | Zips (below) | `/kaggle/working/kaggle_out/*.zip` |

It does **not** run probes or text baselines for the larger model. Those run on the laptop from
the downloaded cache, with the same scripts as the other models.

Each stage has a time limit, is skipped if it already finished, and can fail without stopping the
stages after it. A summary table is printed at the end and saved in
`results/kaggle_run_status.json`.

## Steps

1. **Hugging Face access to Gemma.** Log in at huggingface.co, open
   `https://huggingface.co/google/gemma-2-2b-it` and accept the licence. Create a *read* token
   under Settings → Access Tokens. (Qwen needs no token.)
2. **New notebook.** On kaggle.com: Create → New Notebook → File → Import Notebook → upload
   `notebooks/kaggle_gpu.ipynb` from this repository.
3. **Settings** (right-hand panel):
   - Accelerator: **GPU T4 x2** (P100 also works). This needs a phone-verified Kaggle account.
   - Internet: **On**.
   - Persistence: leave at the default.
4. **Secret.** Add-ons → Secrets → Add a new secret. Label: `HF_TOKEN`. Value: the token from
   step 1. Tick the checkbox that attaches it to this notebook. The notebook reads it from Kaggle
   Secrets into an environment variable and never prints or stores it.
   **A secret is attached per notebook.** The secret itself is stored once in your account, but every
   new or imported notebook starts with nothing attached: open Add-ons → Secrets in *that* notebook
   and tick `HF_TOKEN`. A notebook without it prints "No HF_TOKEN secret is attached to this
   notebook". The Gemma notebooks then stop in their first code cell; the Qwen notebooks continue
   (the Qwen weights are public), so for them the message is harmless.
5. **Optional 15-minute dry run.** In the first code cell set `SMOKE_TEST = True` and run the
   cells interactively (Run All). It pushes a few dozen items through every stage. If the summary
   shows every stage `ok`, set it back to `False`.
6. **Real run.** Save Version → *Save & Run All (Commit)* → Advanced settings → *Always save
   output*. You can close the browser; the run continues on Kaggle.
7. **Wait.** Estimated 3 to 4 hours on a T4 (see the breakdown below). Kaggle's limit is 12 hours
   per run and 30 GPU-hours per week.
8. **Download.** Open the finished version → Output → folder `kaggle_out`. Download:
   - `results.zip` (a few MB): every JSON result, the logs and the status file.
   - `cache_qwen3b.zip` (about 1.2 GB): final-token residuals of the trained model for the three
     probe sets, plus per-item logits and generated answers for the other item sets.
   - `cache_qwen3b-rand.zip` (about 1.2 GB): the random-init control. Optional; only the
     "random-init probe" column needs it. Set `EXPORT_RANDOM_INIT = False` to leave it out.
   - `cache_gemma2b.zip` and `cache_gemma2b-rand.zip` (about 0.3 GB each): Gemma on the debiased
     pool (the laptop already has Gemma's other caches).
9. **Put the zips into `kaggle_in/` in the repository root and run the ingest command.** Do not
   unzip them by hand (a plain unzip would overwrite newer local results and the metadata of the
   local Gemma cache). Details and what a partial run looks like: `notes/kaggle_ingest.md`.

   ```bash
   .venv/bin/python scripts/kaggle_ingest.py
   ```

   It checks the zips, the code commit and every stage, extracts only what the run produced,
   verifies the caches, and regenerates tables, figures and documents.
10. **Laptop analysis for the new model**, CPU only:

    ```bash
    OMP_NUM_THREADS=2 nice ./scripts/run_tier1.sh qwen3b 2
    ```

## Estimated run time (T4)

| Part | Minutes |
|---|---|
| Install pinned environment, download models and data | 15–25 |
| Gemma cache (7,800 items) + random-init (2,400) | 10–15 |
| Gemma steering, 7 runs of 300 items | 30–45 |
| Larger model: cache (≈11,000 items), generation, behaviour, lens | 25–40 |
| Larger model: steering, 5 runs, including layer selection | 40–60 |
| Larger model: random-init cache | 10–15 |
| Gemma patching (40 pairs, every second layer) | 20–40 |
| Gemma SAE features + ablation | 10–20 |
| Export and zip | 5 |
| **Total** | **about 3–4 hours** |

These are estimates from the laptop timings scaled to a T4; I could not run on Kaggle myself.
Hard upper bound from the per-stage time limits: about 11 hours.

## If something goes wrong

- **A stage shows FAILED.** The other stages still ran. The last lines of its log are printed in
  the summary and the full log is in `results.zip` under `results/kaggle_logs/`. Send me the log.
- **Re-running.** In an interactive session, re-running the "run" cell skips finished stages. A
  new "Save & Run All" starts from scratch (the cache lives on temporary disk).
- **Out of GPU memory in a cache stage.** Add a cell before the run cell with
  `os.environ["CAUSAL_PROBE_TOKEN_BUDGET"] = "2000"` and re-run.
- **Skipping a part.** `SKIP_STAGES = "gemma.gemma2b.sae,gemma.gemma2b.patching"` (stage-name prefixes), or
  `RUN_GEMMA = False`, or `LARGE_MODEL = None`. This is also how to split the work over two
  sessions.
- **The pinned install fails.** The install cell falls back automatically: full freeze → pinned
  direct dependencies → Kaggle's own packages. The level used is printed and the actual versions
  are saved in `results/kaggle_environment.json`. At level 3 versions differ from the laptop;
  treat those results as provisional and tell me.
- **"NO GPU FOUND".** The accelerator is not set; see step 3.

## The 7B option

`LARGE_MODEL = "qwen7b-4bit"` loads Qwen2.5-7B-Instruct in 4-bit (bitsandbytes NF4). This code
path could not be tested locally (bitsandbytes needs CUDA). Limitations if you use it:
activations and logits are those of the quantised model, there is no random-init control, and the
exported cache is about 1.9 GB. The default is the 3B model in float16, which was the tested path.

## What is tested and what is not

- Tested locally: every stage of `scripts/kaggle_run.py`, end to end, on Qwen2.5-0.5B with a small
  subset (`--tiny`), including resume, export and unpacking the export back into `load_cache`. The
  SAE stages were exercised with a random dictionary in place of Gemma Scope, because no Gemma
  forward pass runs on the laptop any more; loading the real Gemma Scope SAEs through SAELens was
  tested separately on the cached Gemma activations.
- Not tested: anything that only exists on Kaggle (the CUDA device path, the pinned install on
  their image, 4-bit loading, the T4 timings).

---

# Round 2 (after the first run came back)

The first run finished in 143 minutes with all 25 stages succeeding. Round 2 is four smaller
notebooks, one per model, so that they can run on two machines at once and a failure in one does not
cost the others. All four are thin wrappers around `scripts/gpu_round2.py`.

| Notebook | Where | Token | What it runs | Estimated GPU time |
|---|---|---|---|---|
| **A** `notebooks/kaggle_r2_qwen3b.ipynb` | Kaggle, T4 x2 | none | Qwen 3B: matched steering controls (final token and end of hypothesis), adjacency causal test at four sites, label-preserving and label-changing patching pairs | 3 – 4.5 h |
| **B** `notebooks/kaggle_r2_qwen7b.ipynb` | Kaggle, T4 x2 | none | Qwen 7B behaviour only: answer logits on the debiased pool, the official test set, CLadder and the original pool (in that order) | 1.5 – 3 h |
| **C** `notebooks/kaggle_r2_qwen1.5b.ipynb` (Colab: `colab_r2_qwen1.5b.ipynb`, the same run) | Kaggle, T4 x2 (or Colab, T4) | none | Qwen 1.5B: matched steering controls at both tokens, adjacency causal test with all its arms at four sites, label-preserving patching pairs. One run; no follow-up needed | about 1.25 h (72 min measured on Kaggle, 13 Nov) |
| **D** `notebooks/kaggle_r2_gemma.ipynb` | Kaggle, T4 x2 | `HF_TOKEN` secret | Gemma 2B: matched steering controls at both tokens, adjacency causal test, label-preserving patching pairs | 2.5 – 4 h |

**Updated Fri 23:10 (see the table under "Round 3" for the current pin of every notebook).** What changed since `cdb08d7` in
the round-2 notebooks is where the output goes, how that is checked, and a fix for models split over two GPUs; the experiments
are identical. Runs from `6d9ce8d` (output-path fix, last cell raises instead of writing `STATUS.txt`) are valid.

*What happened.* The Qwen 1.5B notebook was written for Colab and chose its output directory by testing whether `/content`
exists. Kaggle's image has a `/content` folder too. Run on Kaggle, the notebook finished all 12 stages in 72 minutes and wrote
its two archives to `/content/out`; Kaggle keeps only `/kaggle/working`, so they were lost. My note that such a run would write
to `kaggle_out/` was wrong, because I had assumed `/content` does not exist on Kaggle and had not checked.

*What every notebook does now* (`scripts/notebook_cells.py`, tested in `tests/test_notebook_cells.py`):

1. The first code cell detects the platform, Kaggle first (the `KAGGLE_KERNEL_RUN_TYPE` variable or a `/kaggle` folder), and
   writes the archives to `/kaggle/working/kaggle_out` on Kaggle and `/content/out` on Colab. Either notebook file therefore
   works on either platform.
2. The same cell stops at once, before any GPU time is spent, if that directory is not under the path the platform keeps or
   cannot be written. The runner (`scripts/gpu_round2.py`) refuses to start on Kaggle with any other output directory.
3. **The last cell lists the output directory with sizes, writes `STATUS.txt` into it and prints a banner. It never
   raises** (changed at pin `041f11e`, on the user's instruction: whether Kaggle keeps the output of a version whose last
   cell failed is not known, so nothing may depend on it). First line of `STATUS.txt`: `OK`, `FAILED` (followed by the
   list of problems: an expected archive missing, empty or misplaced, or the runner never reached its export step) or
   `SKIPPED` (the run decided not to run; the reason follows). Expected are `results_<run>.zip`, every archive the runner
   recorded, and one `cache_<run>_<model>.zip` per model whose cache stage succeeded. A copy of the status is inside the
   results archive. **Download `STATUS.txt` together with the zip files.** The ingest refuses a run from such a notebook
   unless the status says `OK` for that run; `--ignore-status` overrides this, `--allow-partial` does not.
4. The Hugging Face token is read from the platform's own Secrets (attached per notebook, see step 4 above). The Gemma
   notebooks stop in the first cell without it, and also if Hugging Face refuses that token access to the gated weights
   (one small request, before anything is installed).

*Notebooks already launched.* A (Qwen 3B) and B (Qwen 7B) are Kaggle notebooks and wrote to `/kaggle/working` in every
version, so their output is kept; their zips are at the top level of the Output tab, not in `kaggle_out`. **A was run from
`cdb08d7`, which already contains the fifth audit's corrections, so its follow-up A2 is not needed.** The rerun of C from this
commit likewise replaces both the original C and its follow-up C2.

Earlier pins (`4d6afe5`, `5c31791`, `bab7d1b`) remain usable: their raw outputs are re-analysed on the laptop with the corrected analysis, and the ingest accepts them. Compared with this version they lack some of the following: two small arms of the adjacency test; steering controls shuffled within hypothesis string on the debiased pool (fifth audit); and the runner hardening (a smoke test kept under its own run name, an explicit finished flag), which I reported as done at 18:10 and which was in fact only applied in this commit. An independent review of the round-2 code
(`notes/audit_round2_code_raw.md`) found that the first version of the adjacency test could not
answer its question (its "removal" left the information readable; one of its two directions did
not separate the classes for "X directly causes Y" hypotheses). Notebooks A, C and D now run the
corrected test (`scripts/17_adjacency_test.py`). If you launched A, C or D from an earlier version
(pinned commit `9fdec33` or `3c7b87d`, visible in the notebook's first code cell), let it finish:
its steering-controls and patching results are unaffected. Then run the matching follow-up, which
does only the corrected adjacency test:

| Follow-up | For | Where | Token | Estimated GPU time |
|---|---|---|---|---|
| `notebooks/kaggle_r2b_qwen3b_adjacency.ipynb` | A launched from a pin before `cdb08d7` (**not the case: A ran from `cdb08d7`; skip**) | Kaggle T4 x2 | none | 1.5 – 2.5 h |
| `notebooks/colab_r2b_qwen1.5b_adjacency.ipynb` | C launched from a pin before `cdb08d7` (**not needed: C is rerun from `6d9ce8d`**) | Colab T4 or Kaggle | none | about 1 h |
| `notebooks/kaggle_r2b_gemma_adjacency.ipynb` | D launched from a pin before `cdb08d7` | Kaggle T4 x2 | `HF_TOKEN` | 1.5 – 2.5 h |

Each follow-up reruns, at the current pin, the steering controls at both tokens (on the debiased pool with labels shuffled within hypothesis string) and the adjacency test with all its arms. It does not repeat the patching. Launch it after the main run of that model has finished, so that the two do not compete for quota.

Their downloads go to `kaggle_in/r2b_qwen3b/`, `kaggle_in/r2b_qwen15b/`, `kaggle_in/r2b_gemma/`
and are ingested with `--run r2b_qwen3b` and so on. Notebook B (7B behaviour) is unchanged in
substance. A smoke test (`SMOKE_TEST = True`) now writes under its own run name and is refused by
the ingest, so it can never be mistaken for a real run.

Kaggle total: about 7 – 11.5 hours of the weekly 30 (the first run used 2.4). Colab: 1.5 – 3 hours.
Qwen 0.5B runs the same jobs on the laptop. The SAE tier already ran in the first notebook.

**Where the estimates come from.** The first run's measured stage times on a T4: one steered
forward pass of the final token costs 6–8 ms for Gemma and Qwen 3B; the 3B cache of 4,800 items
takes about 15 minutes. A matched-controls run is 150 items x 287 vectors; an adjacency-test site
is up to 500 items x 259 vectors. Steering at the end of the hypothesis re-runs the five or six
template tokens that follow it, which I have assumed costs two to four times as much: that factor
is the main uncertainty, and I could only measure it on the laptop. Each adjacency-test site
measures its own speed on the first five items and shortens itself to fit 45 minutes (it keeps a
prefix that is balanced over hypothesis templates, and records how many items it ran).

**The 7B notebook is the least certain.** It first tries 4-bit loading (bitsandbytes), which has
never run for this project (the package is not available on the laptop, and the first Kaggle run
recorded it as not installed; this notebook installs it). If that stage fails within twelve
minutes, it falls back to float16 weights spread over the two T4s, which is also untested. The
cache builder stops itself every 50 minutes and resumes, up to four times; whatever is finished
is reported, and the debiased pool comes first.

## Steps

**Kaggle (A, B, C, D).** As for the first notebook: *Create → New Notebook → File → Import
Notebook*, upload the `.ipynb`, set **Accelerator = GPU T4 x2** and **Internet = On**, attach the
`HF_TOKEN` secret (Add-ons → Secrets, per notebook; required for D, optional for the others), then
*Save Version → Save & Run All (Commit)*. The archives are in the Output tab under `kaggle_out/`. If Kaggle
lets you run two GPU sessions at once, start A and B together; otherwise A first.

**Colab (C, only if you prefer it to Kaggle).** Open <https://colab.research.google.com>, *File → Upload notebook*, choose
`notebooks/colab_r2_qwen1.5b.ipynb`. *Runtime → Change runtime type → T4 GPU*. *Runtime → Run
all*. Keep the tab open until the last cell finishes: a free Colab session ends when the tab is
closed or left idle, and its files are deleted. The last cell downloads the zip files through the
browser; allow multiple downloads when asked. If the downloads are blocked, open the Files panel
(folder icon on the left) and download them from `/content/out`.

## Bringing a round-2 run back

Each run goes into its own sub-folder of `kaggle_in/`, named after the run, zipped or unzipped:

```
kaggle_in/
├── r2_qwen3b/     results_r2_qwen3b.zip,  cache_r2_qwen3b_qwen3b.zip      (about 0.8 GB)
├── r2_qwen7b/     results_r2_qwen7b.zip,  cache_r2_qwen7b_qwen7b-4bit.zip (about 1 GB)
├── r2_gemma/      results_r2_gemma.zip,   cache_r2_gemma_gemma2b.zip      (about 0.6 GB)
└── r2_qwen15b/    results_r2_qwen15b.zip, cache_r2_qwen15b_qwen1.5b.zip   (about 0.5 GB)
```

Then, one command per run (the run name is the folder name):

```bash
.venv/bin/python scripts/kaggle_ingest.py --run r2_qwen3b
```

It checks the files, the pinned commit, and that every planned stage succeeded; extracts only what
the run created or changed; keeps the laptop's activations unless the download holds more token
positions for the same items; and rebuilds tables, figures and documents. `--allow-partial` and
`--dry-run` work as before. For the 7B run it prints the behaviour table.

## What is tested and what is not (round 2)

Tested on the laptop with Qwen 0.5B: every stage of `scripts/gpu_round2.py` in smoke-test mode
through the same command the notebooks issue; the steering shortcut at the end-of-hypothesis token
against a full forward pass (`tests/test_steer.py`); the ingest on the smoke-test output. Not
tested: anything that needs CUDA (4-bit loading, two-GPU placement), the Colab secrets and
download cells, and run times. The platform choice and the final output check are tested as functions
and through the notebook's own cells with the real runner in smoke mode on the laptop
(`tests/test_notebook_cells.py`); on Kaggle itself they have not run yet. The first cell of the next
run shows the result within a minute: it must print `platform: kaggle | archives will be written to
/kaggle/working/kaggle_out`.

---

# Round 3 (Fri 13 Nov, late): depth on the three headline claims

Decision D32 in `EXPLAINER.md`. Quota is no longer the limit; two concurrent sessions and Sunday 10:00 are.
**Nothing is launched after Saturday 18:00 (Toronto).** Every notebook below is one *Save & Run All*, writes its
archives and `STATUS.txt` to `/kaggle/working/kaggle_out`, and is pinned to commit `e336431` (shown in its first
code cell).

| | Notebook | Run folder in `kaggle_in/` | HF secret | GPU hours | State (Sat 14 Nov, morning) |
|---|---|---|---|---|---|
| A | `notebooks/kaggle_r2_qwen3b.ipynb` | `r2_qwen3b` | no | 2.3 (measured) | **done, ingested, audited** |
| C | `notebooks/kaggle_r2_qwen1.5b.ipynb` | `r2_qwen15b` | no | 1.2 (measured) | **done, ingested, audited** |
| B | `notebooks/kaggle_r2_qwen7b.ipynb` | `r2_qwen7b` | no | 1.5 – 3 | launched Fri night from pin `6d9ce8d` (see the note on B below) |
| D | `notebooks/kaggle_r2_gemma.ipynb` | `r2_gemma` | **yes** | 2.5 – 4 | launched Fri night from pin `6d9ce8d` |
| **H** | `notebooks/kaggle_r3_settle_qwen3b.ipynb` | `r3s_qwen3b` | no | about 1.5 | **your decision.** Test and threshold fixed in advance; reviewed twice. It has about one chance in three to one in two of an informative outcome (see below) |
| H | `notebooks/kaggle_r3_settle_qwen1.5b.ipynb` | `r3s_qwen15b` | no | about 1 | the same for the Qwen 1.5B arm (gold probe, negative slope); the more informative of the two (a little better than even) |
| E | `notebooks/kaggle_r3_gemma9b.ipynb` | `r3_gemma9b` | **yes** (licence of `google/gemma-2-9b-it` accepted) | 2 – 3.5; ends within about 15 minutes if it cannot run | ready |
| F | `notebooks/kaggle_r3_patch240_qwen3b.ipynb` | `r3p_qwen3b` | no | about 2 | ready |
| F | `notebooks/kaggle_r3_patch240_gemma.ipynb` | `r3p_gemma` | **yes** | 1 – 2.5 | ready |
| G | `notebooks/kaggle_r3_controls100_qwen3b.ipynb` | `r3c_qwen3b` | no | about 3 | ready, **no longer recommended** (the audit: 100 controls on the same split refine one percentile and decide nothing; H contains the 100-control answer for the arm in question) |
| G | `notebooks/kaggle_r3_controls100_qwen1.5b.ipynb` | `r3c_qwen15b` | no | about 2 | ready, not recommended (same reason) |
| G | `notebooks/kaggle_r3_controls100_qwen0.5b.ipynb` | `r3c_qwen05b` | no | about 1 | ready, low priority (places the borderline end-of-hypothesis contrast of Qwen 0.5B among 100 controls) |
| G | `notebooks/kaggle_r3_controls100_gemma.ipynb` | `r3c_gemma` | **yes** | 2.5 – 4.5 | not before D is ingested (it stops early otherwise) |

**Note on B.** B was launched from the notebook pinned to `6d9ce8d`. That version first tries to load Qwen 7B in 4-bit; if that works,
the run is fine. If 4-bit loading failed and it fell back to float16 over two GPUs, the cache stage will have failed at its first
batch (the collector bug fixed in `041f11e`), the summary will show `FAILED` for `r2.qwen7b.behaviour.cache` and there will be no
activation archive. In that case rerun B from the current notebook; nothing from the failed run needs keeping. Runs from `6d9ce8d`
write no `STATUS.txt`; the ingest accepts them without it.

## H: the settle run (added Sat 14 Nov after the eighth audit; second version after the ninth)

Two steering comparisons at the end of the hypothesis ranked first among 21 (Qwen 3B: gold difference of class means, +0.74 per s;
Qwen 1.5B: gold probe weights, −0.25 per s). The eighth audit found each undecided between a chance first rank (they are 2 of 12
comparisons on that pool, 24 in all) and a modest real effect, and found that more controls on the same split would not decide it.
`scripts/07d_steering_settle.py` is the run designed to decide it. Its first version was reviewed before it ran (ninth audit,
`notes/audit_settle_code_raw.md`) and three of its comparisons would not have meant what I had written; this is the second version.

- **Primary test:** the direction refitted on the mirror half of the 25 blocks and steered on 150 items of the other half, one-sided
  in the sign of the first run, against **500** control directions fitted to labellings that have both balances of the gold label
  (half valid within every hypothesis string; one valid and one invalid item per premise).
- **Threshold 0.05/24** (the real slope beyond all 500 controls), because this arm is one of the first ranks picked out of 24
  comparisons. Refitting on the other half gives nearly the same vector (cosine 0.84 for Qwen 3B), so it is not a new draw.
- **What each outcome is worth** (second review, before any run). Beyond all 500: moderate evidence for a real effect (about 3% of
  runs with nothing label-specific, 25% for Qwen 3B and 48% for Qwen 1.5B with a real effect of the observed size). Above 0.05:
  moderate evidence against. **Between the two: evidence for neither**, and for Qwen 3B that is the most likely outcome whatever
  is true (51% and 67%). The Qwen 3B run has about one chance in three to one in two of telling us anything; the Qwen 1.5B run a
  little better than even.
- The label beyond adjacency (the label minus its mean in its string-and-adjacency cell, with the adjacency direction projected
  out) against 100 controls built the same way: "stands out" = one-sided p ≤ 0.05; a null on this question shows nothing;
  the adjacency direction itself; two neighbouring layers; smaller steps; the original split with 100 controls of each kind.
- The primary configuration runs first, and the directions are saved before the model loads, so a run that is cut short still
  leaves the primary test or at least everything needed to read what was computed.

Test and threshold are in `notes/log.md` ("Ninth independent audit ... and the AMENDED pre-registration"). What each outcome would do
to the wording of claim 3 is in `notes/claim3_proposed_rewording.md`, option C. The script refuses to start unless the 20-control
result of the model is in its checkout (it is, from pin e336431 on). **What has and has not been tested:** end to end with a stand-in
for the model (both arms, smoke configuration, resume from a partial file); every control family checked to be calibrated under a
response unrelated to any label, if that response is isotropic (exact computation on the Qwen 3B and Qwen 1.5B activations); the
second version was reviewed independently (tenth audit, `notes/audit_settle_code2_raw.md`: it can run; five further changes, applied);
**not** run on a GPU, and those last five changes have been seen by me alone.

The G notebooks for Qwen 1.5B, 3B and Gemma use the sites of the 20-control run, which are recorded in result files
that only exist in the repository after that run is ingested. I re-pin those three notebooks then. **Launched
earlier they stop in their first intervention stage** (about ten minutes in, after the install and the cache), with
the message that the 20-control result is not in the checkout.

## E: Gemma-2-9B on two T4s, and how the notebook decides whether to run

*Why it should fit.* The Hub lists 9,241,705,984 parameters in four half-precision files of 17.2 GiB (18.5 GB)
together. A Kaggle T4 reports 15.6 GB, which is 14.5 GiB (`results/kaggle_environment.json`, first run), not the
15 GiB I first assumed. The loader (`models.multi_gpu_caps_gib`) lets at most 9 GiB of weights onto the first card,
which also holds the inputs, and 13 GiB onto the second: 22 GiB of room for 17.2 GiB. The independent review computed
the placement transformers makes under these limits, without loading weights: 8.35 GiB on the first card (embedding
and 18 layers) and 8.86 GiB on the second (24 layers and the final norm). The activations of one batch (at most
3,000 tokens) are well under 2 GiB per card. The download goes to `/kaggle/temp`; the preflight asks for the
download plus 12 GiB of free disk.

*What I have not been able to test.* Anything with two GPUs. On Friday night I found by reading that the activation
collector would have failed on a model split over two cards (fixed in `041f11e`), and the review found that my
first version of the memory check would have refused real T4s every time (it assumed 15 GiB cards; fixed). The
review traced the rest of the two-GPU path through the cache code by reading and found no other place where two
devices meet. None of this has run on CUDA. That is what the checks are for:

1. **First cell (seconds, nothing installed):** the `HF_TOKEN` secret is attached; Hugging Face grants that token
   access to `google/gemma-2-9b-it` (one small request); two GPUs are present. Otherwise the cell stops with the
   instruction (attach the secret / accept the licence / set Accelerator to GPU T4 x2).
2. **Runner, stage `preflight.resources` (about a minute after the install):** the real size of the weights from the
   Hub, the real memory of the two GPUs, the per-GPU limits, free disk for the download.
3. **Runner, stage `preflight.load` (download plus about five minutes):** the model is loaded exactly as the real
   stages load it, and 16 items of the debiased pool go through the real cache code into a scratch cache. Reported:
   where the weights were placed, peak memory per GPU, seconds per item, the first logit differences.
4. If 2 or 3 fails, the model is **skipped**: no further stage runs, the summary prints the reason between lines of
   exclamation marks, `STATUS.txt` says `SKIPPED` with the reason, and the session ends normally. Send me the reason.
   (`SKIP_STAGES` cannot bypass the preflight; to run without it, remove `preflight` from `JOBS`.)
5. Not covered by 16 items: whether every one of the 9,000 items stays finite in half precision. If one does not, the
   cache stage stops with "non-finite activations" at the same place on every retry: loud, and the items cached up
   to there are reported.

The run itself is the same as B's: answer logits on the debiased pool, the official test set, CLadder and the original
pool (in that order, resumable in 50-minute parts), then the export of the debiased-pool activations. The probe
comparison runs on the laptop from the download.

## F: patching on all available pairs

`--n-pairs 240` asks for 240 pairs per kind; the pool of premises with at most five variables holds about 210
label-changing and 230 label-preserving pairs, so that is what runs (the result files record the number). Activation
patching only: no component patching and no random-donor control, both of which exist in the 60-pair runs. Results:
`patching_<model>_n240.json` and `patching_<model>_same_n240.json`; the 60-pair files are not touched. The 60 pairs
are a subset of the larger set, but the mix differs (label-preserving: 31 of 60 are Yes/Yes pairs, 89 of 231 in the
larger set), so the two files are compared within hypothesis template and by common label, not pooled. There is no
checkpoint inside the patching script, so each of the two stages has a limit of 175 minutes; I will narrow the time
estimate when the 60-pair stage times of runs A and D are in.

## G: 100 control directions

`--n-shuf 100 --sites best --control-tags c2c_debiased`: the steering controls on the debiased pool at both tokens
and the adjacency test at the two best-layer sites, with 100 shuffled-label directions instead of 20. Results:
`steering_controls_<model>_c2c_debiased[_hyp]_c100.json`, `adjacency_<model>_c2c_debiased_{final,hyp}_best_c100.json`.
An adjacency item costs 1,126 forward passes instead of 259; each site has 110 minutes and shortens itself to a
prefix of items balanced over hypothesis templates if it would not fit (the result file records how many it ran).
It plans for 85% of its time and looks at the clock again every 50 items. **A 100-control run refuses to start unless
the 20-control result of the same model and site is in its checkout, and it must select the same layer**: that is why
the notebooks for Qwen 1.5B, 3B and Gemma can only run after I have ingested the 20-control run and re-pinned them.
The Qwen 0.5B results are already in the repository, so its notebook can run now.

## What is tested and what is not (round 3)

Tested on the laptop, CPU only, Qwen 0.5B, a few items per stage through the runner: the preflight on a model that
runs; the preflight on Gemma-2-9B (it must skip itself: no token, no CUDA) and the `SKIPPED` status that follows; the
100-control steering stage at the final token. The swap watchdog stopped the rest of that batch (two models were in
memory at once), and a model load on its own was stopped again, so the remaining pieces were tested without a model:
the adjacency script end to end with 100 and with 20 shuffles on a stand-in whose answer is a linear function of the
added vector (identical real-direction numbers, as they must be), its refusal to start without the 20-control result,
and its mid-run time check with items that turn out slower than the first estimate. **Not run on the laptop:** the
end-of-hypothesis steering stage with 100 controls (same script as the final-token stage) and the patching options
(two more arguments to a script that is otherwise unchanged). Independent review: `notes/audit_round3_code_raw.md`.
Unit tests
(`tests/test_notebook_cells.py`): `STATUS.txt` in every state, the ingest's refusal, the access check against a fake
server, the GPU count. One live request without a token confirmed that Hugging Face answers a public repository with
a redirect and the gated Gemma repositories with 401. **Not tested:** two GPUs, the gated download with a real token,
Kaggle's disk, and every run time above.


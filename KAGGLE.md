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
| export | both | Zips (below) | `/kaggle/working/*.zip` |

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
5. **Optional 15-minute dry run.** In the first code cell set `SMOKE_TEST = True` and run the
   cells interactively (Run All). It pushes a few dozen items through every stage. If the summary
   shows every stage `ok`, set it back to `False`.
6. **Real run.** Save Version → *Save & Run All (Commit)* → Advanced settings → *Always save
   output*. You can close the browser; the run continues on Kaggle.
7. **Wait.** Estimated 3 to 4 hours on a T4 (see the breakdown below). Kaggle's limit is 12 hours
   per run and 30 GPU-hours per week.
8. **Download.** Open the finished version → Output. Download:
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
| **C** `notebooks/colab_r2_qwen1.5b.ipynb` | Colab, T4 | none | Qwen 1.5B: end-of-hypothesis steering controls, adjacency causal test, label-preserving patching pairs | 1.5 – 3 h |
| **D** `notebooks/kaggle_r2_gemma.ipynb` | Kaggle, T4 x2 | `HF_TOKEN` secret | Gemma 2B: matched steering controls at both tokens, adjacency causal test, label-preserving patching pairs | 2.5 – 4 h |

**Updated Fri 20:30 (commit `bab7d1b`). Earlier pins `4d6afe5` and `5c31791` remain valid: their raw outputs are re-analysed on the laptop with the corrected analysis. They lack two small arms of the adjacency test, and their debiased-pool steering controls are shuffled over the whole pool instead of within hypothesis string (fifth audit).** An independent review of the round-2 code
(`notes/audit_round2_code_raw.md`) found that the first version of the adjacency test could not
answer its question (its "removal" left the information readable; one of its two directions did
not separate the classes for "X directly causes Y" hypotheses). Notebooks A, C and D now run the
corrected test (`scripts/17_adjacency_test.py`). If you launched A, C or D from an earlier version
(pinned commit `9fdec33` or `3c7b87d`, visible in the notebook's first code cell), let it finish:
its steering-controls and patching results are unaffected. Then run the matching follow-up, which
does only the corrected adjacency test:

| Follow-up | For | Where | Token | Estimated GPU time |
|---|---|---|---|---|
| `notebooks/kaggle_r2b_qwen3b_adjacency.ipynb` | A launched early | Kaggle T4 x2 | none | 1 – 2 h |
| `notebooks/colab_r2b_qwen1.5b_adjacency.ipynb` | C launched early | Colab T4 | none | 1 – 1.5 h |
| `notebooks/kaggle_r2b_gemma_adjacency.ipynb` | D launched early | Kaggle T4 x2 | `HF_TOKEN` | 1 – 2 h |

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

**Kaggle (A, B, D).** As for the first notebook: *Create → New Notebook → File → Import
Notebook*, upload the `.ipynb`, set **Accelerator = GPU T4 x2** and **Internet = On**, attach the
`HF_TOKEN` secret for notebook D only, then *Save Version → Save & Run All (Commit)*. If Kaggle
lets you run two GPU sessions at once, start A and B together; otherwise A first.

**Colab (C).** Open <https://colab.research.google.com>, *File → Upload notebook*, choose
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
download cells, and run times.

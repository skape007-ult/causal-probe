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
9. **Unpack into the repository** (from the repository root; the zips contain `results/…` and
   `cache/…` paths):

   ```bash
   unzip -o ~/Downloads/results.zip -d .
   ```

   ```bash
   for z in ~/Downloads/cache_*.zip; do unzip -o "$z" -d .; done
   ```

   `results.zip` overwrites nothing that exists locally except files the Kaggle run recomputed
   (it only adds Gemma steering/patching/SAE files, the larger model's files and `kaggle_*`).
10. **Tell me (or run yourself)** the laptop analysis for the new model, CPU only:

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

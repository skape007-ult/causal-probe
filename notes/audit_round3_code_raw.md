# Audit, round 3 code (seventh audit): raw report

Scope: `git diff d52735a -- scripts src tests` (working tree, Fri 13 Nov about 23:30) plus commit `041f11e`.
Reviewer had no CUDA. "ran" = executed on the laptop CPU with scratch directories; "read" = from the source;
"reasoned" = inference that a run would have to confirm. Line numbers are those of the working tree at that time.
Scratch files: `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit7/`.
Nothing in the repository was changed except this file. (`notes/progress.md` changed while I worked; not by me.)

## Findings

### F1. critical (blocks run E every time): the preflight rejects real T4s
`scripts/preflight_model.py:32-44` (`fits`), `:82-86`; `src/causal_probe/models.py:92-95` (`multi_gpu_caps_gib` = [9, 14]).

`fits` requires every cap to be at most `GPU total - 0.8 GiB`. The cap on the second card is 14 GiB. The docstring and
KAGGLE.md assume a T4 has 15 GiB. The repository's own record of the first Kaggle run says otherwise:
`results/kaggle_environment.json` has `gpu_memory_gb: [15.6, 15.6]`, which is `total_memory / 1e9`, i.e. 14.48 to
14.58 GiB. So `14 > 14.5 - 0.8` and the check returns False.

- Outcome: `preflight.resources` exits 3 with "gemma9b does not fit: GPU 1 has 14.5 GiB; the plan puts up to 14 GiB of
  weights on it and needs 0.8 GiB spare", the runner skips the model, STATUS.txt says SKIPPED. The session costs
  minutes, but run E can never start from this code.
- The real load would work: I computed the device map transformers 5.22.2 builds for a Gemma-2-9B-shaped model with
  `max_memory={0: "9GiB", 1: "14GiB"}` (meta tensors, no weights): 8.35 GiB on GPU 0 (embedding, tied `lm_head`,
  18 layers), 8.86 GiB on GPU 1 (24 layers, final norm, `rotary_emb`), 17.21 GiB in total.
- How established: ran `fits(17.214, [g, g], [9, 14])` for g = 14.48, 14.53, 14.58, 14.76 GiB: all False; 15.0: True.
  The GPU size is from the committed environment record, not measured by me.
- Smallest fix: `multi_gpu_caps_gib` returns `[9, 13]` (ran: `fits` passes at 14.5 GiB; 22 GiB of caps for 17.2;
  the placement needs only 8.86 GiB on the second card). Add a unit test of `fits` with 14.5 GiB cards. Correct
  "A T4 has 15 GiB" in the docstring and KAGGLE.md.
- Note: `SKIP_STAGES = "r2.gemma9b.preflight"` does not bypass the preflight (a skipped stage returns False and the
  model is skipped). The bypass is to remove `preflight` from `JOBS`.

### F2. major: on Colab the archives are downloaded before STATUS.txt exists, so the ingest refuses every Colab run
`scripts/make_round2_notebooks.py:275-276` (cell order: SUMMARY, COLAB_DOWNLOAD, FINAL_CHECK);
`scripts/notebook_cells.py:119-132`, `:175-187`.

The download cell hands `results_<run>.zip` to the browser, then the final cell writes STATUS.txt and inserts the
status copy into the zip that stays in `/content/out`. The downloaded zip has no copy, and STATUS.txt is never
offered for download.

- Outcome: a complete Colab run from pin `041f11e` or later arrives without any status. `kaggle_ingest.py --run` exits
  with "STATUS.txt is missing ... the last cell did not run or the output is incomplete". The run is good; the message
  invites a rerun or a routine `--ignore-status`.
- How established: read. The end-to-end test executes the cells in this order on platform "local", where the download
  cell does nothing, so it cannot see this.
- Smallest fix: in FINAL_CHECK, after `write_status`, on Colab call `files.download` for STATUS.txt and again for the
  results zip, inside its own try/except; or drop the separate download cell and download from the final cell.

### F3. major: an adjacency site that shortens itself uses its whole time limit, on an estimate from six items
`scripts/17_adjacency_test.py:756-767`; `scripts/gpu_round2.py:234-240` (`--max-minutes 110`, stage timeout 122).

With 100 controls an item costs 1,126 vectors instead of 246, and the limit only doubles, so shortening becomes the
normal case for the larger models (KAGGLE.md expects it). `keep = n_ + int(budget / per_item)` fills the budget to
100 %. The only margin is the 12 minutes between the script's limit and the subprocess timeout (11 %). There is no
second look at the clock.

- Evidence that six items are not enough: `results/round2_logs_r2_qwen05b/r2.qwen0.5b.adjacency.final.best.log`:
  estimate "0.78 s/item" at 13 s; the 500 items finished at 529 s, which is 1.04 s/item, 34 % above the estimate
  (laptop, MPS; the second log of the same site is about 5 % above). Prompt lengths in the pool range from about 250
  to 1,650 tokens.
- Outcome if the estimate is 12 % or more too low on a shortened run: killed at 122 minutes. No result JSON. The
  checkpoint (`..._partial.npz`, exported) holds `idx`, `out`, `norms` only, not the fitted directions, class-mean
  differences or erasers; `--analyse-only` cannot read it; and these runs use `--no-resid`, so the Kaggle activations
  needed to refit are not exported. Up to two GPU hours per site.
- How established: read; the log is from a laptop run. Decided by: per-item time variance on a T4.
- Smallest fix: shorten to 85 % of the remaining budget, and at every checkpoint (each 50 items) re-project; if the
  end would pass `--max-minutes`, cut `idx` to the items done (a multiple of five keeps the template balance) and
  leave the loop.

### F4. major: a stale STATUS.txt beside the archives overrides the copy inside the results archive
`scripts/kaggle_ingest.py:204-209`.

The loose file is read first; the copy in the zip is used only if the loose file is absent. The two are never compared.

- Outcome (ran, dry run): folder with `STATUS.txt` = OK (left from an earlier download of the same run name) and a
  results zip whose inner copy says FAILED -> "ACCEPTED (exit 0)".
- Also not compared: the `commit:` line of STATUS.txt against the environment record (ran: OK with another commit is
  accepted).
- Smallest fix: when both exist, require both to say OK and to have the same `written:` line; otherwise refuse. Compare
  `commit:` with `git_commit`.

### F5. major: without a usable environment record the STATUS check switches itself off, and `--allow-partial` then ingests
`scripts/kaggle_ingest.py:201-202`, `:212-218`.

`writes_status` is False when `git_commit` is missing, not 40 hex digits, or not a commit of the local repository.
The run is then treated as "predates STATUS.txt".

- Outcome (ran, results and cache redirected to scratch): results zip without `round2_environment_<run>.json` and
  without any status, `--allow-partial --no-fold` -> "accepted without it", files extracted, exit 0. KAGGLE.md and the
  help text say `--allow-partial` does not override the status check.
- Smallest fix: if the run's commit is unknown, decide with the pinned commit (`commit`), and if that is unknown too,
  set `writes_status = True`.

### F6. major (older code, but it is the path runs E and B use): a smoke test followed by the real run in the same session gives 64-item results labelled as the real run
`scripts/gpu_round2.py:86-88` (tiny redirects results, markers and run name, not the activation cache), `:184-186`
(`--limit 64` in the behaviour job); `src/causal_probe/cache.py:122-124` ("already complete").

- State: interactive session, `SMOKE_TEST = True`, run; then `SMOKE_TEST = False`, run again. Both use
  `CAUSAL_PROBE_CACHE = TEMP + "/cache"`.
- Outcome: the smoke test leaves `cache/<model>/<tag>/meta.json` with 64 items for all four item sets. The real run's
  `behaviour.cache_part0` finds them complete and succeeds in seconds. Status `tiny: false`, `all_ok: true`,
  STATUS.txt OK. The ingest accepts it (step 6 compares the first `meta["n"]` items only). Behaviour numbers for the
  model are then on 64 items per set; only the `n` column shows it.
- Not affected: "Save & Run All" (fresh session), and the `cache` job (the smoke test builds those caches in full).
- How established: read.
- Smallest fix: in tiny mode also set `CAUSAL_PROBE_CACHE` to `<cache>/_tiny` before `config` is imported (same three
  lines as for results). Optionally let the ingest refuse a cache whose `n` is smaller than the item set.

### F7. major: every round-2 and round-3 notebook in the working tree is pinned to the placeholder "TESTSHA"
`notebooks/*_r2*.ipynb`, `notebooks/*_r3*.ipynb`, line 30 (15 files; `kaggle_gpu.ipynb` is not affected).

- Outcome if uploaded or committed as they are: `git checkout TESTSHA` fails in the clone cell (seconds, no GPU time).
- Outcome on the laptop now: `kaggle_ingest.pinned_commit` (`:56`, needs 40 hex digits) returns None for every run,
  including A, B and C that are in flight. The commit check falls back to "any ancestor of HEAD since 3c7b87d", and a
  mismatch between the jobs run and the notebook's JOBS becomes a note instead of a problem (`:248-251`).
- The notebooks are otherwise identical to what the current generator produces (ran: compared all 15 in memory).
- `tests/test_notebook_cells.py::test_every_notebook_ends_with_the_output_check` passes with this pin.
- Smallest fix: regenerate after the code commit is pushed; add to that test: COMMIT matches `[0-9a-f]{40}` and
  `git cat-file -e` succeeds.

### F8. major: "same sites as the 20-control run" is stated, not enforced
`scripts/07c_steering_controls.py:59-65`; `src/causal_probe/steer.py:136-149`; `scripts/17_adjacency_test.py:635-641`;
notebook text in `scripts/make_round2_notebooks.py:89-116`.

A `_c100` run takes its site from whatever selection file is in the clone and computes one if there is none.

- In the repository now: final-token site of 07c is fixed for all four models (`steering_<model>_c2c_debiased.json` is
  tracked). Tracked for Qwen 0.5B only: `steer_select_*_c2c_debiased_content_end.json` and
  `feature_select_*_c2c_debiased_adj_{final,content_end}.json`. For Qwen 1.5B, 3B and Gemma-2-2B a `_c100` run from
  a pin made before the 20-control ingest is committed selects its own layer for the end-of-hypothesis steering site
  and both adjacency sites, on a rebuilt cache. If two layers are close, the layer can differ.
- Outcome: `..._c100.json` at another layer than the 20-control file, both named as the same site. Each JSON records
  `layer`, so it can be seen, but nothing compares them. The c100 run would also write the selection file under the
  shared name; ingested first, it becomes "the" selection file.
- KAGGLE.md says this in prose ("I re-pin those three notebooks then"). The code and the notebook titles do not.
- How established: read; `git ls-files results`.
- Smallest fix: when `--out-suffix` / `--suffix` is given, require the selection file to exist (exit before the GPU
  pass otherwise), and if the 20-control result JSON exists, assert its `layer` equals the chosen one.

### F9. major for the fold-in after run E: the tables script has no label for gemma9b
`scripts/12_tables.py:16-17` (`LAB`), `:437`; `scripts/06_figures.py:19-22` (`MODEL_COL`, `MODEL_LAB`).

- Outcome (ran, with results and tables redirected to scratch and a `gemma9b` entry in `answer_bias.json`):
  `KeyError: 'gemma9b'` in `t_answer_bias`. In the ingest, step 7 regenerates `answer_bias.json` with the new model and
  then `12_tables.py` fails; the tables built after that point (`probe_content`, `patching_label_preserving`,
  `adjacency_*`, SAE) keep their old content. The ingest prints FAILED for the script.
- Smallest fix: add `"gemma9b": "Gemma-2-9B"` to `LAB` and `MODEL_LAB`, and a colour to `MODEL_COL`.

### F10. minor: the cell before the last one can raise, and it ignores the runner's exit code
`scripts/make_round2_notebooks.py:209-228` (SUMMARY).

- `json.load(open(st))` raises on a truncated status file (`gpu_round2.py` writes it with `write_text`, not
  atomically). Under "Run All" the final cell then never runs: no STATUS.txt. The "never raises" property holds for
  the last cell only.
- If the runner exits at start-up (unknown option in `EXTRA_ARGS`, "refusing to start") in a session that already
  holds a finished status and archives from an earlier invocation, the final cell reads the old status and writes OK.
- How established: read.
- Smallest fix: wrap the SUMMARY body in try/except with `s = {}` and `not_finished` set to the error; add
  `if p.returncode != 0: not_finished = "runner exit code %d"`.

### F11. minor: preflight, four smaller defects
- `scripts/preflight_model.py:64-76`: any exception from `model_info` or the `config.json` download (a timeout, a 5xx)
  is reported as "no access" and skips the model. The first cell treats the same situation as a warning. Fix: retry
  three times; skip only on 401/403/404; otherwise continue with `PARAMS_B`.
- `scripts/preflight_model.py:126`, `cache.py:122`: if the scratch cache already exists and the load marker does not
  (smoke test then real run in one session; or a load check that died after the cache was written), the check prints
  "already complete" and reports OK without one forward pass. Ran: "about 0 min projected", peak memory = memory
  after load. Fix: `shutil.rmtree` the scratch cache directory at the start of `load`.
- `scripts/gpu_round2.py:171-175`: if a preflight process is killed before it writes, an older
  `preflight_<model>.json` with `ok: false` supplies an old reason. If the file is half-written, `json.loads` raises
  and the runner stops without its export (status FAILED, not SKIPPED). Fix: read it once inside try/except.
- The disk check (`:94-100`) asks for the download plus 3 GiB. The four activation caches add 8.3 GiB in
  `/kaggle/temp` (9,000 items x 43 x 3 x 3584 x 2 bytes), plus chunk files during a build.

### F12. minor: the first-cell GPU check now stops CPU sessions of every notebook, with the wrong advice
`scripts/notebook_cells.py:112-115`. `MIN_GPUS` defaults to 1, so a Kaggle or Colab session without a GPU raises
"needs 1 GPU(s) ... set Accelerator = GPU T4 x2 (not P100, not a single T4)". Before tonight this was a printed
warning; a smoke test on a CPU session is no longer possible, and the advice is wrong for one-GPU runs. Fix: enforce
only when `MIN_GPUS > 1`, or word the message by `MIN_GPUS`.

### F13. minor: ingest, smaller points
- A UTF-8 BOM makes an OK status unreadable: "STATUS.txt says ﻿OK, not OK" (ran). Safe direction. Fix:
  `st_txt.lstrip("﻿")`. CRLF is handled (ran).
- Round 1 (`kaggle_ingest.py` without `--run`) has no status check, although `kaggle_gpu.ipynb` now writes STATUS.txt.
- The rule "the run's commit has STATUS.txt in `scripts/notebook_cells.py`": right for `6d9ce8d` (file present, no
  STATUS.txt; the notebooks pinned to it at `ae0757f` do not write one) and for `041f11e` (both yes; checked with
  `git show`), and right for any later pin. Wrong only when generator and pin are on different sides of `041f11e`:
  an old pin regenerated with today's generator (a missing status is then accepted), or an old notebook file with a
  hand-edited newer COMMIT (refused). `kaggle_gpu.ipynb` at `d52735a` is the first kind (pin `406d994`, new last cell).

### F14. minor: what the tests do not test
`tests/test_notebook_cells.py` (7 pass, 0.4 s).
- `fits` and `multi_gpu_caps_gib` have no test (F1 would have been caught by one line).
- The access-check test replaces `build_opener`, so it passes if `NoRedirect` is deleted. (I tested the real opener
  against two local servers: the redirect is not followed and the second host receives nothing.)
- The ingest tests replace `git_blob`, so the real rule is not exercised; the case "loose OK, inner copy FAILED" (F4)
  and "no environment record" (F5) are absent.
- The end-to-end test runs on platform "local": no Colab order (F2), no smoke-test names, `PERSIST` is None.
- The one-GPU test edits the cell text with `str.replace`; it tests the edited cell.
- No test that the pinned COMMIT is a commit (F7).

### F15. minor: the 240-pair label-preserving set has another label mix than the 60-pair set
`src/causal_probe/patch.py:90-97`. Ran `build_pairs` for both sizes with the Qwen and Gemma tokenizers: 60-pair run
31 Yes/Yes of 60; 240-pair run 89 Yes/Yes of 231. Pooled recovery over all pairs is therefore not like for like
between the two files; `by_common_label` is. The counts in the docs are right (210 label-changing, 231
label-preserving, all token-aligned for both tokenizers), and the 60-pair sets are subsets of the 240-pair sets.

## Checked and found correct

1. Two-GPU cache path, by reading transformers 5.22.2 and accelerate 1.15.0 in `.venv`: `body_forward` sends ids and
   mask to the first parameter's device (GPU 0, embedding); layer hooks move hidden state, mask and rotary tuple per
   layer; `ResidCollector._take` now indexes on the activation's device; the final norm and the last hidden state are
   on GPU 1; `_W_ans` is created on that device; `lm_head` is tied, placed on GPU 0 with the embedding, and its hook
   moves the input; `save_readout` and `unembed_rows` go through the CPU; `_softcap` is applied in both logit paths.
   No other place where two devices meet in `01_build_cache.py --no-gen`.
2. Memory arithmetic for the real load: 9,241,705,984 parameters (Hub, read anonymously: BF16, four shards, 17.21 GiB)
   -> 8.35 + 8.86 GiB under caps 9 + 14. Batch of 3,000 tokens with eager attention and the key-value cache: well
   under 2 GiB per card by my estimate.
3. Scratch-cache isolation of the preflight (ran through the runner on Qwen 0.5B): `config.CACHE` is read at import;
   the 16 items went to `<cache>/_preflight/` only; results got `preflight_<model>.json` only.
4. Skip path (ran through the runner for gemma9b without token): `skipped` set, `models_run` empty, export step runs
   with no model, `finished` true, results zip written. A timeout is caught by `stage` and ends as a skip.
5. Stage names and markers: `_c100` and `_n240` are in the stage names; markers are per run name; no collision.
   Tiny mode with the new options builds valid commands.
6. `--n-pairs 240` does what the docs say (see F15 for the counts). `08_patching.py` needs no activation cache, so the
   jobs list without `cache` is fine. `--no-resid` exports metadata, logits and `steering*.npz` / `patching*.npz`.
7. 07c with 100 shuffles (ran, 4 items, CPU): 607 vectors, families 100/100/100, files `steering_controls_c100.npz`
   and `..._c100.json`, nothing hard-coded to 20. The first 20 shuffled-label draws equal those of a 20-shuffle run
   (ran a copy of lines 74-102 for three models): true.
8. 17 with 100 shuffles: `fit` with 20 and 100 gives identical first 20 shuffled probes and class-mean differences
   (ran): true for 17 as well. `item_vectors` gives K = 1,126; the self-check indices are general in `n_shuf` (read).
   `analyse` and `audit5_checks` ran on a raw file widened to 100 controls: every control count is 100, 11 s.
   Re-analysis of the committed 20-control raw file reproduces the committed JSON.
9. A `_c100` or `_n240` output cannot overwrite a 20-control or 60-pair file (raw npz, JSON, checkpoint), and
   `12_tables.py` / `06_figures.py` read exact names only, so they neither show nor confuse them.
10. Probe fitting is not a time risk: 100 shuffled probes take about 10 s for Qwen 3B and Gemma-2-2B on the laptop.
11. Time limits against timeouts: behaviour and cache 50 / 58; controls 64 / 74; controls_hyp 106 / 114 (projection
    check only, no checkpoint, as before); patching 175 with no own limit; adjacency see F3.
12. First cell: the token is sent to huggingface.co only (ran against local servers and live: public repo True, gated
    repo without token False with the reason); it is not printed; an unreachable Hub is a warning.
13. Status copy in the results zip (ran three rewrites on `kaggle_out/results_r2_qwen05b.zip`): `testzip` clean, 197
    original members unchanged, one copy, no leftover. The rewrite goes through a temporary file and `os.replace`.
14. Final cell: OK needs every archive the runner recorded plus the results zip; a runner that never wrote a status
    gives FAILED; smoke-test names and paths match the runner; a second run of the cell replaces the copy.
15. Ingest: unzipped results folders work for the inner status copy (ran); another run's status is refused;
    `--allow-partial` does not override a present non-OK status; `--dry-run` refuses before anything is written.
16. `tests/test_cache.py` passes on CPU with the `ResidCollector` change.

## Could not check

- Anything on CUDA: the real placement by `device_map="auto"`, a forward pass over two cards, peak memory, speed.
- The real `total_memory` of today's Kaggle T4s (F1 rests on the committed record of the first run).
- Whether Gemma-2-9B stays finite in float16 on all 9,000 items. If not, `build_cache` asserts and the stage fails
  at the same chunk on every retry (loud, not silent); the preflight's 16 items would not necessarily show it.
- Kaggle's scratch disk quota for `/kaggle/temp` (weights 17.2 GiB, caches 8.3 GiB, the environment).
- Whether Kaggle keeps output when a cell raises; Colab's `files.download` behaviour.
- The steering and patching helpers on a two-GPU model (not used by run E; not traced beyond a first reading).
- 17 end to end with 100 controls through the model (parts verified separately, see item 8).
- That the commit to be pinned is pushed to GitHub when the notebooks are launched.

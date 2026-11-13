# Round-2 code audit (independent reviewer): adjacency causal test, steering at an earlier token, label-preserving pairs, CLadder sample fix, GPU runner and ingest

Written Fri 2026-11-13, 17:30 to 18:10, by a fresh reviewer given the code and raw outputs, not the
author's conclusions. No language model was loaded; everything below was checked by reading the code,
by CPU-only scripts on the cached activations and raw `.npz` files, or by running the runner and the
ingest on empty scratch directories. Code reviewed is the working tree at commit `acbdfbb`; for
`scripts/` and `src/` it is identical to `9fdec33`, the commit pinned in all four round-2 notebooks
(only `scripts/06_figures.py` differs). During the review the working tree gained uncommitted edits
to `scripts/12_tables.py`, `scripts/06_figures.py` and `scripts/run_all.sh`; two of them are touched
on below (F2, F6).

At 18:00 an uncommitted edit to `scripts/kaggle_ingest.py` also appeared: it exits on
`status["tiny"]`, compares the run's `jobs` with the notebook's `JOBS`, and reads a new
`status["finished"]` key instead of `export`. That covers the ingest half of F1 and F8. Two things
were still open when I looked: `scripts/gpu_round2.py` was unchanged, so it neither writes
`finished` (the edited ingest would call every run cut short, including any run made from the
notebooks as pinned to `9fdec33`) nor stops the marker reuse that resets `tiny` to `False` (F1's
main path, which the new `tiny` check cannot see). Line numbers below refer to the committed
versions.

Check scripts and their outputs are in
`/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit4/`
(`c1_layout.py` … `c12_inlp_models.py`). Nothing in the repository was edited except this file.

Raw outputs used: the four smoke files (20 items, 3 shuffles) and the laptop's full-size run for
Qwen 0.5B, sites `adj_final_best` (layer 15) and `adj_hyp_best` (layer 13), 500 items and 20
shuffles each. Numbers in the findings are from `adj_final_best` unless they say otherwise; the
addendum has `adj_hyp_best`. The two `first90` sites had not finished and were not checked.

## What is correct (checked, not assumed)

- **Column layout, `build_vectors` vs `analyse`.** Layout is `[0 | +2SD × n | −2SD × n | ablate × n |
  shift: feat_probe, feat_dom, then the shuffled directions in name order (probe, dom interleaved) |
  rank-k ablation: real, then one per shuffle]`, K = 1 + 3n + 2 + 2·n_shuf + 1 + n_shuf (259 for the
  full run, 52 for the smoke run). Two tests: (1) `build_vectors` on one-hot directions, every row
  identified by its support and coefficient, both classes; (2) `analyse` on a synthetic array in
  which every column's value encodes its block and index. Every slice in `analyse`, including the
  family sub-selection inside the shift block and the real/control split of the subspace block,
  picks the intended columns (`c1_layout.py`).
- **JSON vs raw.** Pooled and per-template slopes, absolute and signed percentiles, Pearson r and
  its percentile, sign agreements, observational gaps, ablation changes in AUROC with control
  min/max, subspace ablation, pooled shift effects: all recomputed independently from the raw
  arrays. Maximum absolute difference from the JSON is 0 for the four smoke files and for the
  full-size `adj_final_best` and `adj_hyp_best` (`c2_recompute.py`). Same for `07c` at both tokens (difference
  3e-8).
- **Selection/report separation.** Layer and C, directions, mean, own SDs, class gaps, INLP bases:
  selection half only (blocks 0,3,4,7,8,10,12,15,17,18,19,20). Evaluation items: all in the report
  half, unique, 100 per template. One exception that does not matter: the median used for the
  model-answer label (line 341) is taken over all 2400 items.
- **Raw array integrity.** No all-zero rows, no exact zeros, no non-finite values, `idx` and `out`
  the same length in every file. By reading: a shortened or resumed run cannot leave unfilled rows
  in the final file, and `idx` is truncated together with `out` in both the checkpoint and the
  final save.
- **`quick_select` moved into the library.** The old inline version (from `1893e9e`) and
  `probes.quick_select` give identical grids and the same (layer, C) on Qwen 0.5B, debiased pool and
  CLadder, final token, accuracy metric (`c6_quickselect.py`, stride 8).
- **`build_pairs`, label-changing mode.** Old code (`1893e9e`) and new code return byte-identical
  JSON for both kinds at n = 12, 120, 200. The author's claim holds.
- **`build_pairs`, label-preserving mode.** 120 pairs per kind, 60 Yes/Yes and 60 No/No, at most two
  per premise, no duplicates. "hyp": same premise, exactly one character differs, both are dataset
  items with the same label. "prem": same hypothesis, every standalone occurrence of the two
  letters swapped and nothing else, corrupt label equals the dataset label of (premise, swapped
  hypothesis), and the 15 structural features of (swapped premise, hypothesis) equal those of
  (premise, swapped hypothesis) in every pair, so the relabelling argument holds. With the Qwen
  tokenizer all 120 pairs are token-aligned; one differing token for "hyp", 8 to 78 for "prem"
  (`c4_pairs.py`, `c4b_pairs_tok.py`).
- **Float16 addition in the hook.** Simulated with IEEE half precision on the cached activations:
  a 2-SD step along the adjacency probe direction is applied with 0.998 of its intended size along
  the direction (cosine 0.99) on Qwen 0.5B, and 1.000 (cosine 0.999) on Qwen 1.5B, Qwen 3B and
  Gemma. The vectors are not lost to rounding. The problem at this scale is elsewhere (F3).
- **Pinned CLadder sample.** `cladder_sample()` returns the pinned list, which equals what
  `stratified_sample` draws on this machine. Both draws contain the same 47 stories, so the
  selection/report split agrees on every shared item, and `07b` asserts that the kept items are
  in the report half.

## Findings

### Critical

**F1. A smoke-test run can be reported and ingested as a full run.**
`scripts/gpu_round2.py:72-75`, `scripts/make_round2_notebooks.py:147`, `scripts/kaggle_ingest.py`
(no reference to `tiny` anywhere in the round-2 branch).
The notebook's `SMOKE_TEST = True` adds `--tiny` but keeps the same `RUN_NAME`. Markers
(`_round2_markers_<run>`), the status file, the output file names and the `feature_select_*` /
`steer_select_*` files do not depend on `--tiny`. So after a smoke test in a session, setting
`SMOKE_TEST = False` and re-running the cell (the notebook says re-running resumes) skips every
stage on its marker, and line 75 overwrites `tiny` with `False`. Reproduced on scratch directories:
second invocation ran nothing, wrote `"tiny": false, "all_ok": true`, and
`kaggle_ingest.py --run … --dry-run` printed every stage as ok and "run reached the export step:
yes". The committed ingest never reads `status["tiny"]`, so even an honest tiny run with
`"tiny": true` passes.
Consequence: 20-item, 3-shuffle, stride-6 results written into `results/` and `cache/` under the
names of the real experiment. Kaggle "Save & Run All" starts from an empty disk, so the exposure is
interactive sessions, which is how the Colab notebook is used.
Fix: put `tiny` in the marker directory name and the output names; refuse to start if the stored
status has a different `tiny`; make the ingest refuse `tiny: true` and check `n_items` in each
result against the planned size.

**F2. A tiny smoke output sat in the real cache under a real result's name, and `--analyse-only` would have accepted it.**
`cache/qwen0.5b/c2c_debiased/steering_feature_adj_hyp_best.npz`, written 16:46: 20 items, 13
directions, layer 13. It was there before the laptop run started (17:28) and stayed until 18:02,
when that run's `content_end/best` stage finished and overwrote it with the 500-item file. So the
file is gone now; the mechanism is not. `16_feature_causal.py:334-336`
(`--analyse-only`) takes `layer_info` from the current selection file and everything else from the
npz, with no check that `z["layer"]` equals the layer just chosen, that the number of directions is
the expected one, or that the file is newer than the selection. The uncommitted loop in
`scripts/run_all.sh` runs `--analyse-only` on every `steering_feature_*.npz` it finds. If the stage
had failed, the result would have been a `feature_causal_…_adj_hyp_best.json` with n = 20 and a
`selection_auroc_at_layer` belonging to a different selection.
Fix: assert `int(z["layer"]) == layer` and `len(z["idx"])` against a minimum in `analyse`; delete
stale files before a stage starts; store `per_template`, `n_shuf` and the stride in the npz.

### Major

**F3. The ablation arm does not remove the adjacency information, so its null cannot be read as "not used".**
`scripts/16_feature_causal.py:70-80, 98, 142`.
On the cached activations (final token, layer 15; `c3_design.py`): report-half AUROC of a fresh
adjacency probe is 0.876 before, 0.866 after projecting out the `feat_probe` direction, and 0.844
after projecting out the whole rank-4 INLP subspace. At layer 14: 0.860, 0.835, 0.797. The rank-4
subspace holds 0.013% of the activation variance. The script does not compute or save this
residual decodability, and it does not save the INLP bases (only `D`). What the raw file shows for
these columns is the numerical noise floor: mean |Δ logit difference| is 0.0035 after ablating
`feat_probe`, 0.0036 for the rank-4 subspace, 0.0030 to 0.0047 for their controls, and 0.0035 to
0.0037 for every other column whose vector is this small, whatever its content (`c7_noise.py`).
For the probe-weight family the per-item spread of (plus − minus)/2 is 0.0024 and of
(plus + minus)/2 − baseline is 0.0039, the same for `feat_probe` (own SD 0.0078 s),
`gold_probe` (0.0030 s) and the shuffled probes (0.0032 s), so it does not scale with step size.
The per-item shift effect along `feat_probe` is uncorrelated with the per-item slope (r = −0.01).
The pooled numbers are still valid as upper bounds (pooled shift effect −0.00015, CI [−0.0005,
+0.0002]; per template about ±0.001; against observational gaps of −0.23 to +0.055), but they bound
the effect of one direction out of many that carry the property.
The same holds for the models the GPU time is meant for (final token, one layer each, chosen by me,
C = 0.1; `c12_inlp_models.py`): Qwen 1.5B layer 18, 0.963 → 0.944 → 0.897; Qwen 3B layer 24,
0.955 → 0.932 → 0.878.
Fix before GPU time: erase with a method that actually removes linear decodability (LEACE, or
INLP continued until the report-half AUROC is near 0.5), save the bases, and write the residual
AUROC into the JSON. Report the noise floor next to every |Δ| statistic (this part can be done
from the saved arrays).

**F4. The pooled difference-of-means direction does not separate adjacency in the "parent" template, the one template whose predicted sign differs.**
`scripts/16_feature_causal.py:87, 94, 140`.
Report-half AUROC of the `feat_dom` projection for adjacency, by template (layer 15, final):
collider 0.74, confounder 0.63, descendant 0.66, ancestor 0.69, parent 0.46. Within-template class
gaps along it in own SDs: 0.59, 0.59, 0.56, 0.22, −0.12. Pooled AUROC 0.61 (the probe direction:
0.88, and 0.81 to 0.95 in every template). So for parent items "shift towards the other class
along `feat_dom`" does not move them towards the other class, and the per-template comparison with
the observational pattern is not interpretable for that direction. `feat_dom` is the only real
feature direction with a measurable effect, so this matters. Parent is also the weakest template
for this direction in Qwen 1.5B (layer 18: 0.55, others 0.68 to 0.78) and Qwen 3B (layer 24: 0.52,
others 0.67 to 0.75).
Fix before GPU time: per-template class-mean differences (or hypothesis-string-centred means), with
shuffles drawn the same way.

**F5. "Matched in own SDs" favours the real directions, and the conclusion depends on the convention.**
`scripts/16_feature_causal.py:136, 140-141, 343-344`.
The real directions have a larger natural SD than all 20 of their controls: `feat_probe` 0.0078 s
against 0.0029 to 0.0037; `feat_dom` 0.440 s against 0.165 to 0.438; `model_out` 0.062 s against
the same shuffled-probe range. Steps and shifts are therefore 1.5 to 2.5 times larger in norm for
the real direction. Percentile of |pooled slope| among matched controls, per own SD versus per unit
norm (`c8_units.py`): `feat_dom` 0.90 vs 0.75; `gold_dom` 0.50 vs 1.00; `feat_probe` 0.40 vs 0.20.
At the end-of-hypothesis site the probe direction's own SD is three times its controls' (0.0167 s
against 0.0053 to 0.0064) and its percentile is 0.95 per own SD against 0.35 per unit norm.
`07c` reports both conventions; `16` reports only one. Fixable afterwards for the slopes (the
arrays, `own_sd` and `s_unit` are saved); the shift columns can only be rescaled under linearity.

**F6. The shift readout "AUROC of the answer for adjacency after the shift" is mechanically tied to the slope's sign, and its control is a mean that cancels.**
`scripts/16_feature_causal.py:272-288` (line 287), shown in `12_tables.py` as "…after the shift" /
"…control mean".
The shift is +v for non-adjacent items and −v for adjacent ones. Any direction with a consistent
slope then moves the two classes apart or together, in every template, by construction. Controls
have arbitrary sign, so their mean stays at baseline. Full-size file, `feat_dom`, collider:
baseline 0.269, after the real shift 0.384, "control mean" 0.270; but the individual controls range
from 0.123 to 0.385 and the real change exceeds 90% of them in absolute value. Same in all five
templates (real change +0.11 to +0.18; control changes from −0.15 to +0.12 in collider, up to −0.30
to +0.20 in descendant). The mean pooled shift
effect equals class gap × slope to four decimals (−0.0723 both), so the shift arm carries no
information beyond the ± step arm; it costs 42 of 259 columns. The uncommitted column "…as a share
of the observed gap" divides this by gaps of ±0.05 with no interval; with the current numbers it
would print about +30%, +62%, −125%, +178%, −158% for a direction whose effect is the same −0.07
everywhere.
Fix afterwards: report the control distribution of the change (signed range and |change|
percentile), not its mean; drop the share column or give it only where the gap's CI excludes zero.

**F7. `pattern_across_templates` cannot carry a claim.**
`scripts/16_feature_causal.py:206-218, 232-240`.
(a) Five points. Under independent slopes P(r > 0.8) = 0.05; in the full-size file the 20 control
r's run from −0.89 to +0.76 for the probe family and −0.91 to +0.94 for the dom family, so
"exceeds all controls" needs r above roughly 0.8 to 0.95. (b) The observational gaps have no
uncertainty attached. With a block bootstrap: collider −0.230 [−0.282, −0.172], confounder −0.113
[−0.184, −0.026], descendant +0.055 [+0.024, +0.082], ancestor −0.042 [−0.099, +0.015], parent
+0.050 [+0.003, +0.094]. One sign is not established and r is driven by collider and confounder.
(c) `n_sign_agreements` has a default of 3/5 for any direction with a uniformly negative slope and
2/5 for a uniformly positive one (the control lists are almost all 2s and 3s); only 5/5, or the
specific flip for parent and descendant, says anything. (d) The gaps are computed on all 2400
items, selection half and evaluated items included; that is not leakage, but the halves differ
(confounder −0.157 vs −0.068). The computation itself (within hypothesis string, strings with
both classes, unweighted mean over strings) is sensible; all 48 strings qualify.
Fix afterwards: replace r by a direct test of the predicted sign difference (parent and descendant
minus the other three, with a CI and the same contrast for each control), and print the gap CIs.

**F8. A partial runner session can be ingested as complete.**
`scripts/gpu_round2.py:74-75, 213-217`, `scripts/kaggle_ingest.py:176-181, 197`.
The ingest takes the list of expected stages from the run's own `planned`, and "finished" from the
presence of the `export` key. `planned` is rebuilt as stages are reached; the status file is loaded
from the previous invocation, so `export` and `all_ok` survive. If an invocation that reached the
end (with failures, or with fewer jobs) is followed by a re-run that is cut short, the zip holds
`export` from the first and a `planned` list that stops where the second stopped: nothing missing,
nothing failed, finished. (This sequence is from reading the code; it was not reproduced, because
a stage that can be cut short needs a model.) Separately, completeness is self-declared: a run started with a reduced
`--jobs` is complete by definition (in the scratch reproduction, `--jobs export` alone passed the
stage check). Fix: write the full plan before running anything, clear `export`/`all_ok` at start,
and have the ingest compare `jobs` with the notebook's list for that run.

**F9. The within-template shuffle does not control for the hypothesis string.**
`scripts/16_feature_causal.py:62-67, 101`.
Adjacency rate varies by hypothesis string inside a template (0.38 to 0.86); a string prior fitted
on the selection half predicts adjacency on the report half at AUROC 0.60 (template prior: 0.53).
The real probe direction picks this up: over the 48 strings, its string-mean projection correlates
+0.81 with the string's adjacency rate after removing template means; the 20 controls lie between
−0.22 and +0.20 (`c9_string.py`). Most of the probe's information is within string (within-string
AUROC 0.86), so this is a second-order leak, but a positive result would be open to "it moved the
representation of which hypothesis is being asked". Fix before GPU time: shuffle within string
(24 to 26 selection-half items per string) and centre `feat_dom` by string.

**F10. `09_cross_model.py` pairs models item by item with no check that the items are the same.**
`scripts/09_cross_model.py:74-85`. For CLadder, Qwen 3B's cache is the other draw (2271 of 3000
shared). The pairing takes labels and clusters from one model and scores from the other. Today it
would stop with a shape error, because the report halves have 1639 and 1638 items; that is luck.
`scripts/queue_ingest.sh` runs `run_tier1.sh qwen3b` and then `09`. Fix: assert equal `uids` (or
intersect by uid) before pairing.

### Minor

- **M1. The four "sites" are two.** `16_feature_causal.py:125`. For Qwen 0.5B: final token best =
  layer 15, first90 = layer 14; end of hypothesis best = 13, first90 = 11. In the smoke run both
  hypothesis sites were layer 13 and the two output files are identical. Nothing detects
  best == first90. Two of four GPU stages per model buy almost nothing. Choose a clearly earlier
  site (first layer at 50%, or a fixed depth) or skip when equal.
- **M2. Ablation readout has no room to move.** `16_feature_causal.py:244-268`. Baseline AUROC of
  the answer for the gold label, per template, on 100 items: 0.56, 0.42, 0.46, 0.40, 0.60, each
  with a CI of about ±0.11. A drop from chance cannot be seen. The AUROC for the property (0.27 in
  collider) is the usable readout. These AUROCs are pooled over strings, unlike the observational
  ones (within string).
- **M3. `place` mixes conventions.** `16_feature_causal.py:157-165`. Percentile is absolute by
  default and one-sided when `signed=True`; `z_vs_controls` is always signed; `exceeds_all_controls`
  under `signed=True` tests only the upper tail, so a real value below every control gives
  `False` and percentile 0.0. The table prints an absolute percentile next to the signed control
  maximum. A NaN real value gets percentile 0.0. Pooled |slope| across templates is the headline
  line of the printout, although the hypothesis under test predicts opposite signs that cancel.
- **M4. Resume check is incomplete.** `16_feature_causal.py:363-369`. It compares K, the `idx`
  prefix and `D`. Not compared: the INLP bases and `--rank` (K does not depend on rank), `ADD_SD`.
  A checkpoint from a run with a smaller `--per-template` is a valid prefix and is accepted as
  "shortened", so the new run silently keeps the smaller size. `np.savez` to the checkpoint is not
  atomic; a kill during the write leaves a file that makes every later attempt fail at `np.load`.
  If `analyse` raises after the raw file is saved, the checkpoint is already deleted and a re-run
  repeats the whole GPU pass.
- **M5. Timing check.** `16_feature_causal.py:379-388`. Uses the first five items. On the laptop
  it was 6.2 min projected against 7.2 actual at the final token, and 16.3 against 16.5 at the end
  of the hypothesis. On CUDA the first item includes warm-up, which pushes the other way.
  Setup time is not counted against `--max-minutes`. With 45 against a 70-minute stage limit there
  is room; `gpu_round2.py:168-169` runs the final-token `07c` with its default 85 minutes inside a
  40-minute stage, and `07c` has no checkpoint, so a slow run is killed with nothing saved.
- **M6. Ingest overwrite rules.** `kaggle_ingest.py:235-241`: a conflict needs the file to exist at
  the pinned commit and to differ between the pin and HEAD. Files that are new on both sides, and
  tracked files with uncommitted local changes, are overwritten without notice.
  `kaggle_ingest.py:264-272`: local activations are replaced when the download has a strict
  superset of positions and the same uids. Nothing compares the incoming final-token activations or
  logits with the ones being replaced, and the old `meta.json` is not kept; `probe_oof_*.npz` and
  every probe result for that model then refer to activations that are no longer on disk.
  `steering*_partial.npz` files are exported and copied into the local cache.
- **M7. CLadder: what is on which sample.** Laptop draw (pinned): everything for Qwen 0.5B, Qwen
  1.5B and Gemma that was computed from the local caches, including Gemma's random-init cache.
  Other draw: Qwen 3B `cladder`, Qwen 3B random-init, Qwen 3B `cladder_sub_fewshot` (311 of 500
  shared, and different demonstrations: 29274/24584/22310/11442 against 29266/24656/22291/11547),
  Gemma's GPU steering on CLadder (handled in `07b`), and the SAE results, which were computed on
  the GPU machine's Gemma cache (`sae_features.json` has 1155 nonsense-story items, 566 for
  discovery; the GPU draw has 1155 and 566, the laptop draw 1147 and 558). `15_answer_bias.py:60` gets `other_sample` from `items_for` and
  drops it; `11_sae.py` re-run locally would replace `sae_features.json` with numbers from the
  other draw with no flag; `04_probes.py` asserts equal uids for the random-init cache, which is
  right. New GPU runs use the pinned list, so the 7B behaviour check will be on the laptop draw.
- **M8. Patching raw file lacks pair identity.** `08_patching.py:153-154` saves groups and the
  common label but not uids, swap or template. Template mix differs between the two kinds of pair
  (label-changing, kept 60: parent 38, collider 9; label-preserving: parent 21, collider 17, child
  8, and Yes/Yes pairs exist only for parent and collider), so any comparison of |D| needs
  stratifying, which currently means rebuilding the pairs with the tokenizer.
- **M9. Fallback bookkeeping.** `gpu_round2.py:158`: `KeyError` if the cache stage was skipped with
  `--skip` and the model has a fallback. In the cache job the failed 4-bit stage stays in
  `planned`, so a successful fallback still fails the ingest check; in the behaviour job it does
  not. `models.load_model` for `qwen7b` asserts two GPUs, which is right; that path and the 4-bit
  path have never run through the steering or patching hooks.
- **M10. Probes do not converge.** The log shows lbfgs stopping at `max_iter=300` for many fits.
  Directions are deterministic on one machine but need not agree between ARM and x86, so the
  "best" layer (0.840 vs 0.838 for layers 15 and 16) can differ between a laptop and a GPU run of
  the same model.

### Nits

- `16_feature_causal.py:335, 399`: `allow_pickle=True` is not needed for these files.
- Bootstrap clusters are premises; hypothesis strings recur across blocks and are not clustered.
- `06_figures.py` hard-codes "/5" and "20 directions" in the adjacency figure.
- `steer.AddAtLast._pre` assumes the hidden state is the first positional argument.
- `timeout` in `subprocess.run` kills the stage process but not joblib workers.
- No test covers `build_pairs`, the column layout of `16`, or the runner and ingest. `c1_layout.py`
  and `c4_pairs.py` are CPU-only and could be turned into tests.

## Answers to the specific questions

**Item 1, shift intervention.** Signs are right: +v for non-adjacent items, −v for adjacent ones,
and the effect is multiplied by the same sign, so it reads "towards Yes per push towards adjacent".
Magnitudes are right: the class gap along the direction on the selection half, after removing
template means. The pooled gap is not attenuated by unequal class rates (pooled 1.54 SD, mean
within-string 1.52), but it hides a spread of 1.08 to 2.19 across templates for the probe and a
sign change for the mean difference (F4). For `feat_dom` the shift vector is the whole pooled
class-mean difference. The control shift uses the real direction's gap in the control's own SDs,
which is consistent with the rest of the script and has the norm problem of F5.

**Item 1, CUDA with float16 versus MPS.** By reading, nothing breaks: outputs are moved to the CPU
before `.numpy()`, vectors are cast to the device and dtype of the activation they are added to
(this also covers the two-GPU model), and the MPS-only cache clearing is guarded. What differs and
is not controlled: (a) the numerical noise floor, measured here only for MPS float16 (0.0035
logits per reading; baseline against the cache 0.009 mean, 0.040 max), unknown for CUDA and for the
4-bit model, and it decides whether the probe-direction arm measures anything; (b) ablation
vectors are computed from cached activations (left-padded batches, larger batches on CUDA) and
added to live ones (single unpadded sequence); `baseline_max_abs_diff_vs_cache` records the
mismatch at the logit level but nothing acts on it; (c) the timing check (M5); (d) probe fits that
stop at the iteration limit (M10).

**Item 2, `steer.py` and its tests.** The generalisation is right by reading: tokens before `pos`
run once, the continuation runs with the hook at index 0, the mask has the full length.
(a) Wrong layer: caught. `test_hook_changes_exactly_the_intended_layer` records the residual stream
with an independent hook and checks layers 0, 7, 23, 24; the shortcut uses the same hook class and
is tied to it by the equality tests. (b) Wrong token: an off-by-one in either path is caught,
because the fast path indexes relative to the continuation and the reference path indexes
absolutely, and the layer-24 assertion shows neither adds at the final token. Not caught: a wrong
`pos` handed to both (that rests on `tests/test_prompts.py`), and the independent recorder test
covers only the final token. (c) Mask length: not caught, and for this use it cannot matter: one
unpadded sequence with an all-ones mask gives the same numbers for any length that does not raise.
Not covered by any test: float16, MPS/CUDA, 4-bit, two GPUs, Gemma at an earlier token. The slow
tests were not run for this review.

**Item 3.** `07c` at the final token behaves as before when the main steering JSON exists.
`choose_site` reproduces the rule in `07_steering.py`. Selection files are reused whatever stride
they were made with. Live token positions equal the cache's for the 120 evaluated items checked;
the steered token at the end of the hypothesis is its final full stop, and six tokens are re-run
per vector.

**Item 5, the "other sample" branch of `07b`.** Coherent: steered items are mapped by uid, the
GPU run's saved directions are used, and the report-half assertion would catch a different split.
Means, natural SDs and principal directions in that analysis come from the laptop's draw and the
laptop's activations; the output note records the item mapping and the source of the directions,
not this.

**Item 4, can the two kinds of pair support "patching localises a prompt difference, not a
judgement"?** Only weakly. Pooled recovery is normalised by each pair's own D and is 1 by
construction at the differing tokens early and at the final token late, for any pair of prompts.
The same picture for both kinds is what both readings predict. What the control can show is
behavioural: that mean D is near zero and |D| is as large when the label is preserved as when it
changes. That supports "the answer responds to the edit, not to the label". Compare within
template (M8).

**Item 6.** Could a failed or partial run be ingested as complete: yes, three ways (F1, F8, and a
shortened `16` run exits 0 with fewer items, which is by design and visible in `n_items`). Could the
ingest overwrite newer local results: yes, for untracked and uncommitted files (M6).

**A correction to the background.** In the debiased pool, for `has_confounder`, P(valid | adjacent)
is 0.51 and P(valid | not adjacent) is 0.48. Adjacency goes with invalid in three of the other four
templates, not four. For Qwen 0.5B the model's own dependence is negative for collider and
confounder, positive for descendant and parent, and not distinguishable from zero for ancestor.

## The adjacency experiment as coded: what it can license

**Strongest claim from a clean positive.** At this layer and token of this model, moving the
activation by a natural-scale amount along the direction fitted to adjacency changes the answer
logit difference more than directions built the same way from labels shuffled within template, and
with the per-template signs of the observed dependence. That is: later computation is sensitive to
this direction, and in the way the observational pattern predicts. It does not show that the model
uses adjacency rather than something the direction also carries (pair correlated, the hypothesis
string, F9), or that this is the route used on unmodified inputs.

**Strongest claim from a clean null.** At this site, the observed dependence of the answer on
adjacency is not carried by this one probe direction: moving an item by the full class gap along it
changes the answer by less than the CI half-width (about 0.0004 logits pooled, 0.001 per template)
where the observational gap is up to 0.23. It does not show that the adjacency information at this site is
unused, because the information is still decodable after the ablation (F3); it says nothing about
other tokens or layers.

**The two most likely misreadings.**
1. Reading the null of the probe-direction and subspace arms as "decodable but not used". Those
   arms did not remove the information and their per-item numbers are at the noise floor.
2. Reading a high percentile for `feat_dom` as an adjacency effect. In the full-size file its slope
   is −0.23 to −0.26 in all five templates, where the observed dependence changes sign; its step is
   larger in norm than every control's (F5); it does not separate the classes in parent (F4); and
   the AUROC-after-shift column looks specific only because the control is a mean (F6).

## Before more GPU time

1. F1: key markers, status and output names by `tiny`; ingest refuses tiny results.
2. F2: add the consistency assertions to `analyse`, and clear a site's old raw file when its stage
   starts (the stale smoke file itself has since been overwritten).
3. F3: an erasure that removes linear decodability, with residual AUROC and bases saved.
4. F4 and F9: per-template (or per-string) mean differences and within-string shuffles.
5. M1: replace `first90` by a site that is actually different, or skip when it equals `best`.
6. F6: drop the shift columns, or keep them and use the freed budget for a second step size
   (dose-response separates a small real slope from noise).
7. A one-item self-check on the GPU at stage start (shortcut vs full forward for three vectors, at
   the steered token, in the model's own dtype), and a threshold on
   `baseline_max_abs_diff_vs_cache`.
8. F8: full plan written up-front; stale `export` cleared. M5: `--max-minutes` below the stage
   limit for `07c`. M4: atomic checkpoint; skip the GPU pass when a matching raw file exists.
9. M8: save pair uids, swap and template in the patching raw file.
10. F10: uid assertion in `09_cross_model.py` before the queue reaches it.

## Fixable afterwards from the saved raw arrays

- Noise floor next to every |Δ| statistic (symmetric part of the ± pair) (F3).
- Percentiles per unit norm as well as per own SD (F5).
- Control distributions instead of control means for the AUROC readouts (F6).
- CIs on the observational gaps; the sign-contrast test in place of Pearson r; sign agreements
  dropped or reduced to the predicted flips (F7).
- Two-sided, consistently signed percentiles (M3); within-string AUROCs (M2); a bootstrap that also
  resamples strings.
- Template-stratified comparison of the two kinds of patching pair, after rebuilding the pair list
  (M8).

## Addendum: end-of-hypothesis site

Design checks on the cached activations at the end-of-hypothesis token (`c3_design.py`), which do
not depend on the laptop run:

| Site | Fresh-probe AUROC for adjacency, report half: before / probe direction removed / rank-4 subspace removed | `feat_dom` within-template gap in own SDs (collider, confounder, descendant, ancestor, parent) | Own SD / s: probe, mean difference |
|---|---|---|---|
| layer 13 (best) | 0.938 / 0.924 / 0.865 | 1.22, 0.97, 0.55, 0.24, 0.08 | 0.017, 0.158 |
| layer 11 (first90) | 0.918 / 0.915 / 0.900 | 0.56, 0.86, 0.41, 0.09, 0.02 | 0.006, 0.177 |

So F3 and F4 hold at this token as well.

State of the laptop run when the review closed (18:03): `adj_final_best` and `adj_hyp_best` were
complete (500 items each, 7.5 and 17.0 minutes); `adj_final_first90` was running; `adj_hyp_first90`
had not started. Neither `first90` file was checked.

`adj_hyp_best` (layer 13, 500 items), from the raw file (`c2_recompute.py`, `c7_noise.py`,
`c8_units.py`, `c13_contrast.py`):

- Every recomputed statistic equals the JSON (maximum absolute difference 0). All items in the
  report half, 100 per template, no zero rows. Baseline against the cache: 0.009 mean, 0.034 max.
- Noise floor: mean |column − baseline| is 0.0039 to 0.0042 for every probe-weight column,
  whatever the vector; 0.0039 after ablating `feat_probe`, 0.0052 for the rank-4 subspace against
  0.0039 to 0.0059 for its controls.
- No sub-batch offset is visible at either site: the ablation columns average within 0.0002 of the
  baseline in every sub-batch.
- `feat_probe`: pooled slope −0.0009 per own SD; percentile 0.95 per own SD, 0.35 per unit norm
  (F5). Pooled shift effect −0.0016, below 19 of 20 controls, against observational gaps of −0.23
  to +0.055.
- `feat_dom`: slope negative in all five templates (−0.012 to −0.043 per own SD); report-half AUROC
  for adjacency 0.52 in parent, 0.68 to 0.88 elsewhere (F4).
- Pearson r with the observed gaps: −0.54 (`feat_probe`), +0.13 (`feat_dom`); control r's from
  −0.79 to +0.76 (F7).
- The contrast proposed under F7 (mean slope in descendant and parent minus mean slope in the
  other three; predicted positive), per own SD, with an item bootstrap within template and the
  range over 20 controls:

| Site | `feat_probe` | controls | `feat_dom` | controls | `model_out` |
|---|---|---|---|---|---|
| final, layer 15 | −0.00005 [−0.0003, +0.0002] | −0.0003 to +0.0003 | −0.008 [−0.011, −0.004] | −0.014 to +0.019 | +0.0024 [+0.0014, +0.0033] |
| end of hypothesis, layer 13 | −0.0010 [−0.0012, −0.0007] | −0.0008 to +0.0010 | +0.010 [+0.009, +0.012] | −0.014 to +0.025 | −0.0029 [−0.0037, −0.0020] |

  These are raw numbers for the author to judge, computed to show the statistic is available from
  the saved arrays. The signs used for "predicted" are Qwen 0.5B's observed gaps, one of which
  (ancestor) is not established.

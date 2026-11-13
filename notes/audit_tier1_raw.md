# Tier 1 audit (independent reviewer, raw)

Scope: data/prompts/cache/behaviour/lens/probes/confound, qwen0.5b + qwen1.5b. No model was loaded; everything below was read from code or recomputed on CPU from `cache/` and `results/`. Scratch scripts: `/private/tmp/claude-501/-Users-SahilPaliwal--claude-spares-spare-259L0t/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit1/` (c1..c9.py, rerun05.py).

Bottom line: the pipeline is mechanically sound (no fold leakage, positions/layers/signs correct, every table number traces and reproduces). The problems are in what the baselines and the "model-wrong" criterion actually test. The one positive headline (qwen1.5b / Corr2Cause "criterion met", "survives structural baseline") does not survive a query-only baseline that I added.

## Findings

### 1. CRITICAL - Text baselines are handicapped; a hypothesis-string-only prior matches or beats the gold probe on Corr2Cause
- Files: `src/causal_probe/probes.py:111-122` (`query_only_oof` uses only `template`, 6 values), `:94-108` (TF-IDF over the full user text), `scripts/05_confound.py:182-205` (verdict uses only these two).
- Evidence (c2.py, c4.py, c5.py). Per-fold prior of the label given the exact hypothesis string (120 strings, e.g. "A directly causes D." is 81% positive in the pool), same grouped folds/seeds, no premise, no activations:
  - pool OOF accuracy 0.856 / 0.855 / 0.855 (seeds 0/1/2), AUROC 0.913; report half 0.859 / 0.915.
  - compare report half: gold probe 0.807 (0.5b), 0.870 (1.5b); TF-IDF 0.787; type prior 0.699; structural 0.83.
  - probe minus hyp-prior, report half, cluster bootstrap: 0.5b -0.052 [-0.077, -0.030]; 1.5b +0.011 [-0.012, +0.032].
  - on model-wrong items (the criterion), qwen1.5b: raw 0.892 vs 0.966, diff -0.074 [-0.104, -0.047]; calibrated 0.871 vs 0.899, diff -0.028 [-0.056, +0.000]. qwen0.5b: raw -0.145 [-0.184, -0.107]; calibrated -0.080 [-0.122, -0.037].
  - within strata (AUROC) the 1.5b probe beats it only in "said No" (raw +0.022 [+0.006, +0.036]; calibrated +0.046 [+0.027, +0.066]); "said Yes" is null.
  - external test set (pool-trained prior, test.csv): AUROC 0.885 vs probe 0.887 (0.5b) / 0.919 (1.5b). So it is a property of the dataset (variable naming/order), not only of the 4+4 pool sampling.
  - query-only TF-IDF (hypothesis text, 1-3 grams) also gets 0.845 on the pool, versus 0.78 for the full-text TF-IDF: the long premise dilutes the hypothesis tokens in the L2-normalised vector.
- Consequence: `verdict.represents_but_doesnt_say = True` and `post_hoc_structural_baseline.survives = True` for qwen1.5b/c2c_probe are artefacts of weak baselines. The H&L control already hints at this (probe decodes hypothesis identity well enough for 0.69-0.78 on random type labels; the real label is 86% a function of that identity).
- Fix: add the hypothesis-string prior (and query-only TF-IDF) as baselines in `04_probes.py`/`05_confound.py` and in the verdict; make `query_only_oof` use the same "type" as `control_task_labels` (`it["query"]`). Better: rebuild the pool balanced within hypothesis string, or residualise the probe target on the string prior.

### 2. MAJOR - Raw "accuracy on model-wrong items > 0.5" is yes-recall, not a test against chance
- Files: `scripts/05_confound.py:151, 181-188, 199-204`; `scripts/06_figures.py:219, 238, 244` (dashed "chance" at 0.5).
- Evidence (c5.py): fraction gold-Yes among raw model-wrong items in the report half: qwen0.5b c2c 0.998 (1 gold-No item of 599), qwen0.5b CLadder 0.965, qwen1.5b c2c 0.902, qwen1.5b CLadder 0.357. An always-Yes predictor scores 0.998 / 0.965 / 0.902 on these sets. Balanced accuracy on the same items: 0.5b c2c probe 0.418 vs TF-IDF 0.427.
- `notes/log.md` acknowledges this for qwen0.5b, but the code still emits `probe_beats_chance_on_wrong_items: True` there, and the qwen1.5b/c2c "**yes**" in `confound_wrong_items.md` rests on a 90%-Yes set, where the probe-vs-TF-IDF gap can be a threshold shift.
- Fix: on wrong items use balanced accuracy or AUROC (when both classes have enough items), or compare against the stratum's majority rate instead of 0.5; set the raw verdict to "not applicable" when `degenerate_raw_answer` or when the wrong set is >80% one class; fix the fig5 baseline/label.

### 3. MAJOR - With model answers near-independent of gold, the confound design reduces to "probe vs baseline"
- Files: `scripts/05_confound.py:156-171` (calibrated), whole design.
- Evidence: model logit-diff AUROC for gold is 0.50-0.59 everywhere (tables; recomputed). Calibrated-wrong items are therefore close to a random half: probe accuracy on calibrated-wrong vs all report items is 0.871 vs 0.870 (1.5b c2c), 0.596 vs 0.605 (1.5b CLadder), 0.558 vs 0.561 (0.5b CLadder).
- Consequence: the calibrated verdict carries no "knows but doesn't say" information beyond plain decodability, so everything depends on baseline strength (finding 1, and the structural baseline on CLadder).
- Fix: say so in the write-up; treat the within-stratum AUROC differences against the strongest text baseline as the test.

### 4. MAJOR - CLadder gold signal is available before the question is read; final position is not the best position on Corr2Cause
- Files: `scripts/04_probes.py:87-91`, `scripts/06_figures.py:193-203` (only the last model plotted), tables (final position only).
- Evidence (c5.py; layer chosen on selection half, accuracy on report half, selected C): CLadder qwen0.5b final 0.567 / content_end 0.560 / premise_end 0.564; qwen1.5b 0.598 / 0.595 / 0.572. At `premise_end` the model has not seen the question, so most of the 0.56-0.60 CLadder "gold probe" is a context-only label shortcut. Corr2Cause: content_end 0.868 (0.5b) and 0.892 (1.5b) vs final 0.803 / 0.867; premise_end 0.519 / 0.524 (correctly at chance, the pool is balanced within premise).
- Fix: report the premise_end number next to every CLadder probe claim; add a paired final-minus-premise_end CI.

### 5. MAJOR - fig2 (gold-signed logit lens split by correctness) shows the Yes/No bias, not knowledge
- Files: `scripts/03_logit_lens.py:47, 53-54`; `scripts/06_figures.py:103-117`.
- Evidence: opened `figures/fig2_logit_lens.png`: the "model correct" and "model incorrect" curves are near mirror images in all four panels. With balanced gold and a near-constant answer, "incorrect" is the gold-Yes set (finding 2), so gold-signed = +/- the raw bias curve. The log's "incorrect items leaning towards the gold answer at middle layers and being overwritten" reads this artefact as a result. The script's own docstring warns of it and stores `raw_by_gold_and_correct`, but the figure plots the confounded version.
- Fix: plot `raw_by_gold_and_correct` (four curves) or the gold-Yes minus gold-No difference; keep fig2b (AUROC) as the headline lens figure.

### 6. MAJOR (figure) / MINOR (JSON) - Outputs on disk are stale relative to the current code
- Evidence: `scripts/05_confound.py` and `06_figures.py` mtime 23:45:57; figures 23:45:22; confound JSONs 23:18-23:44. Saved confound JSONs lack `structural_accuracy_all_items` / `structural_auroc_all_items`, which the current script writes. `figures/fig3_probes_*.png` on disk draws TF-IDF at C=1 (0.750 on c2c) while tables and the current code use the selected C=10 (0.779 all items, 0.787 report half); its legend omits the orange model-output line; the structural line is absent. Regenerating with the current code (c7.py) fixes the first two.
- Effect: on-disk fig3 overstates the probe-minus-TF-IDF gap for qwen0.5b/c2c (0.80 vs 0.75 drawn; table says +0.019 [-0.007, +0.046]).
- Numbers are otherwise unaffected: re-running current `05_confound.py` for qwen1.5b/cladder into scratch reproduced the saved JSON exactly apart from the two new keys (c9.py).
- Fix: re-run 05 (all 8 configs), then 09, 06, 12.

### 7. MINOR - Hyperparameter grids end at their optimum; C=1 probes are not converged
- Files: `probes.py:19-20, 51` (`max_iter=300`, warnings suppressed at `:34-35`).
- Evidence: TF-IDF selects C=10 (grid top) in all four configs; on c2c seed-0 folds C=100 gives 0.798 vs 0.781 at C=10 (c8.py), i.e. level with the 0.5b probe. Probe selects C=1 (grid top) for qwen0.5b/c2c; at C=1 lbfgs stops at 300 iterations (fold-0 test accuracy 0.829 vs 0.835 converged, c6.py).
- Fix: extend both grids until the optimum is interior; raise `max_iter`; do not blanket-suppress convergence warnings.

### 8. MINOR - Type prior on CLadder is anti-predictive
- Files: `probes.py:111-122`.
- Evidence: all-item accuracy 0.470-0.477, report half 0.484; 8% of scores are exactly 0 and are thresholded to "No". The sample is balanced within label x query_type, so a held-out per-type prior is biased against the test fold. "Probe beats type prior" is then easier than "beats chance".
- Fix: drop it for CLadder or state that it is below chance by construction.

### 9. MINOR - Control task / selectivity reporting
- `probes.py:73-87`: sound as a memorisation control (no type unseen in any training fold; balance 0.529 / 0.498). On c2c the random control labels happen to correlate r = 0.108 with gold (seed-dependent); use several control seeds.
- `12_tables.py:132-133`, `09_cross_model.py:44-45`: "Selectivity", "Shuffled", "Control" are all-item, mean-per-seed numbers from `probes_*.json`, printed beside report-half, score-averaged-ensemble probe accuracy. Selectivity is not (table gold) minus (table control); the `cross_model.json` note "all metrics are on the report half" is false for `selectivity_*`, `max_selectivity`, `first_layer_within_1pt_*`.
- Ensemble vs per-seed gap is small (report half: 0.807 vs 0.803; 0.870 vs 0.867; 0.561 vs 0.567; 0.605 vs 0.598) but label it.

### 10. MINOR - Bootstrap caveats
- `stats.py:53`: NaN draws are dropped silently. I measured 0 of 2000 one-class draws in the two worst strata (qwen0.5b CLadder said-Yes n=64; model-wrong with 28 negatives), so no bias in current outputs; still return the dropped fraction.
- Few clusters: 24 stories (CLadder report half), 21 premises (c2c test), 15 in the said-Yes stratum. Percentile intervals under-cover at these sizes.
- Clustering by premise ignores the crossed hypothesis-string effect (finding 1), so c2c intervals are somewhat narrow.
- Paired differences are correctly paired (same resample indexes all arrays, `stats.py:52`).
- `stats.py:40-41`: `fn` is called on empty arrays before the `n == 0` guard (nit).

### 11. MINOR - Multiplicity
- 4 cells x {raw, calibrated} x {selected C, fixed C} = 16 verdicts plus 12 strata checks each, all at an unadjusted 95%. qwen1.5b/c2c calibrated flips yes -> no between selected and fixed C. The within-strata checks are computed (`05_confound.py:189-195`) but do not enter the boolean; the module docstring implies they matter.
- The boolean itself implements its stated definition exactly (`:199-204`).

### 12. MINOR - Split-half is selection-only, not a strict hold-out
- `05_confound.py:74-94`: (layer, C) is chosen on selection-half OOF accuracy and numbers are reported on the report half; verified. But the fold models scoring report items were trained on items of both halves, and vice versa. No direct optimism; effect bounded by oracle-on-report-half minus selected: 0.007, 0.000, 0.016, 0.006 (c5.py). The log's "no reported item influences any choice" is slightly too strong.

### 13. NIT
- Few-shot: all four CLadder shots come from stories also in the evaluated subset (nonsense9, nonsense8, nonsense3, penguin; different uids); two c2c shots share one dev premise (`cache.py:33-42`).
- `04_probes.py:68`: calibrated-answer median uses all items (one scalar across folds/halves).
- `04b_probe_sensitivity.py:2` docstring ("never used to pick C") is stale; diff-of-means there is unstandardised and seed 0 only.
- `tests/test_prompts.py:55` uses `max_chars=400`, production uses 450.
- `cache.py:251`: `startswith("no")` also matches "not", "none" (no effect: 0% unparseable).
- Layer labels "15/24" index the cache (0 = embeddings), so 15 = output of block 14; say so in table captions.

### 14. NOT VERIFIED (needs the model)
- That the production cache (float16, MPS, left-padded batches) equals an unbatched float32 forward. Only `tests/test_cache.py` (slow, CPU, 12 items) covers this and I did not run it. Indirect evidence is good: lens at last layer reproduces stored logits to 0.004, layer-0 vectors at `final` are identical across items, premise_end probes on c2c sit at chance. Running `-m slow` settles it.

## Checked and found correct
- Grouped folds: no group on both sides, every item held out once (pytest: 28 passed; my own StratifiedGroupKFold reproduces saved OOF scores with max |diff| = 0.0).
- c2c train/test: no shared premise; additionally no pool premise is isomorphic to another pool premise or to any test premise under variable relabelling (canonicalised all 300 + 21 premises over all permutations).
- c2c few-shot demos disjoint from evaluated premises; CLadder demos disjoint by uid. No duplicate prompts in the CLadder sample; no context shared across stories.
- TF-IDF vocabulary/IDF and the scaler are fitted on training folds only (`probes.py:50, 103-105`); type prior uses `y[tr]` only; structural baseline uses the same folds, out-of-fold, no tuning; external test check fits everything on the pool only.
- Structural baseline is advantaged, not handicapped (gets `template` / `query_type` / `graph_id`); conservative for the probe claim.
- Shuffled-label control: 0.499 / 0.510, as it should be. Random-init cache is not degraded by float16 (per-dimension std is 600-1200 float16 steps).
- Item/row alignment: `meta.uids` and `meta.labels` equal `get_dataset(tag)` for all 6 caches checked; `oof["y"]` equals labels; control labels regenerate identically.
- Positions: `final` = n_tokens-1 = the "\n" after `<|im_start|>assistant`; content_end is "." / "?" immediately before `<|im_end|>` (gap to final always 5); premise_end is ".\n" before "Hypothesis" (c2c) and the last context token before the question (CLadder); meta positions equal a fresh tokenisation, including few-shot. Left-padding offset `pos + (T - n)` is right (`cache.py:148`).
- Layer indexing: index 0 = embeddings (identical across items at `final`), last index = pre-norm output of the last block (lens reproduces logits). Consistent in cache/lens/probes/tables.
- Label 1 = Yes everywhere; logit diff = logsumexp(Yes variants) - logsumexp(No variants), recomputed from `ans_logits` (error 4e-6); top-1 token is always an answer token and agrees with the sign in >= 99.7%.
- Qwen final norm in the lens is x/rms*w with eps 1e-6, no softcap; my float64 reimplementation matches.
- No NaNs in any saved OOF array.

## Traced numbers
All 15 Tier 1 tables regenerate byte-identically from `results/*.json` with `12_tables.py` (c6.py). Independent recomputations from the cache:

| Quantity | Table / JSON | Recomputed | Match |
|---|---|---|---|
| qwen0.5b c2c_test accuracy (behaviour_zero_shot) | 0.845 | 0.8451 from meta logit_diff | yes |
| qwen0.5b c2c_test pred. Yes | 0.00 | 0.0000 | yes |
| qwen0.5b CLadder accuracy / pred. Yes | 0.500 / 0.04 | 0.4997 / 0.041 | yes |
| qwen1.5b c2c_test accuracy / pred. Yes | 0.773 / 0.10 | 0.7728 / 0.0998 | yes |
| qwen1.5b CLadder accuracy / pred. Yes | 0.497 / 0.64 | 0.4970 / 0.6437 | yes |
| qwen0.5b c2c_probe model accuracy (probes json) | 0.5021 | 0.5021 | yes |
| lens last layer vs model, max abs err, 6 caches (lens json) | 0.0031-0.0041 | 0.0031-0.0041 (own RMSNorm) | yes |
| lens AUROC layer 12, 6 caches (lens json) | 0.5248, 0.5286, 0.5081, 0.3894, 0.4022, 0.5143 | same | yes |
| qwen0.5b c2c gold probe L15 C=1 seed 0 (probes json acc_per_seed) | 0.7983 | 0.7983 (own sklearn, grouped) | yes |
| qwen0.5b CLadder gold probe L16 C=0.1 seed 0 | 0.5517 | 0.5517 | yes |
| TF-IDF C=1 seed 0, c2c / CLadder | 0.7504 / 0.5167 | 0.7504 / 0.5167 (own) | yes |
| Selected layer, C (probes_summary), 4 cells | 15/1, 16/0.1, 21/0.1, 24/0.1 | same from OOF + my split | yes |
| Gold probe acc., report half, 4 cells | 0.807, 0.561, 0.870, 0.605 | 0.8067, 0.5607, 0.8700, 0.6046 | yes |
| TF-IDF / type prior, report half, c2c | 0.787 / 0.699 | 0.7875 / 0.6992 | yes |
| TF-IDF / type prior, report half, CLadder | 0.527 / 0.484 | 0.5265 / 0.4838 | yes |
| Model acc. report half (cross_model), 4 cells | 0.501, 0.506, 0.505, 0.506 | 0.5008, 0.5058, 0.5050, 0.5058 | yes |
| n wrong raw / calibrated, 4 cells (confound_wrong_items) | 599/524, 810/791, 594/542, 810/800 | same | yes |
| Probe acc. on wrong items, raw, 4 cells | 0.835, 0.567, 0.892, 0.604 | 0.8347, 0.5667, 0.8923, 0.6037 | yes |
| Probe acc. on wrong items, calibrated, 4 cells | 0.763, 0.558, 0.871, 0.596 | 0.7634, 0.5575, 0.8708, 0.5962 | yes |
| Within-strata probe AUROC (11 rows of confound_within_strata checked) | e.g. 0.884, 0.865, 0.895, 0.694, 0.585, 0.907, 0.935, 0.674, 0.628 | same to 3 dp | yes |
| Control task at selected layer, qwen0.5b | 0.685 / 0.791 | 0.6851 / 0.7909 | yes |
| Full `confound_qwen1.5b_cladder.json` | saved | re-run of current script: identical except 2 added keys | yes |

No mismatch beyond rounding.

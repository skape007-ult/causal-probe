<!-- Sixteenth review of the project (the reviewer numbered its scratch folder 17). Raw report, unedited. -->
# Audit 17: Gemma-2-9B answers on the debiased pool (independent recomputation)

## Summary (B to F)

**B.** Every stored point estimate reproduces to four decimals with my own code. gemma9b: pooled 0.512 [0.489, 0.534]; mean within-string 0.542 (premises [0.523, 0.563], blocks [0.521, 0.564]); permutation p = 4×10⁻⁵ (200,000 premise swaps) and 4×10⁻⁴ under a null that keeps every string 25/25; 31/48 strings above 0.5; string-and-adjacency 0.541 [0.512, 0.571]; valid item higher in 52.6% of premises (binomial p = 0.08). 7B: 0.549, p ≤ 5×10⁻⁶, 33/48, 0.589, 53.6%. The 9B effect is robust to block resampling, the median (0.532), ranks versus raw logits, cell exclusions, and leaving out any block or string. It is not robust to leaving out the parent type (0.512, p = 0.39). Eight strings carry 95% of the net excess; with strings as units p = 0.04.

**C.** Six values: 0.506, 0.507, 0.473 (p = 0.004, below chance), 0.511, 0.549, 0.542. Both large models survive Bonferroni over 6 and over 78 (9B: 0.0002 and 0.003). The 9B is a prospective replication of one scalar only: mean within-string AUROC above 0.5.

**D.** The type difference is not noise: model-by-type interaction p ≤ 5×10⁻⁵ (paired swap by premise or by block), Wald χ²(4) ≈ 100, strings-as-units p = 5×10⁻⁵. The per-string correlation 7B–9B is +0.06 [−0.09, +0.28], unremarkable among the 15 pairs.

**E.** Numerical noise is irrelevant: 7B batch noise is SD 0.12 logits (9.5% of within-string SD; 9B 0.8%), and injecting it into the 9B moves 0.5419 to 0.5411. The real confound is prompt length. The 9B logit correlates with token count inside string-and-label cells from +0.80 (parent) to −0.73 (confounder), and length predicts the label within strings. With token count removed the 9B falls to 0.516–0.524 (p = 0.02–0.12); the 7B stays at 0.533–0.540 (p ≤ 0.001).

**F.** (i) The 9B ranks valid above invalid within strings on this pool, but this is not distinguishable from following prompt length. (ii) The sentence is fair for the 7B only; the 9B needs "not shown beyond prompt length". (iii) Real, partly explained: the parent difference vanishes within adjacency cells once length is removed (+0.002); the ancestor difference remains (+0.18, p = 10⁻⁴), unexplained.

---

## Findings, by severity

1. **HIGH: the gemma9b within-string result does not survive a prompt-length control; the 7B result does.**
   - Inside (string, label) cells the 9B answer logit follows token count with a type-dependent sign: parent +0.80, ancestor +0.66, descendant +0.53, collider −0.41, confounder −0.73.
   - The slope is positive in 14/14 parent, 8/9 ancestor and 9/9 descendant strings, and in 0/10 collider and 0/6 confounder strings. It explains 41% of the within-cell logit variance (7B: 18%).
   - Token count alone carries label information within a string: AUROC 0.576 for parent (p = 10⁻⁴) and 0.397 for confounder (p = 0.001). Within (string, adjacency) cells it is 0.436 overall (p < 10⁻⁴): parent 0.747, the other four types 0.37–0.42.
   - A surrogate that follows only length (the 9B's own per-string slope, residuals shuffled inside the string) gives 0.525 ± 0.010. The observed 0.542 is 1.7 SD above it; with length and adjacency the surrogate gives 0.534 ± 0.010 (0.8 SD).
   - With token count regressed out label-blind per string, the 9B gives 0.523 [0.504, 0.541], p = 0.018 (linear) or 0.516 [0.496, 0.537], p = 0.115 (quadratic in log tokens). A placebo regression on shuffled token counts gives 0.537 ± 0.003, so the drop is not a degrees-of-freedom artefact.
   - Length-matched pairs within 10%: 0.524 (p = 0.09). The best specification (pairs within 2%) gives p = 0.009, which is 0.055 after ×6.
   - Same controls on the 7B: 0.536 (p = 2.5×10⁻⁴) and 0.533 (p = 0.001); matched pairs within 10% and same adjacency 0.574 (p = 10⁻⁴).

2. **HIGH: the 7B-versus-9B type difference is real, but part of it is nuisance-cue sensitivity.**
   - Raw difference (7B − 9B): parent −0.174, ancestor +0.236, descendant +0.102; interaction p ≤ 5×10⁻⁵ under four swap variants.
   - Within adjacency cells with length removed: parent +0.002 (p = 0.98), descendant +0.055 (p = 0.15), ancestor +0.179 (p = 10⁻⁴); interaction p = 0.002.
   - So the "direct causation" reversal is explained (9B: length; 7B: adjacency), and the ancestor reversal is not.

3. **MEDIUM: the effect is concentrated and type-dependent in both models.**
   - 9B without parent strings: 0.512 (p = 0.39, 19/34 above 0.5). Without parent and confounder: 0.485.
   - 7B without ancestor: 0.521 (p = 0.08). Without ancestor and descendant: 0.482.
   - In both, dropping the top 5 of 48 strings gives p > 0.08, and the top 8 carry 95% of the net excess.
   - The spread of per-string AUROCs is SD 0.14–0.16 against 0.083 expected from noise. The mean averages opposite-signed effects.
   - With strings as the unit: t-test p = 0.043 (9B) and 0.035 (7B); neither survives ×6. The claim holds for these 48 strings, not for hypothesis strings in general.

4. **MEDIUM: "replication" holds for the scalar only.**
   - The per-string profiles of the two models are essentially uncorrelated: r = +0.06; cross-half r on disjoint blocks −0.01; type means r = −0.42; within-type residual r = +0.34.
   - The label-permutation null for this r has mean +0.24, because the two models share item-level sensitivity (item-level r = +0.24).

5. **MEDIUM: the "52.6% of premises" figure is not significant on its own.** 631/1200, binomial p = 0.078; the script reports it without an interval. A block bootstrap gives [0.510, 0.541], because the block design cancels string effects.

6. **LOW: the script's permutation null is the more liberal of two defensible ones.** Swapping labels within a premise breaks the 25/25 balance per string and gives null SD 0.0102. Swapping the valid and invalid item of the same (block, string) gives 0.0119–0.0122, matching theory (0.0825/√48). The 9B p goes from 4×10⁻⁵ to 4.3×10⁻⁴; conclusions are unchanged.

7. **LOW: two-sided reading.** gemma2b is below chance (0.473, p = 0.004; Holm over six 0.016), and it also survives ×6 under the script's null. A sentence about "larger models above chance" should sit beside "one small model below".

8. **LOW: which of each model's own type patterns hold up.**
   - 7B ancestor and descendant are solid: 9/9 strings each; 0.62–0.64 within adjacency and after length removal.
   - 7B "against the label for direct causation" (0.442) is not consistent across strings (5/14 above 0.5; string-resampling interval [0.355, 0.533]) and disappears within adjacency cells (0.530 [0.42, 0.63]).
   - 9B parent (0.616; 12/14 strings) is length-following: within adjacency it goes from 0.710 to 0.50–0.59 after length removal.
   - 9B confounder (0.634) drops to 0.57–0.62 on only 6 strings.
   - 9B "against the label for ancestor" (0.437) is weak: p = 0.02 uncorrected, 3/9 strings above 0.5, string interval [0.354, 0.519].

9. **INFO: precision is not the explanation.** The 7B's batch-composition noise is about ten times that of the other models in relative terms, but simulations show it cannot create, erase or reshape any of these statistics. There are no ties in any model.

10. **INFO: ingest and run are clean** (details in A). One restart after the time budget; the main run does not log its device map.

---

## A. Ingest and run

- **Files identical.** SHA-256 of `meta.json` matches between `kaggle_out/r3_gemma_9b/cache/gemma9b/<tag>/` and `cache/gemma9b/<tag>/` for all four tags; `readout.npz` also matches. `resid.npy` has the same byte size (1,479,475,328) and was not opened or hashed, as instructed. The 7 log files are byte-identical to the download.
- **Commit 7ae5382.** 16 files: 4 modified (`notes/log.md`, `report/tables/answer_bias.md`, `report/tables/answers_within_string.md`, `results/answer_bias.json`) and 12 added, all specific to the gemma9b run. No other pre-existing result file changed.
  - `answer_bias.json`: +275 lines, 0 deletions; the five earlier models' entries are identical before and after (JSON comparison).
  - The download carries older copies of four `probe_content_*_c2c_debiased.json` files and `steering_controls_multiplicity.json` that differ from the repository; the ingest correctly left the repository versions alone.
  - The run-side `answer_bias.json` (commit e336431, no within-string block) gives the same pooled gemma9b numbers as the local recomputation to six decimals in all four tags.
- **Logs.**
  - Two T4s. The preflight records modules per device {0: 20, 1: 26} and peak memory 9.08/9.72 GiB.
  - `behaviour.cache_part0` ended with "time budget reached; re-run to resume" (exit code 1, 52.2 min). Part 1 resumed `c2c_probe` at chunk 7/10. `c2c_debiased` was computed entirely in the first load.
  - The restart left no visible seam in `c2c_probe`: cache-to-cache noise SD 0.0075 before versus 0.0086 after; correlation with gemma2b 0.78 versus 0.80.
  - Run environment matches the 7B run (torch 2.15.0+cu132, transformers 5.22.2).
- **Data sanity.**
  - Item counts 1162/2400/2400/3000; uids unique; no NaN or inf.
  - Order and labels are identical to the other five models for `c2c_test`, `c2c_probe` and `c2c_debiased`. For `cladder` only qwen3b differs (its different sample draw). Uids and labels equal the derived item files.
  - `n_tokens` is identical to gemma2b for every item (no truncation); the final position equals `n_tokens − 1`.
  - 2,400 distinct logit differences; answer-token logits range −12.4 to 22.1 (below the soft cap of 30); top-1 token is always Yes or No (2371 No, 29 Yes).
  - The first 8 debiased logits agree with the preflight within 0.019.
- **Dtype.** Recorded as `model_dtype: torch.float16`, device `cuda:0` (first parameter); cached activations float16. The 7B records `torch.bfloat16`.
- **Timing of the statistic.** `within_string` entered the script in commit a396597 (14 Nov 08:34 −0500). The run output was written at 18:05Z (13:05 local), and the script is unchanged since. The code predates the 9B data.

---

## B. Same tests for both models, one code path

Design facts: 48 strings × 50 items (25 valid, 25 invalid each); 1,200 premises; 25 blocks. Strings per type: parent 14, ancestor 9, descendant 9, collider 10, confounder 6. String-level class imbalance and small n do not arise; the exclusion test applies to (string, adjacency) cells only.

| Statistic | gemma9b | qwen7b-4bit |
|---|---|---|
| Pooled AUROC [premise bootstrap] | 0.5122 [0.489, 0.534] | 0.5135 [0.491, 0.536] |
| Mean within-string AUROC | 0.5419 | 0.5491 |
| – interval, premises resampled | [0.523, 0.563] | [0.530, 0.569] |
| – interval, blocks resampled | [0.521, 0.564] | [0.529, 0.569] |
| Permutation p, premise swap (200,000) | 4.0×10⁻⁵ (z = 4.11) | 5×10⁻⁶, at floor (z = 4.81) |
| Permutation p, (block, string) swap (200,000) | 4.3×10⁻⁴ (z = 3.52) | 6.0×10⁻⁵ (z = 4.02) |
| Strings above 0.5 (sign test) | 31/48 (p = 0.060) | 33/48 (p = 0.013) |
| Strings as units: t-test / Wilcoxon | 0.043 / 0.024 | 0.035 / 0.026 |
| Median over strings [premises], perm p | 0.532 [0.511, 0.582], 0.019 | 0.539 [0.513, 0.592], 0.004 |
| Trimmed mean (5 strings each side) | 0.547 | 0.553 |
| Within (string, adjacency) cells | 0.5411 [0.512, 0.571], p = 1.4×10⁻⁴ | 0.5891 [0.559, 0.619], p = 5×10⁻⁶ |
| – cells with min class ≥5 / ≥8 / ≥10 | 0.541 / 0.528 / 0.546 | 0.589 / 0.580 / 0.579 |
| – unweighted mean of cells | 0.533 | 0.595 |
| Premises with valid item higher | 52.6% (631/1200), binomial p = 0.078 | 53.6% (643/1200), p = 0.014 |
| Same block and string: valid above invalid | 54.6%; 17/25 blocks; p = 0.004 | 55.2%; 20/25 blocks; p = 0.0009 |
| Leave one block out (range) | 0.538–0.546 | 0.546–0.554 |
| Leave one string out (range) | 0.534–0.549 | 0.542–0.556 |
| Leave out parent | **0.512 (p = 0.39)** | 0.593 |
| Leave out ancestor | 0.566 | **0.521 (p = 0.079)** |
| Leave out descendant | 0.541 | 0.526 (p = 0.027) |
| Leave out collider | 0.560 | 0.565 |
| Leave out confounder | 0.529 (p = 0.006) | 0.546 |
| Worst two-type omission | 0.485 (parent + confounder) | 0.482 (ancestor + descendant) |
| Drop top 3 / 5 / 8 strings | 0.526 (p = 0.017) / 0.517 (0.13) / 0.503 | 0.530 (0.005) / 0.519 (0.085) / 0.503 |
| Share of net excess in top 5 / top 8 strings | 64% / 95% | 66% / 95% |
| Raw string-centred difference, valid − invalid (t over premises; over blocks) | +0.138 logits (4.76; 3.88) | +0.259 (5.59; 5.11) |
| Within-string rank difference (t) | 4.22 | 4.95 |
| With the Yes-answered items removed | 0.537 (29 items, all confounder, 23 valid) | 0.548 (173 items) |
| **Token count removed, linear** | **0.523 [0.504, 0.541], p = 0.018** | 0.536 [0.517, 0.555], p = 2.5×10⁻⁴ |
| **Token count removed, quadratic in log** | **0.516 [0.496, 0.537], p = 0.115** | 0.533 [0.514, 0.553], p = 0.001 |
| – same, within adjacency cells | 0.528 (p = 0.006) / 0.518 (p = 0.09) | 0.557 / 0.552 (both p ≤ 5×10⁻⁵) |
| Length-matched pairs (<10%); and same adjacency | 0.524 (p = 0.09); 0.527 (p = 0.09) | 0.546 (p = 4×10⁻⁴); 0.574 (p = 10⁻⁴) |

Reading: under exactly the tests applied to the 7B, the 9B effect holds, somewhat weaker. It fails the one test that had not been applied to either model, the prompt-length control.

---

## C. Multiplicity

| Model | Mean within-string [premises] | p, premise swap | p, (block, string) swap | p, strings as units | ×6 (premise swap) | ×78 (premise swap) | ×6 / ×78 (block-string) |
|---|---|---|---|---|---|---|---|
| qwen0.5b | 0.506 [0.487, 0.524] | 0.54 | 0.63 | 0.73 | 1 | 1 | 1 / 1 |
| qwen1.5b | 0.507 [0.489, 0.525] | 0.43 | 0.55 | 0.75 | 1 | 1 | 1 / 1 |
| gemma2b | 0.473 [0.454, 0.492] | 0.0040 | 0.029 | 0.21 | 0.024 | 0.31 | 0.18 / 1 |
| qwen3b | 0.511 [0.492, 0.531] | 0.24 | 0.35 | 0.59 | 1 | 1 | 1 / 1 |
| qwen7b-4bit | 0.549 [0.530, 0.569] | 5×10⁻⁶ | 6×10⁻⁵ | 0.035 | 3×10⁻⁵ | 3.9×10⁻⁴ | 3.6×10⁻⁴ / 0.0047 |
| gemma9b | 0.542 [0.523, 0.563] | 4×10⁻⁵ | 4.3×10⁻⁴ | 0.043 | 2.4×10⁻⁴ | 0.0031 | 0.0026 / 0.034 |

- **Holm over six (premise swap):** 7B 3×10⁻⁵, 9B 2×10⁻⁴, gemma2b 0.016, the rest above 0.7.
- **Family of 78** = 6 models × 13 statistics per model (pooled; mean; 5 types; cell; 5 cell-by-type). Of the 72 with a permutation p, 42 are below 0.05 and 18 are below 0.05/78. Every model has at least one by-type statistic that survives, so type-level heterogeneity is ubiquitous, not a large-model property.
- **Fair family.** For "which models follow validity within string", I consider 6 models × 2 headline statistics (mean, string-and-adjacency) = 12, two-sided. The by-type numbers are descriptive and were never the claim. Both large models pass 12 and also pass 78.
- **Prospective status.** For the 7B all within-string statistics were post hoc. For the 9B the mean (and the cell statistic) were fixed in code before its data existed, so it is a prospective replication with a family of one or two (one-sided p ≈ 2×10⁻⁵). It replicates only "mean within-string AUROC above 0.5", and loosely the cell statistic (0.541 versus 0.589). It does not replicate the type pattern, which reverses for parent and ancestor.
- **After the length control**, the 9B (p = 0.018 to 0.115) fails any family larger than one. The 7B (p ≤ 0.001) still passes ×6.

---

## D. The type-by-type pattern

**(1) Type means of the within-string AUROC, three kinds of interval**

| Type (strings) | Model | Estimate | Strings resampled within type | Premises | Strings and premises | Strings above 0.5 |
|---|---|---|---|---|---|---|
| parent (14) | 7B | 0.442 | [0.355, 0.533] | [0.402, 0.481] | [0.339, 0.549] | 5/14 |
| | 9B | 0.616 | [0.559, 0.681] | [0.576, 0.655] | [0.532, 0.697] | 12/14 |
| ancestor (9) | 7B | 0.673 | [0.597, 0.748] | [0.622, 0.725] | [0.572, 0.771] | 9/9 |
| | 9B | 0.437 | [0.354, 0.519] | [0.386, 0.488] | [0.327, 0.541] | 3/9 |
| descendant (9) | 7B | 0.649 | [0.594, 0.704] | [0.597, 0.698] | [0.560, 0.729] | 9/9 |
| | 9B | 0.548 | [0.475, 0.620] | [0.497, 0.602] | [0.442, 0.654] | 5/9 |
| collider (10) | 7B | 0.487 | [0.432, 0.540] | [0.437, 0.534] | [0.388, 0.572] | 5/10 |
| | 9B | 0.473 | [0.394, 0.539] | [0.425, 0.526] | [0.376, 0.575] | 6/10 |
| confounder (6) | 7B | 0.567 | [0.513, 0.632] | [0.502, 0.631] | [0.460, 0.683] | 5/6 |
| | 9B | 0.634 | [0.552, 0.699] | [0.569, 0.694] | [0.515, 0.738] | 5/6 |

Block-resampling intervals are within ±0.01 of the premise ones.

**(2) Tests of the model-by-type interaction (7B versus 9B)**

| Test | Result |
|---|---|
| Paired swap of the two models' within-string z-scores per premise (20,000) | p = 5×10⁻⁵ (floor); observed statistic 1.04 against a null maximum of 0.26 |
| Same, per block (20,000) | p = 5×10⁻⁵ (floor) |
| Same with normal scores of ranks, per premise / per block (5,000) | p = 2×10⁻⁴ (floor) each |
| Wald χ²(4), paired premise / block bootstrap covariance | 99.7 (p ≈ 10⁻²⁰) / 105.8 |
| Strings as units: ANOVA of per-string difference on type, type labels permuted (20,000) | F = 12.4, p = 5×10⁻⁵; type explains 53% of the variance of the difference |
| Per-type difference 7B − 9B [paired premises] | parent −0.174 [−0.222, −0.127]; ancestor +0.236 [+0.173, +0.302]; descendant +0.102 [+0.028, +0.175]; collider +0.014 (n.s.); confounder −0.066 [−0.135, +0.002] |
| Same, strings resampled | parent [−0.245, −0.094]; ancestor [+0.139, +0.329]; descendant [+0.035, +0.174] |
| Main effect (difference of means) | +0.007, p = 0.54 |
| Same interaction test over all 15 model pairs (2,000 swaps) | 14 of 15 pairs p < 0.05 (exception: 0.5B versus 3B); every model has its own type profile |

**(3) Correlation of the 48 per-string AUROCs between models**

Split-half reliability of a model's own profile (12 versus 13 blocks, Spearman-Brown): 0.52 to 0.76.

| Pair | r [paired premise bootstrap] | Label-permutation null: mean [95%], p | Cross-half r | Type means r / within-type r |
|---|---|---|---|---|
| 7B – 9B | +0.06 [−0.09, +0.28] | +0.24 [−0.04, +0.49], 0.18 | −0.01 | −0.42 / +0.34 |
| 7B – 3B | −0.10 [−0.23, +0.14] | +0.12, 0.12 | −0.10 | −0.62 / +0.09 |
| 7B – 1.5B | −0.22 [−0.32, +0.01] | +0.05, 0.07 | −0.19 | −0.84 / +0.08 |
| 7B – 0.5B | −0.24 [−0.37, +0.03] | −0.02, 0.13 | −0.17 | −0.88 / +0.04 |
| 0.5B – 1.5B | +0.27 [+0.03, +0.43] | +0.15, 0.37 | +0.16 | +0.66 / +0.14 |
| 0.5B – 3B | +0.33 [+0.07, +0.45] | +0.12, 0.14 | +0.21 | +0.79 / +0.21 |
| 1.5B – 3B | +0.29 [+0.06, +0.40] | +0.05, 0.10 | +0.22 | +0.74 / +0.18 |
| gemma2b – gemma9b | −0.18 [−0.34, +0.02] | −0.13, 0.76 | −0.12 | −0.45 / −0.04 |
| 9B – 1.5B | +0.27 [+0.08, +0.43] | +0.23, 0.73 | +0.18 | +0.32 / +0.19 |
| 9B – 3B | −0.12 [−0.30, +0.09] | −0.09, 0.86 | −0.08 | −0.37 / −0.10 |
| 9B – 0.5B | +0.07 [−0.16, +0.25] | −0.05, 0.42 | +0.08 | −0.03 / +0.07 |
| gemma2b – 0.5B / 1.5B / 3B / 7B | +0.21 / +0.07 / +0.29 / +0.19 | p = 0.89 / 0.16 / 0.64 / 0.49 | +0.10 / −0.01 / +0.18 / +0.14 | — |

- 7B–9B is not unusual. The 7B is the odd one inside its own family (negative with all three smaller Qwens), and the two Gemmas are also negatively related.
- No pair's r differs significantly from its label-permutation null (smallest p = 0.07). That null is non-zero because models share label-independent item sensitivities.

**(4) Adjacency of the queried pair**

Design: in parent strings every valid item is adjacent and 13% of invalid ones are (46 items). In ancestor, descendant and collider strings valid items are adjacent in 42–44% of cases and invalid ones in 76–80%.

| | parent | ancestor | descendant | collider | confounder |
|---|---|---|---|---|---|
| 7B observed | 0.442 | 0.673 | 0.649 | 0.487 | 0.567 |
| 7B expected from adjacency alone | 0.531 | 0.545 | 0.501 | 0.477 | 0.499 |
| 7B within adjacency cells [premises] | 0.530 [0.43, 0.63] | 0.642 [0.58, 0.71] | 0.640 [0.58, 0.70] | 0.537 [0.48, 0.59] | 0.557 [0.48, 0.64] |
| 7B within adjacency, length removed (quadratic) | 0.503 | 0.618 (p < 10⁻⁴) | 0.616 (p = 10⁻⁴) | 0.472 | 0.515 |
| 9B observed | 0.616 | 0.437 | 0.548 | 0.473 | 0.634 |
| 9B expected from adjacency alone | 0.475 | 0.538 | 0.518 | 0.426 | 0.505 |
| 9B expected from length alone | 0.554 | 0.488 | 0.511 | 0.514 | 0.553 |
| 9B within adjacency cells [premises] | 0.710 [0.62, 0.79] | 0.405 [0.34, 0.47] | 0.535 [0.47, 0.60] | 0.559 [0.50, 0.62] | 0.605 [0.53, 0.68] |
| 9B within adjacency, length removed (quadratic) | 0.501 [0.40, 0.61] | 0.440 (p = 0.04) | 0.562 (p = 0.03) | 0.525 | 0.561 (p = 0.09) |
| Difference 7B − 9B within adjacency [premises] | −0.179 [−0.30, −0.06] | +0.238 [+0.16, +0.32] | +0.104 [+0.02, +0.19] | −0.023 | −0.048 |
| Difference within adjacency, length removed (swap p) | +0.002 (0.98) | +0.179 (10⁻⁴) | +0.055 (0.15) | −0.053 (0.13) | −0.046 (0.34) |

Adjacency alone explains little of either profile, and both survive conditioning on it (profile r between the models within adjacency cells: −0.80). Adding prompt length removes the parent difference entirely.

**Conclusion.**
- The sign difference is real as a statistical matter: p ≤ 5×10⁻⁵ by the paired premise-swap and block-swap interaction tests, and p = 5×10⁻⁵ with strings as units.
- Its parent component is produced by different nuisance sensitivities (9B to length, 7B to adjacency), not by different treatment of direct causation.
- Its ancestor component survives both controls (interaction within adjacency with length removed: p = 0.002).
- Own patterns: real for 7B ancestor and descendant; for 9B only weakly for confounder and descendant after controls; 7B "against parent" and 9B "against ancestor" are not robust across strings.

---

## E. Precision and other confounds

**(1) Numerical noise, from the 51 items present in both `c2c_debiased` and `c2c_probe`** (identical uid and text; same token count; different batch composition). The two pools share 174 premises but only 51 items.

| Model | dtype | SD of cache-to-cache difference | Max abs difference | Per-run noise SD | As % of mean within-string SD | Answer sign flips |
|---|---|---|---|---|---|---|
| qwen0.5b | float16 | 0.014 | 0.035 | 0.010 | 3.5% | 0 |
| qwen1.5b | float16 | 0.012 | 0.030 | 0.008 | 1.0% | 0 |
| gemma2b | float16 | 0.014 | 0.028 | 0.010 | 1.2% | 0 |
| qwen3b | float16 | 0.034 | 0.106 | 0.024 | 1.0% | 0 |
| qwen7b-4bit | bfloat16 + 4-bit | 0.170 | 0.451 | 0.120 | **9.5%** | 0 |
| gemma9b | float16, 2 GPUs | 0.008 | 0.029 | 0.006 | 0.8% | 0 |

Independent check for the 9B: the preflight's first 8 debiased logits (another batch) differ from the cache by SD 0.010, maximum 0.019.

**(2) Granularity.** All six models have 2,397 to 2,400 distinct logit differences out of 2,400. No item is tied with another item of its string, and there are 0 tied valid-invalid pairs of 30,000 in every model. No grid is visible in the answer-token logits. Median gap between neighbours inside a string: 0.060 (7B), 0.032 (9B).

**(3) Simulations**

| Manipulation | Mean within-string | Types (parent / ancestor / descendant / collider / confounder) |
|---|---|---|
| 9B as cached | 0.5419 | 0.616 / 0.437 / 0.548 / 0.473 / 0.634 |
| 9B + noise at the 7B's relative size (10% of string SD), 300 draws | 0.5411 ± 0.0014 | 0.614 / 0.438 / 0.547 / 0.472 / 0.631 |
| 9B + 5× that | 0.536 ± 0.006 | 0.603 / 0.449 / 0.540 / 0.471 / 0.618 |
| 9B + 10× that | 0.529 ± 0.008 | 0.580 / 0.460 / 0.529 / 0.475 / 0.598 |
| 9B rounded to a 0.10 / 0.25 / 0.50 grid | 0.5413 / 0.5413 / 0.5364 | unchanged to ±0.01 |
| qwen3b + noise at the 7B's relative size | 0.5115 → 0.5115 ± 0.0016 | unchanged to ±0.003 |
| 7B + its own measured noise again | 0.5491 → 0.5494 ± 0.0017 | unchanged to ±0.004 |
| 7B + 10× / 20× / 40× measured noise | 0.539 / 0.526 / 0.514 | shrink towards 0.5; no sign change |
| Pure noise within string (no label information), 2,000 draws | 0.5005 ± 0.0119, maximum 0.540 | — |

Independent noise only attenuates towards 0.5. It cannot manufacture a within-string AUROC or a type pattern, and erasing the 7B's would take noise about 40 times the measured size. Coarsening the 9B to far beyond the 7B's effective resolution changes nothing.

**(4) Other confounds visible in the files**

| Model | Yes rate | Logit mean / SD | Variance between strings | r(logit, tokens) within (string, label) | By type (parent / ancestor / descendant / collider / confounder) | Within-string, raw → length removed (linear / quadratic log) |
|---|---|---|---|---|---|---|
| qwen0.5b | 0.4% | −1.06 / 0.38 | 46% | +0.07 | +0.05 / +0.18 / −0.30 / +0.35 / +0.04 | 0.506 → 0.517 / 0.512 |
| qwen1.5b | 16.3% | −2.14 / 2.14 | 83% | +0.36 | +0.28 / +0.47 / +0.27 / +0.54 / +0.19 | 0.507 → 0.512 / 0.509 |
| gemma2b | 30.8% | −2.85 / 3.51 | 94% | −0.21 | −0.35 / −0.16 / −0.67 / +0.05 / +0.29 | 0.473 → 0.482 / 0.489 |
| qwen3b | 3.9% | −15.3 / 7.31 | 87% | −0.47 | −0.49 / −0.59 / −0.42 / −0.55 / −0.17 | 0.512 → 0.519 / 0.513 |
| qwen7b-4bit | 7.2% | −15.3 / 9.44 | 98% | −0.09 | +0.14 / +0.05 / −0.15 / −0.29 / −0.38 | 0.549 → 0.536 / 0.533 |
| gemma9b | 1.2% | −6.22 / 2.48 | 90% | +0.28 | **+0.80 / +0.66 / +0.53 / −0.41 / −0.73** | 0.542 → **0.523 / 0.516** |

- **Answer tokens.** Six Yes and six No variants for every model; 'Yes' and 'No' dominate. Using only 'Yes' minus 'No', or the maximum variant, gives identical AUROCs to four decimals for all six models.
- **Tokenisers.** Qwen 172–1675 tokens, Gemma 152–1654; the two counts correlate at 1.000 across items, so "length" is the same variable for both families.
- **Number of variables.** Not the channel: inside (string, num_variables) cells the 9B gives 0.543 and the 7B 0.552.
- **Length split of same-string pairs for the 9B.** Parent: concordance 0.890 when the valid prompt is longer, 0.241 when shorter. Confounder: 0.319 versus 0.840. The 7B is much flatter (parent 0.507 / 0.354; ancestor 0.706 / 0.644).
- **Yes rate and logit scale** do not enter a rank statistic within string; removing the Yes-answered items changes nothing.

---

## F. Verdict

**(i) gemma9b on the debiased pool.** The run and ingest are clean, and the numbers are right. The model answers No to 98.8% of items, and its pooled AUROC is at chance (0.512 [0.489, 0.534]). Inside hypothesis strings its answer logit ranks valid above invalid items slightly more often than chance (0.542; p between 4×10⁻⁵ and 4×10⁻⁴ depending on the null), which holds under every test previously applied to the 7B and under correction over six models. That is where the supportable statement ends. The excess sits in parent and confounder strings (0.512 without the parent type), and with strings as the unit it is p = 0.04. Above all, the 9B's logit is strongly driven by prompt length with a type-specific sign, prompt length predicts the label inside strings, and a length-only surrogate reproduces most of the effect. With length removed the estimate is 0.516 to 0.524 with p from 0.02 to 0.12. What may be said: "within hypothesis string the 9B's answers rank valid items slightly higher (0.542); this is not shown to go beyond its sensitivity to prompt length (0.52, not significant after correction)".

**(ii) "The two largest models (7B, 9B; two families) follow validity weakly within a hypothesis string."** As a description of the uncontrolled statistic it is accurate for both, and the 9B is a genuine prospective replication of that one scalar. As a statement about validity it is fair for the 7B: it survives adjacency (0.589), prompt length (0.533 to 0.540, p ≤ 0.001), length-matched pairs, and both together, although it rests on the ancestor and descendant types. For the 9B it needs the qualification that the effect is not established beyond prompt length. For both it needs three more:
- "on these 48 strings" (p ≈ 0.04 with strings as units);
- "driven by different hypothesis types in the two models" (per-string profiles uncorrelated, r = +0.06);
- "one smaller model, Gemma-2-2B, is below chance (0.473)".

"Two families" should not be read as generalisation across families: the two Gemmas are negatively related, and the 7B is negatively related to the smaller Qwens.

**(iii) The type-by-type difference.** It should not be described as within noise: the model-by-type interaction is p ≤ 5×10⁻⁵ by paired premise and block swap tests, and p = 5×10⁻⁵ treating strings as units. It should not be described as wholly unexplained either. The direct-causation (parent) reversal is accounted for by nuisance cues: the 9B follows prompt length, the 7B follows adjacency, and within adjacency cells with length removed both sit at 0.50 (difference +0.002, p = 0.98). The ancestor reversal (7B 0.62 to 0.67, 9B 0.44) survives both controls (+0.18, p = 10⁻⁴) and is real and unexplained. The descendant difference is no longer significant after the controls. A fair wording: "the type profiles differ beyond noise; the direct-causation difference is explained by prompt length and adjacency, the indirect-causation (ancestor) difference is not". Whether quantisation or model family causes the remaining difference is undetermined: the comparison confounds family, size, precision and tokeniser, with one model each.

---

## What I could not check

- `resid.npy` content or hash (not opened, as instructed); identity is established by byte size only. `readout.npz` was hashed but not inspected.
- The systematic effect of 4-bit quantisation and bfloat16 on the 7B, and of float16 on the 9B. There is no full-precision reference run for either. The measured noise is batch-composition noise from only 51 shared items and does not capture deterministic error.
- The device map of the main 9B run: only the preflight logs modules per device, and nothing shows whether both model loads used the same split.
- The exact prompt text and chat template fed to each model; they are not stored in the meta files. Family differences were seen only through token counts.
- Whether a written prediction for the 9B existed before its data. I did not read `notes/log.md` (one attempt to list its headers was blocked, and I did not pursue it). The prospective status rests on commit timestamps alone, which the author controls.
- Generated-text answers; everything here is from answer logits.
- Shallow cues other than adjacency, token count and number of variables. Whether the 7B's remaining effect survives the repository's stronger text-only features was not tested.
- Whether prompt length is a pure nuisance or a proxy for graph sparsity that is legitimately related to validity. The data cannot separate these; either way it is a premise-level cue, not an evaluation of the hypothesis.
- Generalisation beyond these 48 strings (6 to 14 per type) and this premise sample.
- Disclosure: I opened `report/tables/answers_within_string.md` once, because the brief names it as the result table, to cross-check numbers. I read nothing else under `report/`.

The repository was not modified by me. The untracked files that appeared during the audit (`results/probes_gemma9b_c2c_debiased_C0.01.json`, `cache/gemma9b/c2c_debiased/probe_oof_C0.01.npz`, and a change to `notes/log.md`) come from the running probe job.

My scripts and intermediate outputs are in `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit17/`:
- `core.py` (shared loaders and statistics)
- `a_checks.py` (ingest and meta checks)
- `b0_repro.py`, `b1.py`, `b2.py` (reproduction, bootstrap and permutation tests, robustness)
- `c1.py` (multiplicity, token-controlled interaction)
- `d1.py`, `d2.py` (type profiles, interaction tests, correlations, adjacency)
- `e1.py`, `e2.py`, `e3.py`, `e4.py` (noise, granularity, simulations, prompt length)
- `f1.py` (answer-token variants)
- `b1_res.json`, `noise.json`, `per_string.json`, `answer_bias_before.json` (intermediate outputs)

# Tier 2 audit (independent reviewer): steering, patching, cross-model

Received Fri 2026-11-13 11:37. The reviewer was a fresh agent given the code, the result files and
the raw per-item outputs, not my conclusions, and one claim to attack. It could not run any model.
The text below is its report, lightly reformatted. My responses are in `notes/log.md`.

**Verdict on the claim** ("along the gold-label probe direction, steering and ablation have no more
effect on the answer than random directions of equal norm, while the model-output direction and the
gold difference-of-means direction flip most answers"): **does not survive as stated.** A narrower
version survives; wording at the end. Every number in the JSON files and tables reproduces from the
raw files, so the problems are in the controls, the metrics and the interpretation, plus one
stale-results issue.

Conventions: `s` = `s_unit`; "slope" = (LD(+α) − LD(−α)) / 2α at α = 1, in logits per `s`; CIs are
cluster bootstraps over premises/stories.

Core numbers, recomputed from `steering.npz` and the cache:

| Setting (layer) | Gold-probe slope [95% CI] | 3 random slopes | Diff-of-means | Model-out | α = 1 in own SDs (probe / diff-means / model-out) | Probe AUROC for gold, report half |
|---|---|---|---|---|---|---|
| 0.5B Corr2Cause (15) | +0.098 [+0.093, +0.102] | +0.03, +0.49, −0.27 | +0.87 | +1.91 | 38 / 2.9 / 9.5 | 0.816 |
| 0.5B CLadder (22) | +0.024 [+0.016, +0.032] | +0.17, +0.32, −0.16 | −1.03 | +2.07 | 55 / 2.5 / 7.4 | 0.535 |
| 1.5B Corr2Cause (21) | −0.366 [−0.376, −0.357] | −0.14, −0.35, −0.09 | +1.99 | +2.35 | 63 / 1.9 / 2.9 | 0.915 |
| 1.5B CLadder (24) | −0.058 [−0.068, −0.050] | +0.14, −0.26, +0.20 | +0.72 | +3.55 | 150 / 2.5 / 5.2 | 0.622 |

## Findings: steering

1. **Critical. In Qwen-1.5B / Corr2Cause the gold-probe direction is not inert, and the table hides it.** The slope is −0.37, negative for 100% of items and linear from α = 0.25 to 2: pushing toward gold-Yes lowers the Yes−No logit difference. At α = 4 the minus sign flips 30% of answers and at α = 8, 47%; the table and fig. 6 show only `flip_rate_plus` (0.10). Toward-gold accuracy is 0.31 at α = 4 and 0.24 at α = 8; away-from-gold 0.71 and 0.79. Best-sign flip rate at α = 4 is 0.30 against 0.05 / 0.21 / 0.13 for the three random directions; all three paired differences exclude 0. The effect equals the strongest random direction and exceeds the mean of three. Fix: report both signs, report the signed slope, state this row explicitly.
2. **Major. Three random directions cannot support "no more than random", and "mean of 3" is not a meaningful summary.** Random slopes span 17-fold within one setting. `random_mean` averages sign-dependent metrics over directions of arbitrary sign. There is no uncertainty over directions. Gold-probe slope divided by the RMS of the random slopes is 0.31, 0.11, 1.64 and 0.28 across the four settings; neither a difference nor an equivalence can be established. Fix: at least 100 random directions; report the gold-probe slope as a percentile of the |slope| distribution.
3. **Major. Norm matching is formally correct, but the scale is far outside the probe direction's natural range.** All directions are unit-normalised in raw activation space and scaled by the same `s`. Along the gold-probe direction the projection SD is only 0.007–0.026 `s`; the natural gap between gold classes along it is 0.029 `s` (0.5B) and 0.022 `s` (1.5B) on Corr2Cause. So α = 0.25 is already 9–11 class gaps and 10–37 own SDs; α = 8 is 300–1,200 SDs. Extrapolating the linear response, moving the probe readout across the whole class gap changes the logit difference by +0.003 (0.5B) and −0.008 (1.5B); the logit-difference SD is 0.40 and 2.26. In own-SD units the picture does not change: probe 0.003–0.006 logits per SD, random 0.001–0.011, diff-of-means 0.30–1.06, model-out 0.20–0.82. So the comparison is not unfair to the probe direction; this is the strong, defensible part of the result. The docstring's "alpha = 1 is one typical item-to-item deviation" and the fig. 6 axis label are misleading: `s` is a total norm over all dimensions, not a spread along one direction.
4. **Major. The probe direction and the diff-of-means direction are two views of the same discriminant**, so "probe inert" does not mean "variable unused". Cov·w has cosine 0.96–0.99 with the diff-of-means direction in all four settings. The probe direction has 0.0–0.5% of its squared norm in the top-10 principal components; diff-of-means has 72–97%. Legitimate reading: later layers do not read the low-variance subspace the probe weights point into.
5. **Major. "Gold diff-of-means flips most answers" is not specific to gold and has no matched control.** On CLadder the direction carries no gold information (split-half cosine −0.37 and −0.19; class-mean gap at the shuffled-label level; report-half AUROC 0.49 and 0.50), yet it flips 95% of answers in 0.5B and every flippable answer in 1.5B at α = 8. On Corr2Cause it separates the model's own answer at least as well as the gold label (AUROC 0.94 vs 0.65 in 1.5B; 0.70 vs 0.67 in 0.5B). Isotropic random directions have projection SD about 0.03 `s`; diff-of-means 0.34–0.53 `s`; shuffled-label diff-of-means 0.32–0.40 `s`: the control is unmatched. "Most answers" needs α ≥ 2–4, which is 6–12 own SDs. Fix: add shuffled-label diff-of-means and random directions drawn from the data covariance.
6. **Major. The CLadder rows are vacuous for the probe claim.** The steered probe direction decodes gold at report-half AUROC 0.535 (0.5B) and 0.62 (1.5B); split-half cosine 0.21 and 0.23, against 0.79 and 0.67 on Corr2Cause. Fix: restrict the claim to Corr2Cause; store each direction's report-half AUROC.
7. **Major. The 0.5B results are stale relative to the current layer and C selection.** They were produced with fixed C = 0.01 before `05_confound.py` was re-run. Current selection: layer 15, C = 1.0 (Corr2Cause) and layer 16, C = 0.1 (CLadder); the saved runs used layer 15, C = 0.01 and layer 22, C = 0.01. The two probe directions have cosine 0.60. Fix: rerun 0.5B, or label these as the fixed-C configuration.
8. **Major. Flip rate and "accuracy toward gold" are poor metrics here.** The models answer No on 89–100% of items in three settings; flip rate is a threshold on slope × α against the margin. "Accuracy toward gold" uses the gold label to choose the sign, so any direction with a consistent slope goes to 1 or 0 (a random direction reaches 1.00). Fix: headline the signed slope with a CI; rename the column "controllability".
9. **Major for the ablation half. Ablation of the probe and random directions is below the noise floor.** Observed mean |ΔLD| 0.004–0.010; the residual after subtracting the linear prediction has SD 0.005–0.008 for every direction; the predicted probe effect is 0.0006–0.005. The floor is probably float16 forward noise. Fix: drop the ablation comparison for these directions, or run in float32.
10. **Major. One layer and one position cannot show "not used".** Steering is only at the final token, at or after the layer where patching says prompt-specific information arrives. The model-out direction is a positive control, not a finding (it is fitted to the sign of the quantity steering measures); its cosine with the Yes−No unembedding difference is 0.12–0.21 and with the gold probe |0.01–0.05|.
11. **Minor. Uncertainty.** Clustering is correct (137 premises; only 24 stories for CLadder, where percentile intervals under-cover). Zero counts give degenerate intervals. No paired comparisons between directions in the JSON or table.
12. **Minor. Script hygiene.** The checkpoint resume checks layer, items and shape but not the directions, C or `s`. The baseline-vs-cache check is now only a warning. `random_mean.add.mean_abs_delta_logit_diff` is the absolute value of a mean, not a mean of absolute values.
13. **Minor. `tests/test_steer.py`.** Both paths use the same hook class, so hooking the wrong layer would pass; layer 24 is exempt from the "steering does something" check without explanation; CPU float32 while the runs were MPS float16. Fix: assert that the recorded residual at index l changes by exactly v and index l − 1 does not.

## Findings: patching

14. **Major. The localisation is about where any difference between the prompts reaches the final token, not where the answer is determined.** The models do not separate the pairs by gold (mean D +0.009, +0.023, −0.088, −0.087; fraction with D > 0 0.47–0.58; both-correct 0.00–0.03). Pooled recovery sign-normalises by each pair's own D; the final-token profile is identical for D > 0 and D < 0 pairs. Allowed: "the effect of a variable-letter change on the Yes−No logit moves to the final token at about relative depth 0.6". Not allowed: anything about causal judgement, or the title "where is the answer determined?". Fix: add label-preserving minimal pairs as the control.
15. **Major. The random-donor control is invalid as run.** Only 16 of 60 and 6 of 40 pairs have it; fig. 7 labels the control panels n = 60 and n = 40. For "hyp" pairs the donor's token at the differing position is identical to the clean token in 8 of 16 and 4 of 6 pairs. The metric has a non-zero expectation by regression to the mean; the observed values sit inside a simulated null. It shows no specificity.
16. **Minor.** "hyp" and "prem" are not independent replications (same clean prompts; correlation of D between kinds 0.80).
17. **Minor. Pair validity.** All pairs are token-aligned, labels are 1/0, "hyp" differs in exactly one token. The "by symmetry" label for "prem" holds in 99.3% of testable cases; 22 of 2,713 candidate "prem" pairs are automorphism cases (none among the pairs used). Fix: filter out automorphism swaps.
18. **Minor. Attribution validation near 1.0 is driven by large components.** Top-10 only: Pearson 0.997–0.999, sign agreement 100%. Random-10 without the one or two large components: Pearson 0.87, 0.76, 0.68, −0.67. Allowed: "accurate for the largest components". Not allowed: "the ranking is complete".
19. **Minor. 1.5B limits.** Stride 2 puts the jump between index 16 and 18; only 40 pairs from 28 premises, 20 for components and 6 for control.
20. **Nit.** Tables mix residual index 15 with block name L14H5 without saying they differ by one.

## Findings: cross-model and docstrings

21. **Minor.** Pairwise differences are on identical items with a valid paired cluster bootstrap (all six reproduced). But CLadder has 24 clusters; "selectivity" and "first layer within 1 pt" use all items including the selection half; three models from two families with non-monotone depth cannot show scaling.
22. **Unsupported statements in docstrings:** `07_steering.py` "is the probed direction causally used?" (one layer, one position); random directions as a control with "same norm, no meaning"; `08_patching.py` title and "random-donor control"; `09_cross_model.py` "does decodability scale".

## Verified and found correct

- Every point estimate and the spot-checked CIs in the steering JSONs, `steering.md`, the patching JSONs and tables, and `cross_model.json` reproduce exactly from the raw files.
- Column layout of the raw steering output, sign conventions and the toward/away construction.
- Evaluation items are all in the report half, no premise overlap with the selection half, 150/150 by gold; `data.selection_mask` reproduces the split exactly; identical to `05_confound.py`.
- Today's edits to `07_steering.py` do not change what the 1.5B runs compute.
- Layer indexing is consistent across cache, steering hook, patcher and recorder.
- By reading the code, the KV-cache shortcut is equivalent to a full forward pass for layers ≥ 1; steered runs show clean linear dose-responses; baseline correlates with the cache at 0.9997 or better.
- Patching sanity: (last layer, final) and (layer 0, differing tokens) reproduce the clean logit difference to 6e-6; the pooled-recovery bootstrap is clustered by premise.

## Could not check (needs a forward pass)

Numerical equality of the KV-cache path and the full path; why layer 24 is exempt in the test; the layer-0 hook path; whether the probe direction matters at other layers or positions; whether the 0.005 floor is float16 noise; the 0.5B result at the currently selected layer and C; Gemma.

## Wording the reviewer would allow (Corr2Cause only, after rerunning 0.5B)

> "At the one layer where the gold-label probe was selected, and at the final prompt token only, adding the probe's weight direction changes the Yes−No logit difference by +0.10 (Qwen-0.5B) and −0.37 (Qwen-1.5B) logits per unit of s. This lies within the range of three random unit directions (0.03–0.49 and 0.09–0.35), and is far smaller than for the difference-of-means direction (0.87, 1.99) or the direction of a probe for the model's own answer (1.91, 2.35). The steps used are 10 to 500 times the natural spread along the probe direction; a shift equal to the natural gap between valid and invalid items would move the logit difference by less than 0.01. In Qwen-1.5B the effect is consistent in sign and opposite to the gold label (accuracy 0.31 when pushed toward gold at α = 4), comparable to the strongest random direction. So later layers do not read the probe's weight direction at this site. This does not show that the decoded information is unused: the difference-of-means direction, which is the activation pattern of the same probe, does shift the answer, but it lies in the top principal components, separates the model's own answer at least as well as the gold label, and was not compared with a variance-matched control. On CLadder the gold directions carry little or no gold information, so no conclusion is drawn there."

Do not state any ablation comparison for the probe direction, "flips most answers" without the scale (α ≥ 2–4, about 10–30 natural class gaps), or any patching sentence about where the answer is determined.

# Third audit (independent reviewer): the debiased-pool claim and the steering wording

Received Fri 2026-11-13 14:45. A fresh reviewer agent got the code, result files, cached activations
and one claim to attack sentence by sentence. It could run CPU analyses but no model. Its report,
lightly reformatted, follows. My responses are in `notes/log.md`.

**Overall.** The 0.75 probe result is real and leak-free, but four of the seven sentences of the claim
do not survive as written.

Unless stated: accuracy on the report half (1,248 items, 624 premises), the author's folds for seeds
0/1/2, scores averaged over seeds, premise-cluster bootstrap.

## Findings

**1. Critical: the incremental-value test cannot see a 2–3 point gain, and a nested logistic stack finds one.**
- The nesting in `05b_probe_content.py` is leak-free.
- The test is insensitive: training rows get scores from probes fitted on 15 blocks, test rows from a probe fitted on 20, and gradient boosting overfits one noisy continuous column. Replication: +0.014 [−0.006, +0.034] seed-averaged, negative in every single seed.
- Power check with a synthetic column of known value: a column whose true gain (logistic stack) is +0.019 shows −0.008 in the GBM-plus-column test; a true +0.036 shows +0.015 [−0.002, +0.034].
- Properly nested logistic stack (inner out-of-fold parse and probe scores, two-parameter logistic regression, 3 seeds): report half +0.022 [+0.004, +0.037] accuracy, +0.017 [+0.010, +0.023] AUROC; all items +0.019 [+0.008, +0.029]. Qwen 0.5B: +0.010 [+0.001, +0.020].
- So withdrawing the non-nested number was correct procedure, but the nested estimate is the same as the withdrawn one; replacing it with "adds nothing measurable" was wrong.
- The parse is not the text ceiling: the parse stacked with a per-hypothesis-string bag of premise 4-grams reaches 0.880 (AUROC 0.954). The probe adds +0.001 [−0.006, +0.008] to that.
- Fix: report the nested logistic stack; say the probe adds about two points to this parse but nothing measurable to a stronger text-only model.

**2. Major: the listed "shallow confound controls" cannot score above 0.5 in this pool by construction, and the informative controls are missing.**
- Hypothesis prior and premise-only are forced to exactly 0.500. A linear whole-prompt bag-of-words is additive in premise and hypothesis features, so it is forced to about 0.5 as well. Any label signal here must be a hypothesis × premise interaction. Greedy trees on template or string id plus premise-level features also return exactly 0.500 (every root split has zero gain).
- Interaction baselines built by the reviewer:

| Baseline | Accuracy | Probe (0.752) minus this |
|---|---|---|
| Template × 5 premise-level counts, no binding | 0.589 | +0.163 [+0.129, +0.194] |
| Random-init probe | 0.579 (0.586–0.597 on 10 other fold assignments) | +0.174 |
| Template × premise bag-of-words, no binding | 0.659 [0.629, 0.687] (1–4-grams: 0.671) | +0.094 [+0.061, +0.125] |
| Template × per-letter statement counts | 0.712 | +0.041 [+0.007, +0.073] |
| Template × adjacency lookup | 0.728 | +0.024 [−0.008, +0.054] |
| Template × statements naming the pair | 0.747 (GBM: 0.752) | +0.006 [−0.026, +0.034] |
| Per-string bag of premise bigrams | 0.781 | −0.029 [−0.058, −0.002] |
| Hand parse | 0.816 | −0.063 |
| Per-string bag of premise 4-grams | 0.838 | −0.086 [−0.113, −0.062] |

- "Shallow" versus "structure-aware" is a continuum. The probe sits at the level of a lookup of the statements naming the pair, and below a parser-free per-hypothesis bag of bigrams.
- Random-init at 0.58 is plausible without a leak: it matches template × premise-level counts and correlates most with template × bag-of-words. It is a weaker control than template × bag-of-words.
- The author's own 11-cell table template × adjacent × common-neighbour predicts gold at 0.760.
- Fix: replace the control list with the interaction controls and state that single-input controls are 0.50 by design.

**3. Major: "the answers do not express this information" is false below the pooled level.**
- Pooled AUROC reproduces (0.510 [0.477, 0.540] on the report half).
- The hypothesis string alone explains 83% of the variance of the answer logit, and templates go in opposite directions (Qwen 1.5B, all items, within-string AUROC):

| Template | Answer AUROC for gold | Answer AUROC for pair adjacency | r(answer logit, probe score), string means removed |
|---|---|---|---|
| parent ("X directly causes Y") | 0.628 [0.581, 0.673] | 0.649 [0.604, 0.691] | +0.359 [+0.284, +0.427] |
| has_collider | 0.500 [0.445, 0.559] | 0.381 | +0.050 |
| has_confounder | 0.442 [0.371, 0.513] | 0.427 | −0.202 |
| non-child descendant | 0.430 [0.375, 0.485] | 0.469 | −0.127 |
| non-parent ancestor | 0.449 [0.393, 0.508] | 0.434 | −0.133 |

- The parent result replicates across halves (0.589 report, 0.593 selection). Overall correlation with the probe after removing string means: +0.137 [+0.072, +0.206].
- Qwen 0.5B: has_collider 0.586 [0.536, 0.636], parent 0.541 [0.502, 0.586], non-parent ancestor 0.445.
- Caveat: post hoc over 5 templates × 2 models; the 1.5B parent effect is about 5 standard errors.
- Fix: say the answers are at chance when pooled, while the answer logit does carry pair information, correctly signed for "directly causes" and wrongly signed elsewhere.

**4. Major: steering numbers reproduce, wording over-reaches.**
- All slopes, intervals and percentiles in `steering_controls_*.json` and the table match the raw outputs.
- Twenty controls are too few for "no more than": the smallest attainable p is 1/21 and effects up to 2–3 times the control median are not excluded. "Within the range of" is supportable.
- Natural spread along the direction is matched only on the debiased pool; on the original pool the gold direction is 2.7–3 times wider than the shuffled ones.
- In own-SD units the sentence fails for one case: for Qwen 1.5B on the original pool the gold probe exceeds all 20 shuffled controls (5.6 times the median). That number is in the JSON but not in the table or README.
- The step is far off-distribution (280–323 natural SDs). The direction itself is mostly fit noise on the debiased pool (split-half cosine 0.42–0.43). One layer, final token, 100–150 items; the probe is stronger at the end-of-hypothesis token, which was not steered.
- The slope is not zero: negative in all four cases, sign-consistent in 92–100% of items. "This direction is not read at this site" is too strong.
- Allowed wording: "Adding the gold-probe direction at the probe's layer and final token (a step of one RMS activation radius, about 300 natural SDs along that direction) shifts the Yes−No logit by an amount inside the range of 20 shuffled-label probe directions (65th percentile for Qwen 1.5B, 85th for 0.5B; 1.8 and 3.0 times the control median). With 20 controls and one site this neither shows nor excludes a causal role."

**5. Major: for Qwen 0.5B "much weaker" understates it.** At the final token the probe (0.633) is not above template × bag-of-words (0.659; −0.026 [−0.061, +0.005]) and barely above random-init (+0.029 [+0.002, +0.060], with one random initialisation and a shuffled-label null SD of 0.019). Positive evidence for 0.5B exists elsewhere: the end-of-hypothesis probe (0.737) and adjacency decodable at 0.89 against 0.66 for random-init.

**6. Minor: adjacency 0.98 reproduces, but "requires binding" needs qualifying.** Own code, held-out premises: 0.979 [0.971, 0.985]; random-init 0.62 at layer 22 and 0.65–0.66 at its own best layers. Text baselines for adjacency itself: per-letter statement counts with no pair lookup 0.946 [0.934, 0.958]; premise-level counts plus hypothesis string 0.70. So most of the 0.98 needs per-variable binding, not pair binding. At the end of the premise the trained model gives 0.53–0.55 (chance, as it should). "Pair correlated" is not a binding property here (random-init 0.88–0.91).

**7. Minor: shuffled controls are implemented correctly but are a single draw and are biased for rare properties.** Below-0.5 values are expected (over 10 shuffles: six-variables mean 0.461, pair-correlated 0.452, adjacency 0.498). The Qwen 0.5B gold shuffled control (0.526) is a lucky draw: over 16 permutations the null is mean 0.498, SD 0.019. Fix: at least 20 shuffles; report null mean and SD.

**8. Minor: near-duplicate properties.** "Pair correlated" is exactly 1 − "pair marginally independent"; "adjacent" and "has a separating set" differ on 8 of 2,400 items. Merge.

**9. Minor: the stratified statement is computed correctly but says little** (single properties predict gold at 0.51–0.55, so they could not carry the signal). Finer strata, probe AUROC (overall 0.829 for 1.5B): within quintiles of the parse score 0.725 [0.686, 0.762] (0.5B: 0.588); within 25 cells of parse score × per-string 4-gram score 0.599 [0.539, 0.657] (0.5B: 0.502). Most of the signal is shared with text-derived scores.

**10. Minor: headline details.** Accuracies and AUROCs are mixed in the claim. 0.752 is a 3-seed score ensemble; single probes give 0.736 / 0.741 / 0.754 and 0.738–0.748 on five other fold assignments. Trained on the selection half only, the probe scores 0.706 [0.681, 0.729] on the report half. Logistic regression stops at 300 iterations with warnings suppressed (converged fits move a single-seed score by under 0.01); C = 10 and 100 are worse than C = 1. Two slightly different intervals are printed for the same number (two scripts, two bootstrap draws).

**11. Nit: index parity equals the label for all 2,400 items** (sort order in `data.c2c_debiased_pool`). Nothing uses the index, but it is a hazard for any future index-based subsampling.

**12. Minor: documents.** Tables are current (regenerating gives byte-identical files). README.md shows one broken value marker. `12_tables.py` silently drops rows when a file is missing; `13_build_docs.py` inlines tables without checking they are newer than the results. The bag-of-words figure differs between two tables (0.518 at C = 100 on the report half; 0.495 at C = 1 on all items). README over-reach: "carries information about the answer that it does not use".

## Verified and found correct

- Pool balance: every hypothesis string and every premise is exactly half valid in all 15 training and 15 test folds (seeds 0, 1, 2) and in both halves of the selection/report split; no premise on both sides of a fold; every item tested once. 1,200 premises, 48 strings, 25 blocks.
- Headline reproduction with own folds, standardisation and logistic regression: 0.752 [0.730, 0.775]; the cached random-init scores reproduce exactly.
- Bootstrap unit: premise clusters, block clusters and iid give the same interval.
- Selection: no contamination found; the random-init probe gets its own layer and C by the same procedure.
- An independent parser agrees with `structural_features` on all 2,400 items for the four main properties.
- Steering: layer and C agree across the steering, controls, analysis and confound files; baseline logits match the cache to within 0.05.

## Could not check

Anything needing the model (the control runs themselves, token positions, how the random-init copy was made, steering at other sites); the pool's construction from the raw train file (label fidelity and disjointness from test were not retested); script 14; Gemma and Qwen 3B.

## The claim in the wording the reviewer would allow

"On a Corr2Cause pool built so that the hypothesis alone and the premise alone are exactly uninformative (every single-input control is 0.50 by construction, so any label signal is a hypothesis × premise interaction), a linear probe on Qwen2.5-1.5B's final-token activations at layer 22 predicts the gold label with accuracy 0.75 [0.73, 0.78] (AUROC 0.83). This is above interaction controls that bind nothing to the queried variables (template × premise bag-of-words 0.66; a probe on a randomly initialised copy 0.58–0.60, about what template × premise-level counts gives). It is about equal to a per-template lookup of the statements naming the queried pair (0.75), and below a per-hypothesis bag of premise n-grams (0.78–0.84) and a hand-written parse (0.82). In nested logistic stacking the probe adds about two points to the hand-written parse (+0.022 [+0.004, +0.037]; AUROC +0.017 [+0.010, +0.023]) but nothing measurable to a stronger text-only model (0.88; +0.001 [−0.006, +0.008]). Whether the queried pair is adjacent in the skeleton is linearly decodable at AUROC 0.98 on held-out premises, against 0.62–0.66 from the random-init copy, 0.70 from text without binding, 0.95 from per-variable statement counts, and 0.55 at the end of the premise before the hypothesis is read. The probe's signal is not carried by any single parsed property, but most of it is shared with text-derived scores (AUROC 0.83 overall, 0.72 within quintiles of the parse score, 0.60 within 25 cells of two text scores). Pooled over hypothesis types the answers are at chance (AUROC 0.51), but not uniformly: for 'X directly causes Y' the answer logit tracks validity (within-string AUROC 0.63 [0.58, 0.67]) and the probe score (r = 0.36), while for three other hypothesis types it is weakly anti-correlated (0.43–0.45). Adding the probe direction at layer 22, final token (a step of about 280 natural SDs), shifts the answer logit by an amount within the range of 20 shuffled-label probe directions (65th percentile; 85th for Qwen 0.5B); this single-site test neither shows nor excludes a causal role. For Qwen2.5-0.5B the final-token probe (0.63) is not clearly above the random-init copy (0.60) and not above the template × bag-of-words control (0.66); adjacency is decodable at 0.89 (random-init 0.66)."

# Fourteenth review: the 7B probe's gain over the text-only stack (raw report)

## Summary (A-F)

**A. Reproduce.** Exact, with independent code: probe 0.7837 [0.762, 0.806] (layer 19, C 0.1; stored out-of-fold scores equal my refit, difference 0.0); over hand parse +4.57 pp [+2.56, +6.49]; over strongest text +1.44 pp [+0.40, +2.56] (33 items fixed, 15 broken, of 1,248); AUROC +0.0052 [+0.0025, +0.0081]. Nesting is sound: premises never split, block folds, stacking weights from inner out-of-fold scores, bootstrap over premises. Caveats: (layer, C) is a near tie (0.7549 vs 0.7529); the text model has no tuned hyperparameters.

**B. Multiplicity.** 40 stacking-increment intervals in result files (25 exclude zero), 5 for the announced statistic; 44 probe-minus-best-text intervals on all pools, all negative. p = 0.015 (premise sign-flip). Bonferroni 0.077 (m=5), 0.15 (m=10), 0.31 (m=20): does not survive. Continuation: 0.5B +0.24, 1.5B -0.08, Gemma +0.88 [-0.08, +1.84], 3B +0.88 [0.00, +1.84], 7B +1.44 [+0.32, +2.56]; 7B minus 3B +0.56 [-0.48, +1.60], p = 0.33.

**C. Sensitivity.** Stable over layers 19-26 with C >= 0.1 (+0.8 to +1.7 pp); C = 0.01 gives +0.48 [-0.48, +1.52]. Selection half is smaller (+0.87; mean about +0.5); all items +1.17 [+0.33, +2.00]. content_end: +2.24 [+1.04, +3.45], p = 0.0005. All outer fits hit max_iter=300 (warnings are suppressed in code); converged fits give the same +1.44.

**D. Held-out.** Fit selection, score report: +0.80 [-0.32, +1.92]; reverse +0.52 [-0.78, +1.74]; with in-half layer choice +0.08 / +1.30. Average over 10 random block halves x 2: +0.55 [+0.21, +0.90]. Survives in sign, about half a point.

**E. Content.** Gain sits on adjacent-pair items across all templates; the probe is right on only 54 % of the text model's errors. Not the answer logit (+1.44 unchanged), not stacking (shuffled-label null within ±0.16 pp), not bfloat16/quantisation (rounding noise ±0.1 pp). But the text model is under-tuned: n-gram C 1 to 100 (chosen on the selection half) lifts text-only 0.865 to 0.887; over it the probe adds +0.80 [-0.08, +1.68] report, -0.43 selection, +0.21 [-0.42, +0.87] all; over a forward-selected text stack -0.08. A second text component adds as much as the probe (+0.9 to +2.4). No random-init 7B control exists (smaller models' random-init probes add 0.0).

**F. Verdict.** Sayable: "the 7B probe adds about one point (1.4 report, 0.9 selection, 1.2 all; p = 0.015 uncorrected) to the repository's fixed parse + 4-gram stack; robust to layer and convergence, not to multiplicity correction or to tuning the text model". Not sayable: evidence beyond text, or a change at 7B. Best next step: pre-specified replication on a disjoint pool with a tuned text model and a random-init 7B pass.

**The report file was not written.** The harness refused the write of `report.md` (subagents may not write report files), and I did not route around it. The full report follows; save it to `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit14/report.md` if you want the file. Scripts, logs and JSON outputs are in `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit14/`.

---

# Audit of one cell: qwen7b-4bit x c2c_debiased, increment of the gold probe over the strongest text-only model

Date: 14 Nov 2026. Independent recomputation from `cache/qwen7b-4bit/c2c_debiased/{resid.npy, meta.json, probe_oof_C*.npz}`, the result JSON files and the code in `scripts/04_probes.py`, `scripts/05_confound.py`, `scripts/05b_probe_content.py`, `src/causal_probe/{probes,data,stats}.py`. Not read: RESULTS.md, README.md, report/, EXPLAINER.md, notes/progress.md, the 14 Nov entries of notes/log.md. All gains are in percentage points of accuracy (pp) unless marked AUROC; intervals are 95 % percentile cluster-bootstrap intervals over premises unless stated.

Claim under audit: probe accuracy 0.784 [0.762, 0.806] (layer 19, final token, C = 0.1); stacking gain over the strongest text-only model +1.4 pp [+0.4, +2.6], AUROC +0.005 [+0.002, +0.008]; over the hand parse +4.6 pp [+2.6, +6.5].

## Verdict in brief

The numbers reproduce to the last digit and the stacking is properly nested. The increment is still not a finding about what the 7B model represents beyond the text.

1. It is 18 net items out of 1,248 (33 fixed, 15 broken), p = 0.015. That does not survive a family-wise correction over even the five model cells (Bonferroni 0.077), and it is indistinguishable from the 3B and Gemma increments (+0.9 pp each). It is a continuation, not a qualitative change.
2. The "strongest text-only model" is not tuned. Changing one hyperparameter of its n-gram component (C = 1 to C = 100, chosen on the selection half) lifts the text-only model from 0.865 to 0.887 on the report half, more than the probe adds. Over that model the 7B probe adds +0.8 pp [-0.1, +1.7] on the report half, -0.4 pp on the selection half, +0.2 pp [-0.4, +0.9] on all items. A second text component added to the repository's text model gives the same +1 to +2.4 pp as the probe.
3. What does hold: relative to the repository's fixed text model the increment is stable over layers 19-26 and C >= 0.1, and unaffected by convergence, rounding-scale noise and the model's answer logit. It is absent for shuffled-label probes and for random-init probes of the four smaller models, and larger at the end-of-hypothesis position (+2.2 pp, p = 0.0005). A residual of a few tenths of a point over tuned text models cannot be excluded.

## Numbered findings

| # | Severity | Finding |
|---|---|---|
| 1 | none (confirmation) | All three reported numbers reproduce exactly from the cached activations with independent code. Stored out-of-fold probe scores equal my refit to 0.0 absolute difference. |
| 2 | none (confirmation) | Nesting is sound: premises (2 items each) never split, folds and the selection/report split are unions of blocks, stacking weights are fitted on inner out-of-fold scores, vectorisers and scalers are fitted on training rows only, the bootstrap resamples premises. |
| 3 | minor | The "selection half" is a partition of evaluation items, not of training data. (Layer, C) is chosen from out-of-fold scores of models that were also trained on report-half labels, and report-half scores come from models trained on selection-half labels. This is not leakage into the report-half test; the strict version is section D. |
| 4 | minor | The choice of (layer 19, C 0.1) is a near tie: selection-half accuracy 0.7549 against 0.7529 for (layer 26, C 1), about two items. Selecting inside one half only gives layer 28/C 1 (selection half) or layer 19/C 1 (report half). The gain does not depend on this choice for C >= 0.1 (section C). |
| 5 | minor | `probes._fit` runs under `warnings.simplefilter("ignore")`, so no ConvergenceWarning can appear in `logs/q7b_*.log` (there are none). All 15 outer 7B fits at layer 19 stop at `max_iter=300` for C = 0.1 and C = 1. Converged fits (float64, tol 1e-6, 470-1,130 iterations) give the same gain (+1.44 and +1.20 pp). |
| 6 | major | Multiplicity. p = 0.015 (sign-flip over premises). Family of 5 (accuracy over strongest text, one per model): Bonferroni/Holm 0.077. Family of 10 (plus AUROC): 0.15/0.09. Family of 20 (plus hand-parse base): 0.31/0.12. It is "first" only for the report-half accuracy statistic: the all-items version already excluded zero for gemma2b (+0.71 [+0.04, +1.42]) and the AUROC version for qwen1.5b, gemma2b and qwen3b. |
| 7 | major | Not a qualitative change. Increments over the same text model, report half: 0.5B +0.24, 1.5B -0.08, Gemma-2B +0.88, 3B +0.88, 7B +1.44 pp; AUROC +0.0001, +0.0027, +0.0032, +0.0045, +0.0052. Paired 7B minus 3B: +0.56 pp [-0.48, +1.60], p = 0.33; 7B minus Gemma +0.56 [-0.56, +1.60]. The increments track probe accuracy (0.633, 0.752, 0.763, 0.765, 0.784). |
| 8 | moderate | The report half is the favourable half. Same statistic on the selection half: +0.87 pp [-0.35, +2.08], p = 0.21, and that is near the top of the selection-half gains over the 45 final-token configurations I ran (range -0.35 to +0.95). Mean over layers >= 19 at C = 0.1: report +1.21, selection +0.48. All items: +1.17 pp [+0.33, +2.00], p = 0.007. The halves do not differ significantly; +1.4 is the upper end of the estimates. |
| 9 | moderate | Strict held-out halves: fit selection, score report +0.80 pp [-0.32, +1.92]; reverse +0.52 pp [-0.78, +1.74]; with (layer, C) also chosen inside the training half +0.08 and +1.30. Averaged over 10 random block halves x 2 directions: +0.55 pp [+0.21, +0.90] (fixed layer 19) or +0.47 [+0.14, +0.82] (layer chosen in the training half). Sign survives; size is about half a point with half the training data. |
| 10 | critical | The comparison model is under-tuned, so "over the strongest text-only model" overstates what was shown. Per-hypothesis 4-gram logistic regression alone: 0.805 at the repository's C = 1, 0.838 at C = 10, 0.857 at C = 100 (selection half 0.813, 0.832, 0.845). Stack parse + 4-grams on the report half: 0.865, 0.881, 0.887; selection-half accuracy picks C = 100. 7B probe over the tuned model: +0.80 pp [-0.08, +1.68] report, -0.43 selection, +0.21 [-0.42, +0.87] all. Over a text model built by forward selection of components on the selection half (0.895): -0.08 pp [-0.96, +0.72]. |
| 11 | major | Placebo: one more text component added to the repository's text model gains as much as the probe. Same n-grams at C = 10: +2.32 pp [+1.04, +3.69]; per-template boosting on parse + per-variable counts + lengths: +1.44 [+0.32, +2.64]; 6-grams +1.04; 5-grams +0.88; bigrams +0.96. |
| 12 | none (rules out) | Not the model's answer: the answer logit has AUROC 0.549 for the label within hypothesis string, adds +0.2 to +0.4 pp (n.s.) to the text model, and the probe's gain over text + per-string answer logit is unchanged (+1.44 [+0.40, +2.56]). Not a stacking artefact: probes trained on labels swapped within premises give gains between -0.16 and +0.16 pp (8 draws). |
| 13 | gap | No random-init control exists for this model. For the four smaller models the random-init probe adds +0.00, +0.08, -0.16, +0.00 pp over the same text model. This is unverified for 7B. Also missing: an unquantised 7B run and a second item pool. |
| 14 | none (rules out) | Activations at layer 19, final token: no NaN/inf, 2,400 distinct rows, no near-duplicates across premises (max cosine 0.993), max abs value 36.75, row norm and token count uninformative (AUROC 0.502, 0.501). Every value has its three low mantissa bits zero (bfloat16 stored as float16, as expected). Relative noise of one bfloat16 quantum moves the gain to +1.36 / +1.52 pp. Nothing suggests an artefact. |
| 15 | note | The end-of-hypothesis position gives a larger, statistically safer increment over the repository's text model: +2.24 pp [+1.04, +3.45], p = 0.0005 (44 fixed, 16 broken; positive in 11 of 13 report blocks, negative in none; all 27 layer x C configurations positive). It shrinks over tuned text: +0.64 pp [-0.40, +1.68] report, +0.46 [-0.29, +1.17] all. In half-data training a residual of +0.4 to +0.6 pp remains even over a richer text model (uncorrected p 0.02-0.04). At this position qwen0.5b already has +0.96 pp [+0.08, +1.92] over the repository's text model, which argues against reading the 7B cell as a threshold being crossed. |

## A. Reproduction

Script: `a_reproduce.py`. It uses its own folds, components, nesting, stacking and bootstrap, and imports from the repository only the feature parser `probes.structural_features` and `stats.bootstrap_ci` (to compare intervals draw for draw).

Structure verified:
- 2,400 items, 1,200 distinct premises, exactly 2 items per premise (one valid, one invalid), every premise inside one block.
- 25 blocks; every hypothesis string exactly once valid and once invalid per block.
- Selection half = 12 blocks (1,152 items); report half = 13 blocks (1,248 items, 624 premises); no premise in both.
- My block folds are identical to `data.grouped_folds` for seeds 0, 1, 2. No premise or block is on both sides of any outer fold; inner folds (4, by block, seed 100 + fold) likewise.

| Quantity | Result file | My recomputation |
|---|---|---|
| (layer, C) chosen on the selection half | 19, 0.1 | 19, 0.1 (0.7549; runner-up layer 26, C 1: 0.7529) |
| Probe accuracy, report half | 0.7837 [0.7620, 0.8061] | 0.7837 [0.7620, 0.8061] with the repository's draws; [0.7612, 0.8061] with my own |
| Stored out-of-fold scores vs refit from `resid.npy` | - | identical (max abs difference 0.0, 0 sign disagreements) |
| Gain over hand parse, accuracy | +4.57 [+2.56, +6.49] | +4.57 [+2.56, +6.49]; own 10,000 draws [+2.80, +6.33] |
| Gain over hand parse, AUROC | +0.0240 [+0.0162, +0.0321] | +0.0240 [+0.0162, +0.0321] |
| Strongest text-only accuracy / with probe | 0.8654 / 0.8798 | 0.8654 / 0.8798 |
| Gain over strongest text-only, accuracy | +1.44 [+0.40, +2.56] | +1.44 [+0.40, +2.56]; own 10,000 draws [+0.32, +2.56] |
| Gain over strongest text-only, AUROC | +0.0052 [+0.0025, +0.0081] | +0.0052 [+0.0025, +0.0081] |
| Same, all 2,400 items, accuracy | +1.17 [+0.33, +1.96] | +1.17 [+0.33, +2.00] |

Detail of the cell:
- The probe fixes 33 items and breaks 15 (net 18 of 1,248; 47 premises with a non-zero difference).
- Per report block (13): +1.0, +3.1, -1.0, +3.1, +4.2, +4.2, +1.0, -1.0, 0, -1.0, +2.1, +3.1, 0 pp (8 positive, 3 negative). Bootstrap over the 13 blocks instead of premises: [+0.40, +2.48].
- Single fold-seeds, no score averaging: +1.28, +0.72, +1.28 pp on the report half; +0.61, -0.52, -0.69 pp on the selection half.
- Mean stacking weights (standardised scores): parse 1.70, 4-grams 1.28, probe 0.64.

Answers to the nesting questions:
- **Layer and C:** report-half labels enter only as in finding 3 (training labels of the models whose selection-half scores are compared; never test labels at selection).
- **Stacking weights:** fitted for each outer fold only on inner out-of-fold scores of that fold's training rows.
- **Text model hyperparameters:** none are selected. N-gram range (1, 4), C = 1 and 200 boosting iterations are constants in 05b. That removes leakage but is the weakness of finding 10.
- **Premises:** never split; the two items of a premise are always in the same fold.
- **Bootstrap:** over premises (`groups[rep]`, 624 clusters, 1,000 draws, percentile). It conditions on the fitted models, as any cross-validated interval does.

## B. Multiplicity

Scripts: `b_family.py`, `b_models.py`, `b2_content_end.py`.

Stacking increments exist only in `results/probe_content_<model>_c2c_debiased.json`, i.e. for one pool and one token position (final):

| Statistic in `incremental_value` | cells | intervals excluding zero (models) |
|---|---|---|
| over_hand_parse / gain_accuracy | 5 | 4 (1.5B, Gemma, 3B, 7B) |
| over_hand_parse / gain_auroc | 5 | 4 (same) |
| over_hand_parse / gain_accuracy_all_items | 5 | 4 (same) |
| over_strongest_text_only / gain_accuracy | 5 | 1 (7B) |
| over_strongest_text_only / gain_auroc | 5 | 4 (1.5B, Gemma, 3B, 7B) |
| over_strongest_text_only / gain_accuracy_all_items | 5 | 2 (Gemma, 7B) |
| gbm_plus_one_column / gain_accuracy, gain_auroc | 10 | 6 (Gemma, 3B, 7B, each twice) |
| Total stacking increments in result files | 40 | 25 |

For the other pools, and in all `confound_*` files, the comparison is the direct difference probe minus best single text baseline (always `structural`). That is 22 files x 2 metrics = 44 intervals (c2c_debiased 5 + 1 fixed-C, c2c_probe 4 + 4 fixed-C, cladder 4 + 4 fixed-C). All 44 are negative with intervals excluding zero, so none could have produced a positive "first". No increment at any other token position is in the result files.

The family is therefore 5 cells in the narrowest reading (the announced statistic), 15 for "any increment over the strongest text-only model", 20 for the core table (2 bases x 2 metrics x 5 models), and 40 for everything in the files.

**p-value of the 7B increment.** I use a sign-flip permutation test on per-premise differences in the number of correct items (200,000 flips). Under the null that adding the probe is as likely to break an item as to fix one, the sign of each premise's difference is exchangeable. The test respects premise clustering and does not depend on a percentile interval of a statistic that moves in steps of 1/1248. p = 0.0153. Cross-checks: exact McNemar on items 0.0133; premise bootstrap 0.012; exact sign-flip over the 13 report blocks 0.034; t-test over blocks 0.022.

| Family (my recomputation, report half) | m | Bonferroni | Holm | Benjamini-Hochberg q |
|---|---|---|---|---|
| accuracy over strongest text, 5 models | 5 | 0.077 | 0.077 | 0.077 |
| + AUROC over strongest text | 10 | 0.153 | 0.092 | 0.031 |
| accuracy over both bases | 10 | 0.153 | 0.092 | 0.031 |
| both bases x both metrics | 20 | 0.307 | 0.123 | 0.024 |

It does not survive a family-wise correction in any family of five or more. It passes a false-discovery-rate criterion only in families that contain the hand-parse and AUROC cells, which says that small positive increments are common in this table, not that this one is special. The all-items version (p = 0.0068) would pass Bonferroni over 5 at 0.034, but all items include the half on which (layer, C) was chosen. The AUROC increment of 7B (z = 3.6, p about 0.0004) survives Bonferroni over 20, as does 3B's (z = 4.1).

**Trend in size**, same statistic for all five models (each model's (layer, C) chosen on the selection half by the repository's rule; text model identical):

| model | params (B) | layer, C | probe acc | accuracy gain over strongest text, report | p | selection half | all items | AUROC gain, report (x100) | accuracy gain over hand parse | random-init probe over strongest text |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen0.5b | 0.49 | 14, 1 | 0.633 | +0.24 [-0.32, +0.80] | 0.58 | -0.35 [-0.87, +0.09] | -0.04 [-0.42, +0.33] | +0.01 [-0.05, +0.06] | +0.88 [-0.08, +1.92] | +0.00 [-0.32, +0.32] |
| qwen1.5b | 1.54 | 22, 1 | 0.752 | -0.08 [-0.88, +0.80] | 1.00 | -0.17 [-1.22, +0.87] | -0.12 [-0.79, +0.54] | +0.27 [+0.07, +0.49] | +2.16 [+0.56, +3.77] | +0.08 [-0.32, +0.48] |
| gemma2b | 2.61 | 17, 1 | 0.763 | +0.88 [-0.08, +1.84] | 0.12 | +0.52 [-0.35, +1.48] | +0.71 [+0.04, +1.42] | +0.32 [+0.12, +0.54] | +3.45 [+2.08, +4.89] | -0.16 [-0.56, +0.24] |
| qwen3b | 3.09 | 25, 1 | 0.765 | +0.88 [+0.00, +1.84] | 0.09 | +0.00 [-1.04, +1.04] | +0.46 [-0.21, +1.17] | +0.45 [+0.24, +0.67] | +3.29 [+1.76, +4.89] | +0.00 [-0.56, +0.56] |
| qwen7b-4bit | 7.62 | 19, 0.1 | 0.784 | +1.44 [+0.32, +2.56] | 0.015 | +0.87 [-0.35, +2.08] | +1.17 [+0.33, +2.00] | +0.52 [+0.24, +0.81] | +4.57 [+2.80, +6.33] | no cache |

These agree with the result files to the rounding of the bootstrap. Paired differences of the increments on the same items, 7B minus other, report half: vs 0.5B +1.20 [+0.24, +2.24]; vs 1.5B +1.52 [+0.48, +2.56]; vs Gemma +0.56 [-0.56, +1.60], p = 0.35; vs 3B +0.56 [-0.48, +1.60], p = 0.33. The 7B accuracy result is the next point on a rising line whose interval happens to clear zero.

Note also that random-init probes add about +0.9 pp over the hand parse (0.5B and 1.5B: +0.96 [+0.16, +1.84]). "Over the hand parse" is therefore weak evidence of anything.

**End-of-hypothesis position** ("content_end"; caches of 0.5B, 1.5B, 7B only; my computation, not in the result files):

| model | layer, C | probe acc | accuracy gain over strongest text, report | p | selection half | all items | AUROC gain (x100) |
|---|---|---|---|---|---|---|---|
| qwen0.5b | 14, 1 | 0.737 | +0.96 [+0.08, +1.92] | 0.058 | +0.61 [-0.43, +1.65] | +0.79 [+0.12, +1.46] | +0.28 [+0.05, +0.52] |
| qwen1.5b | 19, 1 | 0.788 | +1.04 [+0.00, +2.08] | 0.072 | +0.00 [-1.22, +1.22] | +0.54 [-0.29, +1.37] | +0.56 [+0.25, +0.89] |
| qwen7b-4bit | 19, 0.1 | 0.812 | +2.24 [+1.04, +3.45] | 0.0005 | +1.30 [+0.09, +2.60] | +1.79 [+0.92, +2.67] | +0.77 [+0.38, +1.16] |

The 7B increment at this position (p = 0.0005 report, 0.0001 all items) would survive Bonferroni over 40. It is still an increment over the untuned text model (section E).

## C. Sensitivity to regularisation, layer, position, convergence

Script: `c_sensitivity.py` (72 configurations x 75 nested fits, then converged refits). Cell content: accuracy gain over the repository's strongest text model on the report half [interval]; (same on the selection half); probe accuracy on the report half.

**Final token:**

| layer | C = 0.01 | C = 0.1 | C = 1 |
|---|---|---|---|
| 14 | -0.08 [-0.48, +0.32] (+0.00); 0.606 | -0.24 [-0.64, +0.16] (+0.35); 0.619 | -0.08 [-0.64, +0.48] (+0.35); 0.623 |
| 15 | -0.08 [-0.56, +0.40] (-0.17); 0.600 | -0.32 [-0.88, +0.16] (+0.26); 0.608 | -0.24 [-0.80, +0.24] (-0.09); 0.614 |
| 16 | -0.32 [-0.72, +0.00] (-0.26); 0.635 | -0.16 [-0.80, +0.40] (-0.26); 0.642 | -0.16 [-0.80, +0.48] (-0.09); 0.644 |
| 17 | +0.48 [-0.24, +1.20] (-0.35); 0.683 | +0.56 [-0.16, +1.36] (+0.00); 0.696 | +0.40 [-0.40, +1.20] (-0.09); 0.690 |
| 18 | +0.48 [-0.40, +1.44] (+0.09); 0.729 | +1.04 [+0.08, +2.08] (+0.26); 0.733 | +0.88 [-0.16, +1.92] (+0.35); 0.727 |
| **19** | +0.48 [-0.48, +1.52] (+0.26); 0.749 | **+1.44 [+0.32, +2.56] (+0.87); 0.784** | +1.12 [+0.08, +2.24] (+0.26); 0.778 |
| 20 | +0.40 [-0.56, +1.28] (+0.17); 0.740 | +1.52 [+0.64, +2.48] (+0.17); 0.764 | +1.44 [+0.48, +2.48] (+0.26); 0.761 |
| 21 | +0.56 [-0.24, +1.44] (+0.35); 0.740 | +1.20 [+0.24, +2.16] (-0.09); 0.763 | +1.28 [+0.32, +2.32] (+0.00); 0.760 |
| 22 | +0.64 [-0.16, +1.44] (+0.61); 0.736 | +1.36 [+0.48, +2.32] (+0.17); 0.766 | +1.68 [+0.72, +2.72] (-0.35); 0.760 |
| 23 | +0.80 [+0.08, +1.60] (+0.52); 0.728 | +1.36 [+0.48, +2.32] (+0.17); 0.756 | +1.36 [+0.48, +2.32] (+0.35); 0.760 |
| 24 | +0.80 [+0.00, +1.68] (+0.52); 0.710 | +0.96 [+0.00, +2.00] (+0.35); 0.759 | +1.20 [+0.24, +2.24] (+0.26); 0.760 |
| 25 | +0.72 [-0.16, +1.60] (+0.43); 0.728 | +1.20 [+0.24, +2.16] (+0.78); 0.750 | +1.36 [+0.40, +2.40] (+0.52); 0.757 |
| 26 | +0.88 [+0.00, +1.84] (+0.43); 0.720 | +1.44 [+0.48, +2.48] (+0.95); 0.760 | +1.28 [+0.24, +2.40] (+0.61); 0.757 |
| 27 | +0.88 [+0.00, +1.76] (+0.26); 0.718 | +0.80 [-0.16, +1.84] (+0.52); 0.749 | +0.96 [-0.08, +2.08] (+0.26); 0.764 |
| 28 | +0.72 [-0.16, +1.60] (+0.52); 0.718 | +0.80 [-0.16, +1.84] (+0.87); 0.742 | +1.12 [+0.16, +2.16] (+0.87); 0.738 |

Layers 19-28, by C:
- Report-half mean: +0.69 (C = 0.01; interval excludes zero in 1 of 10), +1.21 (C = 0.1; 7 of 10), +1.28 (C = 1; 9 of 10).
- Selection-half mean: +0.41, +0.48, +0.30 (range -0.35 to +0.95).
- All-items mean: +0.55, +0.86, +0.81.

What selection gives:
- **Repository's rule** (best selection-half probe accuracy over layer x C): (19, 0.1), +1.44 [+0.32, +2.56].
- **C fixed at 0.01** (the original pre-registered value, `probes.PROBE_C`): layer 19, +0.48 [-0.48, +1.52], not significant.
- **C fixed at 0.1:** layer 19, +1.44.
- **C fixed at 1:** layer 26, +1.28 [+0.24, +2.40].
- **Layer fixed** at any of 20-23, 25 or 26 with C >= 0.1: the report-half interval excludes zero.

So the gain does not hang on layer 19. It does require C >= 0.1, and it is roughly half as large on the selection half throughout.

**End-of-hypothesis position:**

| layer | C = 0.01 | C = 0.1 | C = 1 |
|---|---|---|---|
| 16 | +0.88 [-0.24, +2.08] (+0.00); 0.786 | +1.36 [+0.24, +2.64] (+0.43); 0.813 | +1.28 [+0.08, +2.56] (+0.95); 0.815 |
| 17 | +1.52 [+0.32, +2.72] (+0.17); 0.794 | +1.52 [+0.40, +2.72] (+0.17); 0.815 | +1.60 [+0.40, +2.81] (+0.09); 0.818 |
| 18 | +1.36 [+0.32, +2.48] (+0.35); 0.792 | +1.04 [-0.16, +2.24] (+0.78); 0.814 | +1.60 [+0.32, +2.88] (+0.87); 0.819 |
| **19** | +1.20 [+0.16, +2.32] (+0.87); 0.788 | **+2.24 [+1.04, +3.53] (+1.30); 0.812** | +1.92 [+0.64, +3.21] (+1.39); 0.813 |
| 20 | +1.60 [+0.56, +2.72] (+0.69); 0.800 | +1.84 [+0.72, +3.04] (+1.13); 0.821 | +1.84 [+0.72, +3.04] (+1.13); 0.816 |
| 21 | +1.68 [+0.64, +2.80] (+0.69); 0.798 | +2.24 [+1.12, +3.45] (+1.13); 0.816 | +2.16 [+1.04, +3.37] (+1.13); 0.812 |
| 22 | +1.76 [+0.72, +2.88] (+0.09); 0.776 | +2.00 [+0.88, +3.12] (+0.17); 0.801 | +2.08 [+0.96, +3.29] (+0.26); 0.796 |
| 24 | +0.96 [-0.16, +2.08] (+0.09); 0.783 | +1.84 [+0.72, +3.04] (+0.35); 0.808 | +2.08 [+0.96, +3.21] (+0.78); 0.800 |
| 26 | +1.12 [+0.08, +2.24] (+0.35); 0.765 | +1.84 [+0.80, +3.04] (+0.26); 0.791 | +1.60 [+0.48, +2.80] (+0.26); 0.790 |

The repository's rule selects (19, 0.1) here as well. All 27 configurations are positive on the report half and non-negative on the selection half; 24 of 27 report-half intervals exclude zero. Selection-half mean: +0.37, +0.64, +0.76 for the three C values.

**Convergence** (layer 19):

| fit | lbfgs iterations (min, median, max over 75 fits) | probe acc | gain report | gain selection | gain all |
|---|---|---|---|---|---|
| final, C 0.1, max_iter 300 (repository) | 249 to 300 (all 15 outer fits at 300) | 0.784 | +1.44 [+0.32, +2.56] | +0.87 | +1.17 |
| final, C 0.1, max_iter 1000, default tol | 249, 274, 332 | 0.784 | +1.44 [+0.32, +2.56] | +0.87 | +1.17 |
| final, C 0.1, float64, tol 1e-6, max_iter 20000 | 471, 506, 602 | 0.782 | +1.44 [+0.32, +2.56] | +0.78 | +1.12 |
| final, C 0.1, max_iter 100 | 100, 100, 100 | 0.777 | +1.28 [+0.24, +2.40] | +0.69 | +1.00 |
| final, C 0.01, repository / converged | 112-145 / 193-251 | 0.749 / 0.749 | +0.48 / +0.48 | +0.26 / +0.26 | +0.37 / +0.37 |
| final, C 1, repository (all fits at 300) | 300, 300, 300 | 0.778 | +1.12 [+0.08, +2.24] | +0.26 | +0.71 |
| final, C 1, float64, tol 1e-6 | 822, 889, 1133 | 0.780 | +1.20 [+0.16, +2.24] | +0.09 | +0.67 |
| content_end, C 0.1, repository / converged | 268-300 / 509-623 | 0.812 / 0.813 | +2.24 / +2.24 | +1.30 / +1.30 | +1.79 / +1.79 |

The iteration limit is reached but immaterial.

## D. Held-out premises

Scripts: `d_heldout.py`, `d2_heldout_tuned.py`, `d3_heldout_rich.py`.

Design, on the training half only:
- Choose (layer, C) by 3-fold by-block cross-validation over layers x {0.01, 0.1, 1}.
- Produce inner 4-fold out-of-fold scores of parse, per-string 4-grams and probe; fit the two stacks.
- Refit the components on the whole training half and score the other half once.

"Fixed" uses (19, 0.1) instead of the in-half choice. With half the data every component is weaker (probe about 0.72-0.75, text model about 0.83-0.84). This is a harder regime for both, and particularly for the per-string n-gram model (12 or 13 premises per string and label).

**Final token, repository's text model** (parse + 4-grams at C = 1):

| split | probe (layer, C) | probe acc | text acc | text + probe | accuracy gain | p | AUROC gain (x100) | gain over hand parse |
|---|---|---|---|---|---|---|---|---|
| fit selection half, score report half | fixed 19, 0.1 | 0.746 | 0.831 | 0.839 | +0.80 [-0.32, +1.92] | 0.20 | +0.65 [+0.33, +0.96] | +1.76 [+0.24, +3.29] |
| fit selection half, score report half | chosen in half: 28, 1 | 0.696 | 0.831 | 0.832 | +0.08 [-1.20, +1.44] | 1.00 | +0.28 [-0.18, +0.70] | +2.88 [+1.12, +4.65] |
| fit report half, score selection half | fixed 19, 0.1 | 0.719 | 0.837 | 0.842 | +0.52 [-0.78, +1.74] | 0.50 | +0.37 [+0.00, +0.73] | +2.08 [+0.35, +3.73] |
| fit report half, score selection half | chosen in half: 19, 1 | 0.717 | 0.837 | 0.850 | +1.30 [+0.00, +2.60] | 0.07 | +0.46 [+0.09, +0.85] | +2.34 [+0.61, +4.08] |
| 10 random block halves x 2 directions, fixed 19, 0.1 | - | 0.736 | 0.836 | 0.841 | mean +0.55 (range -0.24 to +1.12; 19 of 20 positive; 0 of 20 single intervals exclude zero) | - | mean +0.48 (15 of 20 exclude zero) | - |
| same, (layer, C) chosen in the training half (layers 19-27 chosen) | - | 0.726 | 0.836 | 0.840 | mean +0.47 (range -0.16 to +1.20; 18 of 20 positive; 0 of 20) | - | mean +0.45 (15 of 20) | - |

**Interval for the average over random halves.** Each item is scored once per split; the interval is a premise bootstrap of the per-item mean difference, conditional on the splits drawn. "Tuned" means the n-gram C is chosen inside the training half (C = 100 in 15 of 22 fits, C = 10 in 7). "Rich" means tuned plus per-template boosting on extended parse features plus one boosting model with string and template ids.

| position | probe | text model | mean text acc | mean gain |
|---|---|---|---|---|
| final (10 halves) | fixed 19, 0.1 | repository (C = 1) | 0.836 | +0.55 [+0.21, +0.90] |
| final | chosen in half | repository (C = 1) | 0.836 | +0.47 [+0.14, +0.82] |
| final | fixed 19, 0.1 | tuned | 0.847 | +0.38 [+0.07, +0.68] |
| final | chosen in half | tuned | 0.847 | +0.29 [-0.01, +0.59] |
| final | fixed 19, 0.1 | rich | 0.855 | +0.17 [-0.07, +0.41] |
| final | chosen in half | rich | 0.855 | +0.12 [-0.12, +0.35] |
| content_end (5 halves) | fixed 19, 0.1 | repository (C = 1) | 0.835 | +1.08 [+0.53, +1.63] |
| content_end | chosen in half | repository (C = 1) | 0.835 | +0.95 [+0.40, +1.50] |
| content_end | fixed 19, 0.1 | tuned | 0.848 | +0.63 [+0.15, +1.10] |
| content_end | chosen in half | tuned | 0.848 | +0.65 [+0.16, +1.13] |
| content_end | fixed 19, 0.1 | rich | 0.854 | +0.42 [+0.03, +0.82] |
| content_end | chosen in half | rich | 0.854 | +0.53 [+0.10, +0.96] |

Canonical split, other text models:
- **Final, tuned text:** fit selection, score report +0.16 [-0.80, +1.12] (fixed), -0.40 [-1.52, +0.72] (chosen); fit report, score selection -0.52 [-1.48, +0.43] (fixed), -0.69 [-1.65, +0.26] (chosen).
- **Final, rich text:** -0.56 [-1.52, +0.32] and +0.35 [-0.52, +1.22] (fixed).
- **Content_end, repository text:** +1.36 [+0.00, +2.72] and +1.22 [-0.09, +2.60] (fixed).
- **Content_end, tuned text:** +0.48 [-0.72, +1.68] and +0.17 [-0.95, +1.30].

Does the gain survive?
- **Over the repository's text model:** yes in sign, at about +0.5 pp (final) and +1.0 pp (content_end) with half the training data. A single held-out half cannot detect it (no single interval of 20 excludes zero); the average over halves can.
- **Over a tuned text model:** +0.3 to +0.4 pp at the final token (interval touching zero), +0.6 pp at content_end.
- **Over the rich text model:** +0.1 to +0.2 pp at the final token (not distinguishable from zero), +0.4 to +0.5 pp at content_end (uncorrected p 0.02-0.04).
- Layer selection inside 1,150 items is unstable (layers 19 to 28 chosen), which costs the probe about 0.1 pp.

## E. What the gain is made of

Scripts: `e_content.py` (stages strata, text, null, artefacts), `e2_stronger.py`.

**E1. Where the probe helps** (repository text model vs text + probe; layer 19, final; items fixed / broken):

| stratum | report half: n, text acc, text+probe, fixed/broken, net | selection half: n, text acc, text+probe, fixed/broken, net |
|---|---|---|
| template has_collider | 260, 0.804, 0.823, 7/2, +5 | 240, 0.833, 0.825, 4/6, -2 |
| template has_confounder | 156, 0.801, 0.821, 3/0, +3 | 144, 0.819, 0.847, 6/2, +4 |
| template non-child descendant | 234, 0.863, 0.880, 6/2, +4 | 216, 0.856, 0.856, 7/7, 0 |
| template non-parent ancestor | 234, 0.846, 0.859, 11/8, +3 | 216, 0.796, 0.824, 9/3, +6 |
| template parent | 364, 0.951, 0.959, 6/3, +3 | 336, 0.932, 0.938, 6/4, +2 |
| queried pair adjacent in skeleton | 727, 0.835, 0.853, 23/10, +13 | 705, 0.821, 0.841, 28/14, +14 |
| queried pair not adjacent | 521, 0.908, 0.917, 10/5, +5 | 447, 0.915, 0.906, 4/8, -4 |
| pair has a common neighbour | 1038, 0.860, 0.878, 30/12, +18 | 945, 0.852, 0.863, 29/18, +11 |
| pair has no common neighbour | 210, 0.890, 0.890, 3/3, 0 | 207, 0.884, 0.879, 3/4, -1 |
| gold valid | 624, 0.865, 0.885, 17/5, +12 | 576, 0.856, 0.870, 19/11, +8 |
| gold invalid | 624, 0.865, 0.875, 16/10, +6 | 576, 0.859, 0.863, 13/11, +2 |
| text model wrong | 168, 0, 0.196, 33/0, +33 | 164, 0, 0.195, 32/0, +32 |
| text model right | 1080, 1, 0.986, 0/15, -15 | 988, 1, 0.978, 0/22, -22 |

- The gain is spread thinly over all five templates (+3 to +5 items each on the report half).
- It is concentrated on items whose queried pair is adjacent in the skeleton and has a common neighbour: net +27 of the +28 items over both halves are adjacent pairs. That is where the text model is weakest (0.83 vs 0.91).
- On items the text model gets wrong the probe alone is right 54 % (report) and 51 % (selection) of the time, against 82 % and 80 % where the text model is right. The probe has almost no independent knowledge of the answer on the text model's errors; the stack gains by re-ranking borderline items.
- Correlation of probe score and text-stack score on the report half: 0.71.

**E2. Is it something a slightly better text model captures? Yes.** Components are out-of-fold with the same nested folds:
- **ng4:** per-hypothesis-string TF-IDF 1-4-grams + logistic regression (the repository's component at C = 1).
- **tpl_FX:** per-template gradient boosting on the hand-parse features plus per-variable statement counts, premise length in characters and tokens, number of sentences.
- **gbm_FX_ids:** one boosting model on the same features plus string and template ids.
- **str_FX_lr:** per-string logistic regression on those features.
- **ans_\*:** the model's 12 answer logits / logit difference, per string or per template.

Adding one component to the repository's text model (parse + ng4 at C = 1; 0.865):

| added component | its own accuracy, report | accuracy gain, report | p | selection half | all items |
|---|---|---|---|---|---|
| 7B probe (layer 19, final) | 0.784 | +1.44 [+0.32, +2.56] | 0.015 | +0.87 [-0.35, +2.08] | +1.17 [+0.33, +2.00] |
| ng4 at C = 10 | 0.838 | +2.32 [+1.04, +3.69] | 0.001 | +1.56 [+0.17, +2.95] | +1.96 [+1.04, +2.92] |
| ng4 at C = 100 | 0.857 | +2.40 [+1.12, +3.77] | 0.001 | +1.48 [+0.17, +2.86] | +1.96 [+1.00, +2.92] |
| tpl_FX | 0.842 | +1.44 [+0.32, +2.64] | 0.017 | +0.35 [-0.78, +1.56] | +0.92 [+0.08, +1.71] |
| gbm_FX_ids (400 iterations) | 0.834 | +1.12 [+0.16, +2.08] | 0.038 | +0.52 [-0.35, +1.48] | +0.83 [+0.17, +1.50] |
| gbm_FX_ids (200 iterations) | 0.839 | +1.04 [+0.08, +2.00] | 0.053 | +0.43 [-0.52, +1.39] | +0.75 [+0.04, +1.42] |
| per-string 1-6-grams | 0.817 | +1.04 [+0.08, +2.00] | 0.060 | +0.52 [-0.61, +1.65] | +0.79 [+0.04, +1.54] |
| per-string bigrams | 0.742 | +0.96 [+0.00, +1.92] | 0.073 | +0.69 [-0.35, +1.82] | +0.83 [+0.12, +1.58] |
| per-string 1-5-grams | 0.813 | +0.88 [+0.08, +1.68] | 0.051 | +0.95 [-0.00, +2.00] | +0.92 [+0.29, +1.58] |
| str_FX_lr | 0.823 | +0.56 [-0.24, +1.36] | 0.25 | +0.35 [-0.35, +1.13] | +0.46 [-0.08, +1.00] |
| template x per-variable counts | 0.743 | +0.24 [-0.56, +1.04] | 0.70 | +0.35 [-0.52, +1.22] | +0.29 [-0.33, +0.88] |
| template x statements naming the pair | 0.765 | +0.16 [-0.80, +1.12] | 0.86 | -0.17 [-1.13, +0.78] | +0.00 [-0.67, +0.67] |
| answer logits per template + pair statements (boosting) | 0.760 | +0.40 [-0.40, +1.28] | 0.47 | +0.26 [-0.78, +1.30] | +0.33 [-0.29, +0.96] |
| answer logits per template | 0.611 | +0.32 [-0.24, +0.88] | 0.39 | +0.17 [-0.43, +0.78] | +0.25 [-0.17, +0.67] |
| answer logit difference per string | 0.575 | +0.16 [-0.32, +0.64] | 0.75 | +0.52 [-0.17, +1.22] | +0.33 [-0.08, +0.75] |

Probe increment over better text models (accuracy, pp; AUROC x100):

| text-only model | text acc report / selection | + 7B probe, final: report | p | selection | all items | AUROC, report | + 7B probe, content_end: report | p | selection | all items | + qwen3b probe, final: report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| repository: parse + ng4 (C = 1) | 0.865 / 0.858 | +1.44 [+0.32, +2.56] | 0.015 | +0.87 [-0.35, +2.08] | +1.17 [+0.33, +2.00] | +0.52 [+0.24, +0.81] | +2.24 [+1.04, +3.45] | 0.0005 | +1.30 [+0.09, +2.60] | +1.79 [+0.92, +2.67] | +0.88 [+0.00, +1.84] |
| parse + ng4 (C = 10) | 0.881 / 0.865 | +0.88 [+0.00, +1.76] | 0.080 | +0.00 [-0.95, +0.95] | +0.46 [-0.21, +1.12] | +0.31 [+0.08, +0.54] | +0.88 [-0.16, +2.00] | 0.145 | +0.52 [-0.61, +1.65] | +0.71 [-0.08, +1.50] | +0.64 [-0.08, +1.36] |
| parse + ng4 (C = 100); C chosen by selection-half accuracy (0.872 vs 0.865 vs 0.858) | 0.887 / 0.872 | +0.80 [-0.08, +1.68] | 0.120 | -0.43 [-1.30, +0.43] | +0.21 [-0.42, +0.87] | +0.25 [+0.03, +0.48] | +0.64 [-0.40, +1.68] | 0.291 | +0.26 [-0.69, +1.30] | +0.46 [-0.29, +1.17] | +0.24 [-0.56, +1.04] |
| parse + ng4 (C = 1) + ng4 (C = 10) | 0.889 / 0.873 | +0.24 [-0.64, +1.12] | 0.727 | -0.43 [-1.30, +0.43] | -0.08 [-0.71, +0.58] | +0.26 [+0.03, +0.50] | +0.24 [-0.80, +1.28] | 0.767 | -0.26 [-1.39, +0.87] | +0.00 [-0.75, +0.75] | +0.08 [-0.72, +0.88] |
| parse + ng4 (C = 10) + tpl_FX | 0.889 / 0.872 | +0.56 [-0.24, +1.44] | 0.268 | +0.09 [-0.78, +0.95] | +0.33 [-0.25, +0.96] | +0.17 [+0.00, +0.35] | +0.96 [+0.08, +1.92] | 0.064 | +0.26 [-0.78, +1.30] | +0.62 [-0.08, +1.37] | +0.40 [-0.08, +0.88] |
| forward selection on the selection half (parse, ng4, ng4 C = 10, str_FX_lr, per-template 4-grams, 5-grams, gbm_FX_ids 400) | 0.895 / 0.883 | -0.08 [-0.96, +0.72] | 1.000 | -0.61 [-1.30, +0.09] | -0.33 [-0.92, +0.21] | +0.17 [+0.01, +0.35] | +0.16 [-0.80, +1.12] | 0.871 | -0.78 [-1.65, +0.09] | -0.29 [-0.96, +0.33] | +0.08 [-0.48, +0.64] |
| all ten text components | 0.895 / 0.870 | +0.00 [-0.88, +0.96] | 1.000 | +0.78 [+0.00, +1.65] | +0.38 [-0.25, +1.00] | +0.16 [-0.01, +0.34] | +0.00 [-0.96, +0.96] | 1.000 | +0.43 [-0.61, +1.56] | +0.21 [-0.54, +0.92] | +0.16 [-0.48, +0.80] |

Reading:
- A text-only model that needs no new feature, only a less regularised n-gram classifier chosen on the selection half, reaches 0.887 on the report half. That is above the repository's text + probe (0.880).
- Over it the 7B probe's accuracy increment is +0.8 pp with an interval that includes zero on the report half, negative on the selection half, +0.2 pp on all items. Over richer text stacks it is zero.
- A small AUROC increment remains throughout (+0.0016 to +0.0031; lower bounds between -0.0001 and +0.0008). qwen3b shows the same (+0.0022 [+0.0007, +0.0039] over the tuned model).
- All five models over the tuned text model, final token, report / all items: 0.5B +0.00 / +0.00; 1.5B +0.40 / -0.04; Gemma +0.80 [+0.08, +1.52] / +0.33; 3B +0.24 / -0.12; 7B +0.80 [-0.08, +1.68] / +0.21.

Caveat: the label is a deterministic function of premise and hypothesis, so a sufficiently good text model reaches 1.0 and any probe then adds nothing. "The probe adds nothing to text" can only be relative to a named text model, and so can "the probe adds something". The phrase "strongest text-only model" should name the model.

**E3. The model's own answer.** 7B answers "No" on 92.8 % of items (accuracy 0.506). Answer-logit AUROC for the label: 0.514 pooled, 0.549 within hypothesis string. Probe gain:
- over text + per-string answer logit: +1.44 [+0.40, +2.56] (selection +0.61);
- over text + per-template answer logits: +1.12 [+0.08, +2.24];
- over text + all three answer components: +0.56 [-0.40, +1.44].

The first two are the fair ones; the third component mixes answer logits with pair-statement features and is itself a better text model. The answer logit does not explain the gain.

**E4. Null for the stacking step.** The same probe (layer 19, C 0.1) is trained on labels swapped within premise with probability 1/2, then stacked on the true labels with the full nested procedure, 8 draws. Null-probe accuracy 0.48-0.52. Gain on the report half: -0.16, -0.08, 0, 0, 0, +0.08, +0.08, +0.16 pp. All items: -0.04 to +0.12 pp. AUROC within ±0.0003. Adding a useless 3,584-dimensional probe column does not move the stack.

**E5. Specificity to the trained model.** There is no `cache/qwen7b-4bit-rand`.
- Proxy from the four smaller models (random-init probe at its own selected layer, same text model): +0.00 [-0.32, +0.32], +0.08 [-0.32, +0.48], -0.16 [-0.56, +0.24], +0.00 [-0.56, +0.56] pp; random-init probe accuracy 0.57-0.60.
- Low layers of the trained 7B model (14-16, probe 0.60-0.64) give -0.3 to -0.1 pp.

So a probe of about 0.6 accuracy adds nothing and the increment grows with probe accuracy. That the 7B increment needs trained 7B weights is likely but not shown. Missing controls:
1. Random-init 7B at the same layer and position (needs a GPU pass).
2. The unquantised model, to separate the model from the 4-bit quantisation.
3. A probe of an encoder that has seen no causal data, to separate "a trained transformer encodes which statements name the pair" from anything about validity.

**E6. Artefacts in the activations** (layer 19, final token):
- NaN 0, inf 0, max abs 36.75, exact zeros 0.1 % of values.
- 2,400 of 2,400 rows distinct; no pair of rows with cosine above 0.999; highest cosine between items of different premises 0.993.
- Nearest neighbour is the premise-mate for 3.6 % of items and shares the hypothesis string for 68 % (the final token mostly encodes the hypothesis).
- Row norm 76.0-80.8, AUROC 0.502 for the label; token count AUROC 0.501.
- Every stored value has its three lowest mantissa bits equal to zero (8 significant bits: bfloat16 written into float16).
- Multiplying every value by 1 + u, u uniform in ±2^-8, and refitting everything: probe accuracy 0.788 / 0.781; gain +1.36 / +1.52 pp on the report half, +0.96 / +1.17 on all items; 29 and 35 of 2,400 probe predictions change sign.
- Item order alternates invalid/valid (the hazard documented in `data.c2c_debiased_pool`); nothing in the probe path uses the index.

No artefact found. Rounding noise is worth about ±0.1 pp, smaller than every interval above. Quantisation cannot manufacture label information.

## F. Verdict

**What may be said.** On the debiased pool, the layer-19 final-token probe of qwen7b-4bit (0.784 [0.762, 0.806]) improves a stack of the hand parse and the per-hypothesis 4-gram classifier, as specified in 05b with fixed hyperparameters, by 1.4 points on the report half (+0.4 to +2.6; 18 net items; p = 0.015 uncorrected), 0.9 on the selection half and 1.2 on all items. This reproduces exactly and is correctly nested. It does not depend on the layer (19-26) or on convergence, needs C >= 0.1, is about 0.5 point when everything is fitted on half the premises, and is larger at the end of the hypothesis (+2.2, p = 0.0005).

**What may not be said.** That this is the first evidence of information beyond the text, that it marks a change at 7B, or that the probe beats "the strongest text-only model".
- The p-value does not survive a family-wise correction over the five model cells (0.077).
- The increment is not distinguishable from those of 3B and Gemma-2B (difference +0.6 [-0.5, +1.6]).
- It shrinks to +0.8 [-0.1, +1.7] (report), -0.4 (selection), +0.2 [-0.4, +0.9] (all items) once the n-gram component's regularisation is chosen on the selection half, and to zero over a forward-selected text stack.
- A second text component gives the same one to two points.

The earlier summary still stands for all five models: as accurate as a per-template lookup, below the hand parse, adds nothing demonstrable in accuracy to a tuned text-only model. Two wording changes are needed: name the text model instead of calling it "the strongest", and note that small AUROC increments (about +0.002 to +0.005) are present from 1.5B upward and that a residual of about half a point at the end-of-hypothesis position is not excluded.

**The single analysis that would most change my confidence:** a pre-specified replication on a second, disjoint debiased pool. That means new premises, new 7B activations at both positions and a random-init 7B pass, with the text model's hyperparameters chosen by inner cross-validation before the probe is looked at, and one stated statistic (accuracy gain of the stack, premise sign-flip test). An increment of one point or more over that tuned text model would be a finding. On the present pool the data are compatible with anything from zero to about one point.

## Conduct of the audit

- **Environment:** CPU only, 2 threads, one fitting process at a time. `resid.npy` was opened with `mmap_mode="r"` and read one layer and position at a time. No language model was loaded and no network was used. The one overlap: `e2_stronger.py` (stacking of cached scores, 1 thread) ran alongside `d_heldout.py` for about three minutes.
- **Read-only:** no repository script was run. `causal_probe.data`, `causal_probe.probes` (feature and premise parsers) and `causal_probe.stats` (bootstrap) were imported with bytecode writing disabled. All outputs are under `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit14/`.
- **Other activity in the repository:** during the audit other processes changed files (a commit at 09:18; RESULTS.md, README.md, EXPLAINER.md, figures/, notes/, scripts/06_figures.py, scripts/12_tables.py). None of the audit's inputs changed: `results/probe_content_qwen7b-4bit_c2c_debiased.json` (08:45), `results/confound_qwen7b-4bit_c2c_debiased.json` (08:33) and `cache/qwen7b-4bit/c2c_debiased/*` (07:49-08:28) kept their timestamps.
- **Not done:** tuning of the hand-parse boosting model; n-gram C above 100; other n-gram ranges combined with tuned C; text models that implement inference rules; the held-out analysis with the forward-selected text stack; any analysis of other pools for 7B (no probe files exist for them).
- **My own post-hoc choices** (which extra text components to try, looking at content_end) are subject to the same multiplicity argument as the claim. Where a choice mattered it was made on the selection half (n-gram C, forward selection) or inside the training half (section D), and report-half numbers are given alongside selection-half and all-item numbers.
- **Limits of the intervals:** all intervals condition on the fitted models and on this pool of 1,200 premises; the average over random halves conditions on the splits drawn.

## Files (all in the audit14 directory above)

| file | content |
|---|---|
| `aud.py` | folds, components, nesting, stacking, bootstrap, tests (own implementation) |
| `a_reproduce.py`, `a_reproduce.json`, `a_reproduce.log` | section A |
| `b_family.py`, `b_family.json`, `b_family_corrected.json` | family of intervals from result files, corrected p-values |
| `b_models.py`, `b_models.json`, `b2_content_end.py`, `b2_content_end.json` | all five models, random-init controls, content_end |
| `c_sensitivity.py`, `c_sensitivity.json`, `c_main.log`, `c_conv.log` | section C |
| `d_heldout.py`, `d_heldout.json`, `d_heldout_content_end.json`, `d2_heldout_tuned.py`, `d2_heldout.json`, `d2_heldout_content_end.json`, `d3_heldout_rich.py`, `d3_heldout_rich.json`, `d3_heldout_rich_content_end.json`, `d*.log` | section D |
| `e_content.py`, `e_content.json`, `e2_stronger.py`, `e2_stronger.json`, `e_*.log` | section E |
| `nest/*.npz` | cached nested out-of-fold scores of every component (text components are model-independent) |
| `tables.py`, `tables.md` | generated tables (not the report) |

# Wording audit of the proposed RESULTS.md (audit 13, raw report)

Sat 14 Nov 2026. Audited: `results_proposal/RESULTS_proposed_rendered.md` and its template `RESULTS_template_proposed.md`, against `results/*.json`, `report/tables/*.md`, the three embedded figures, the raw audit reports in `notes/`, `notes/claims_pass_14nov_raw.md`, `notes/log.md` and `results/round2_logs_*`. `report/report.md` and `EXPLAINER.md` were not used as evidence (only to confirm that Sections 3.4 and 4.1 exist). Nothing in the repository was written. No model was loaded.

Own scripts, in the audit13 scratch folder:
- `a_within.py`: pooled and within-string answer statistics for five models, from `cache/<model>/c2c_debiased/meta.json` and `data/derived/c2c_debiased_*.json`, with own AUROC, own adjacency parse, own bootstrap and permutation.
- `b_probe7b.py`: 7B and 3B probe accuracy on the report half from the saved out-of-fold scores.
- `c_claim3.py`: the 32 steering comparisons, the alignment file, adjacency shares, direction geometry.

What I did not do: refit any probe (the 7B probe accuracy is recomputed from saved out-of-fold scores only), recompute steering slopes from raw npz files (audits 8 and 11 did; I re-read the JSONs), or check anything about the running held-out ("settle") run beyond what `notes/log.md` records.

## Recomputed numbers

| Quantity | Page | Result file | My recomputation |
|---|---|---|---|
| Hypothesis-only AUROC, official test split | 0.895 [0.863, 0.919] | `benchmark_baselines.json` 0.8947 [0.8628, 0.9193], 21 premise clusters | re-read |
| Probe below strongest text-only classifier | all eight cells | `confound_*`: 8 of 8 paired differences negative, intervals exclude zero (-0.049 to -0.240); also 5 of 5 on the debiased pool | re-read from `audited_verdict.md` |
| Qwen 3B probe, debiased pool | 0.765 [0.743, 0.788] | same | 0.7652 [0.742, 0.788] from out-of-fold scores |
| Lookup, hand parse | 0.765, 0.816 | same | re-read |
| Qwen 7B probe | 0.784 [0.762, 0.806] | `probe_content_qwen7b-4bit`: same; `confound_qwen7b-4bit` (table): [0.760, 0.806] | 0.7837 [0.760, 0.805]; per seed 0.783, 0.773, 0.778 |
| 7B pooled answer AUROC | 0.514 [0.490, 0.536] | same | 0.5135 |
| 7B within-string AUROC | 0.549 [0.530, 0.569] | same; p 0.0005 | 0.5491 [0.531, 0.568]; p 0.001 (floor of my 1,000 permutations); 33 of 48 strings above 0.5 |
| 7B valid item above invalid item of the premise | 53.6% | 0.5358 | 0.5358 [0.510, 0.562]; no ties; 1,200 premises |
| Same statistic, smaller models | not on page | 0.515, 0.517, 0.508, 0.520 | same (0.5B, 1.5B, Gemma, 3B) |
| 7B within string and adjacency cell | not on page | 0.589 [0.560, 0.620] | 0.5891 |
| Steering comparisons, first ranks | 4 of 32, 1.52; 3 of 16, 0.76 | same; P 0.064 and 0.038 | same; at responding sites 4 of 28 (1.33; P 0.042) and 3 of 14 (0.67; P 0.026) |
| Mean rank, gold class-mean differences | 6.44 of 21 | 6.4375; p 0.0011 | 6.4375. Answer position: 4.00 (debiased), 5.25 (original). End of hypothesis: 5.75 (debiased), 10.75 (original) |
| Four slopes | +0.738, -0.251, +0.475, +0.865 | same | re-read |
| Gemma margin | "first by 3%" (typed) | 0.4754 against 0.4622 | ratio 1.029 |
| Natural-scale effect, 3B arm | +0.023 | 0.0228 | re-read |
| Adjacency shares, end of hypothesis | 11%, 6%, 1% | `power_option_b_qwen7b.json`: 0.1150, 0.0580, 0.0056 | contrast / benchmark from the adjacency files: 11.5%, 5.8%, 0.56%; Gemma 0.3% |
| Hypothesis string alone, balanced pool | 86% | 0.85875 | re-read |

## Every sentence and table row

Verdicts: supported / wrong / overstated / stale / unsupported / unclear. "W n" refers to the findings below.

### Header

| # | Sentence | Verdict | Evidence |
|---|---|---|---|
| 0 | Question | not a claim | |
| 1 | "Qwen2.5-0.5B, 1.5B, 3B and Gemma-2-2B on Corr2Cause and CLadder: probes against text-only baselines, steering, patching." | supported | `confound_*`, `steering_*`, `patching_*` for the four models |
| 2 | "Qwen2.5-7B (4-bit): answers, and probes on the debiased pool." | supported | `answer_bias.json` (four item sets), `probes_qwen7b-4bit_c2c_debiased_*`; no steering or adjacency file for this model |
| 3 | "Twelve independent audits." | unclear, and stale on arrival (W6) | `notes/`: eleven reports that call themselves audits plus one "claims pass" |
| 4 | "Numbers are from `results/*.json`; details and all caveats are in `report/report.md`." | supported for the templated numbers; "3%", "of 21", "11 expected" are typed and correct | template |

### Claim 1

| # | Sentence | Verdict | Evidence |
|---|---|---|---|
| H1 | Headline | supported | |
| 5 | Hypothesis-only classifier reaches AUROC 0.895 [0.863, 0.919] | supported; interval rests on 21 premises (W13) | `benchmark_baselines.json`, `corr2cause_official_splits/test/hypothesis_only/auroc` |
| 6 | "below the strongest text-only classifier in all eight model × benchmark cells" | supported | `confound_<model>_{c2c_probe,cladder}.json`, table `audited_verdict.md` |
| 7 | "Figure (four of the eight cells)" | supported | fig5 shows four models on the balanced pool |
| 8 | "Checked by audit 1 ... and audit 6 (recomputed for four models)" | supported | `audit_tier1_raw.md` finding 1; `audit_consolidation_raw.md` (`e_claim1.py`, `f_cells.py`) |

### Claim 2

| # | Sentence | Verdict | Evidence |
|---|---|---|---|
| H2a | "With every single-input shortcut removed" | supported | `probe_content_*`, `A0_single_input` all 0.500 |
| H2b | "a probe on the larger models' activations is as accurate as a lookup ..., no more" | overstated for the 7B model; "the larger models" undefined (W1) | `probe_content_qwen7b-4bit`: probe minus lookup AUROC +0.029 [+0.006, +0.055]; `incremental_value/over_strongest_text_only/gain_accuracy` +0.014 [+0.004, +0.026] |
| H2c | "pooled over hypothesis types the models answer at chance" | supported by the AUROC criterion; wording ambiguous (W7) | `answer_bias.json`: pooled AUROC intervals include 0.5 for all five; the within-string mean, which also pools the types, is 0.549 for 7B and 0.473 for Gemma |
| 9 | 3B: probe 0.765 [0.743, 0.788], lookup 0.765, parse 0.816, pooled answers 0.507 [0.484, 0.530] | supported | `probe_content_qwen3b`, `answer_bias.json` |
| 10 | 7B: probe 0.784 [0.762, 0.806]; pooled answers 0.514 [0.490, 0.536] | supported numerically; misleading by omission, since no comparison is given for the 7B probe (W1) | as above |
| 11 | "equal accuracy is not equal content (probe and text models are right on partly different items)" | supported | `incremental_value/probe_right_when_hand_parse_wrong` 0.57 to 0.64 |
| 12 | '"at chance" holds for the pooled items only' | supported | `answers_within_string.md` |
| 13 | "Inside a hypothesis string the four smaller models' answers follow adjacency of the queried pair, not validity" | supported in substance; "follow" is loose (W14) | `debiased_answers.md` (AUROC for adjacency by type on both sides of 0.5); within-string AUROC 0.506, 0.507, 0.473, 0.511 |
| 14 | "the 7B model's follow validity, weakly: AUROC within string 0.549 [0.530, 0.569], ... 53.6% of premises" | supported and recomputed; one omission (W8) | `answer_bias.json`, `a_within.py` |
| 15 | "Checked by audit 3 ..., audits 5 and 6 ..., audit 11 (the 7B model; it found the within-string result)" | overstated for audit 11 (W2) | `audit_runs_b_d_raw.md`, "Could not check": the 7B probe files "were not opened" |

### Claim 3

| # | Sentence | Verdict | Evidence |
|---|---|---|---|
| H3 | "No intervention shows that what a probe reads is what the answer is computed from." | supported as a statement of absence, given the caveats | see Task 2 below |
| 16 | 32 comparisons (four models, two pools, two tokens, two kinds of direction), 20 matched controls each | supported | `steering_controls_multiplicity.json`, `by_arm` (32 entries, `n_controls` 20) |
| 17 | "exceeds all its controls in 4 of them, where chance gives 1.52; on the debiased pool in 3 of 16 (0.76)" | supported; omission (W9) | `summary/all`, `summary/debiased_pool` |
| 18 | "None survives a correction for multiple comparisons, and with 20 controls none could." | supported | smallest attainable p 1/21 = 0.048; 0.05/32 = 0.0016 |
| 19 | 3B, end of hypothesis, class-mean difference, +0.738 per unit step; "a direction loaded on adjacency, not on validity"; +0.023 at natural scale | numbers supported; "not on validity" unsupported (W5); "per unit step" unclear (W15) | `direction_geometry_qwen3b_c2c_debiased_hyp.json` |
| 20 | 1.5B, same token, gold probe, -0.251, towards "valid" lowers Yes | supported | `steering_controls_qwen1.5b_c2c_debiased_hyp.json`; signed rank 21 of 21 |
| 21 | Gemma, answer position, gold probe, +0.475; "first by 3% and not robust" | supported | ratio 1.029; eleventh audit F2 (first in 81% of item resamples; non-converged fits) |
| 22 | 0.5B, original pool, class-mean difference, +0.865; "weaker controls" | supported; token not named (W15) | `steering_controls_qwen0.5b_c2c_probe.json`; controls shuffled over the whole item set |
| 23 | "taken together the gold class-mean differences do rank above their controls (mean rank 6.44 of 21; 11 expected)" | supported | `summary/all/mean_rank/gold_diffmeans` (p 0.0011) |
| 24 | "At the answer position that is the answers' dependence on adjacency read back, not a use of validity (report 4.1)." | overstated (W3) | `steering_slope_vs_answer_alignment.json` |
| 25 | "moves the answer by 11%, 6% and 1% of what the answers' dependence on adjacency would give (...; Gemma-2-2B's layer there responds to nothing)" | arithmetic supported; wording misleading (W4) | `adjacency_*_c2c_debiased_hyp_best.json` |
| 26 | "At the answer position the same shift does move the answer, alike for every hypothesis type, which locates the answer." | supported; unclear and unquantified (W20) | `adjacency_contrast.md`, `adjacency_transfer_summary.md` |
| 27 | "No intervention was run on the 7B model." | supported | no such result file |
| 28 | "A held-out test of the first two cases is running; how each outcome will be read was fixed beforehand." | unclear (W11) | `notes/log.md` 07:56 only |
| 29 | Figure; "Checked by audits 2, 4, 5, 6, 8 and 11" | supported; figure in tension with sentence 25 for Gemma (W12) | fig12; audit reports |

### Retraction table

| Row | Verdict | Note |
|---|---|---|
| 1 "represents the answer but doesn't say it" / 86% / Audit 1 | supported | 0.85875 |
| 2 "gold-probe direction is inert" / Audit 2 | supported | `audit_tier2_raw.md` findings 1 to 3 |
| 3 "Patching shows where the answer is formed" / Audit 2 | supported | finding 14 |
| 4 "beats the shallow controls and adds nothing to the parser" / Audit 3 | supported | findings 1, 2 |
| 5 "answers do not express what the probe reads" / Audit 3 | supported | opposite signs by template |
| 6 First adjacency experiment / Audit 4 | supported | F3 |
| 7 "removes a sixth ... beyond all 20 controls" / Audit 5 | supported | F1 |
| 8 "letters out of alphabetical order" / Audit 5 | supported | |
| 9 "Splitting the class difference" / Audit 6 | supported | F4 |
| 10 "A pre-specified contrast"; "steering stays within the range of its controls" / "true of one model only" / Audit 8 | first half supported (F5); "true of one model only" unclear (W16) | |
| 11 "First ranks ...: no more than chance gives" / Audit 11 | supported (F4); the claim was a draft, never on this page | |
| 12 "The 7B model answers the debiased pool at chance" / Audit 11 | supported (F1) | |
| 13 "Eight bugs (...)" / "Ingest check, own review, audits 4 to 7 and 11" | count matches its own list; attribution partly unverifiable (W17) | |

### Limitations

| # | Sentence | Verdict | Evidence |
|---|---|---|---|
| L1 | "Interventions on models of up to 3 billion parameters whose answers do not follow validity. The 7B model follows it weakly, was run in 4-bit quantisation, and was not intervened on. Two benchmarks; linear probes. Nothing here says what a model that can do the task would show." | supported | |
| L2a | "four models, three of one family" | supported | |
| L2b | "one layer per model at the end of the hypothesis" | roughly supported (W18) | |
| L2c | "20 controls per comparison and 32 comparisons without adjustment" | supported | |
| L2d | "an erasure that removes about half of the readable adjacency on held-out items" | supported for all four models | `adjacency_sites.md`: excess AUROC over 0.5 falls by 44% to 53% at the eight best-layer sites |
| L2e | '"No intervention shows" is not "unused".' | supported | |
| L3 | "The stronger baselines, the debiased pool, the within-string statistics and every corrected analysis were introduced after seeing results; ..." | supported | notes in the result files |
| L4a | "Probe accuracies are three-seed ensembles" | supported | `robustness/three_seed_ensemble_accuracy` |
| L4b | "most probe fits behind the GPU runs' steering directions stopped at the iteration limit" | overstated (W10) | `results/round2_logs_*/*controls*.log` |
| L5 | 3B CLadder on a different draw | supported | table footnote † |

## Findings

### W1. Must fix. Claim 2's "no more" is contradicted for the 7B model

Evidence, `probe_content_qwen7b-4bit_c2c_debiased.json`:
- probe minus lookup: accuracy +0.018 [-0.011, +0.049], AUROC **+0.029 [+0.006, +0.055]**;
- `baselines/summary`: the 7B probe exceeds two binding references and is not distinguishable from the 4-gram classifier (0.805); the four smaller models are below the 4-grams;
- `incremental_value/over_strongest_text_only/gain_accuracy` **+0.014 [+0.004, +0.026]** (all items +0.012 [+0.003, +0.020]; AUROC +0.005 [+0.002, +0.008]); over the hand parse +0.046 [+0.026, +0.065].

Same key for the others: 0.5B +0.002 [-0.003, +0.008]; 1.5B -0.001 [-0.010, +0.008]; Gemma +0.009 [-0.002, +0.019]; 3B +0.009 [+0.000, +0.018]. Over the hand parse the gain is +0.022 to +0.046, intervals excluding zero, for all four models from 1.5B up. The log (08:34) calls 7B "the first model for which the second interval clearly excludes zero".

"As accurate as the lookup" holds by accuracy for 1.5B, Gemma, 3B and (within its interval) 7B. "No more" reads as "holds nothing beyond the lookup", which the files contradict, most clearly for the one model the page adds. The page also gives the 7B probe number with nothing to compare it with.

Replacement headline: **"2. With every single-input shortcut removed, a probe on the activations of the models from 1.5B to 7B is about as accurate as a lookup of the premise statements about the two queried variables, and less accurate than a hand-written parse; ranked across all items at once, the models' answers are at chance."**

Replacement for the 7B part of the number line: "Qwen2.5-7B: probe {{...gold_probe/accuracy|ci}}; added to the strongest text-only classifier it gains {{probe_content_qwen7b-4bit_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}}, the only model for which that interval clearly excludes zero; pooled answers {{...|ci}}."

### W2. Must fix. "Checked by ... audit 11 (the 7B model ...)" covers the answers, not the probe

`audit_runs_b_d_raw.md`, "Could not check": "Files being written by the running job (`probes_qwen7b*`, `confound_qwen7b*`, `probe_content_qwen7b*`, `logs/q7b_*`) were not opened." The 7B probe accuracy and its increments entered `results/` at 08:28 to 08:45, after that audit's report (08:27). The user's rule in `notes/progress.md` is an independent audit before anything new enters RESULTS.md.

What this audit adds: the 7B probe accuracy on the report half, recomputed from `cache/qwen7b-4bit/c2c_debiased/probe_oof_C0.1.npz` at layer 19, is 0.7837 [0.760, 0.805], matching the file. I did not refit the probe or the text classifiers, and did not recompute the nested stacking behind the increments.

Replacement: "*Checked by* audits 3, 5 and 6 (probe ladder, four models) and 11 (the 7B answers; it found the within-string result). The 7B probe was checked from its saved scores only."

### W3. Must fix. "At the answer position that is the answers' dependence on adjacency read back, not a use of validity"

Evidence, `steering_slope_vs_answer_alignment.json` (debiased pool only, eight sites):

| Site | r of slope with cosine to the answer direction, over the 20 controls | Gold direction: predicted slope | Observed |
|---|---|---|---|
| Qwen 0.5B, answer position | **-0.15** | +0.004 | +0.150 |
| Qwen 1.5B, answer position | +0.90 | +1.60 | +1.89 |
| Gemma, answer position | +0.93 | -1.79 | -1.80 |
| Qwen 3B, answer position | +0.96 | +3.22 | +4.02 |
| End of hypothesis, four models | -0.33 to +0.11 | about 0 | +0.287, +0.095, +0.008, +0.738 |

Four problems with the sentence:
1. It holds in three of four models. In Qwen 0.5B the relation is absent.
2. It was measured on the debiased pool only. The mean rank of 6.44 is over both pools and both tokens. The answer position on the debiased pool is 4 of the 16 comparisons (mean rank 4.00). At the end of the hypothesis on the debiased pool the mean rank is 5.75 and this account explains none of it (3B: predicted +0.022, observed +0.738).
3. The file shows alignment with the answer direction, that is, the answers' within-question correlation with the label. That this correlation runs through adjacency is the author's inference. Within cells of (string, adjacency) the answers of Qwen 1.5B and Gemma still depend on the label, negatively (0.436 and 0.467, both p 0.0005, `answer_bias.json`), so adjacency of the queried pair is not the whole of it. The eleventh audit's wording (F4) is "the behavioural correlation between label and answer, read back through the answer direction".
4. "Not a use of validity" asserts a negative. The file supports "does not show a use of validity".

Replacement: "At the answer position on the debiased pool this is the answers' own within-question correlation with the label read back: over the 20 controls a direction's slope follows its cosine with the answer direction (r 0.90 to 0.96 in three models, absent in Qwen2.5-0.5B), and the gold direction's cosine predicts its slope. It does not show that validity is used. At the end of the hypothesis this account explains nothing."

### W4. Must fix. "moves the answer by 11%, 6% and 1% of what the answers' dependence on adjacency would give"

The ratios are right: translation contrast over full-mediation benchmark at the best end-of-hypothesis layer is 11.5%, 5.8% and 0.56% (`adjacency_*_hyp_best.json`; identical to the three values in `power_option_b_qwen7b.json`). Three things make the sentence misleading:
1. Two of the three are inside the range of their 20 controls: Qwen 1.5B has 45% of controls smaller in absolute value, Qwen 3B 5%. The controls span about ±15% (1.5B) and ±7% (3B) of the benchmark. These are not measured effects.
2. "1%" is 0.56% rounded up; with point-estimate signs for all five types it is -0.7% (`audit5/translation_contrast_under_other_sign_choices`).
3. Qwen 0.5B's 11.5% is the only one outside its controls (above all 20 signed, 95% in absolute value), and it is a push towards No for all five types (`n_translation_effects_towards_no` 5). The current page says both; the proposed page drops both. Under a headline "No intervention shows ..." the one end-of-hypothesis result outside its controls should not disappear.

The quantity is a signed contrast over the hypothesis types whose observed gap excludes zero, with signs read from items that include the evaluated ones (retraction row 10). It is not an amount by which "the answer moves".

The template reads these from the power-calculation file, whose review is still running. The numbers are a plain ratio of two audited keys, so nothing is wrong, but the page then depends on a file outside the audited set.

Replacement: "Shifting items along the encoding of adjacency at the end of the hypothesis gives 11%, 6% and under 1% of what full mediation of the answers' dependence on adjacency would give (Qwen2.5-0.5B, 1.5B, 3B). Only the first is outside its 20 controls, and it pushes towards No for every hypothesis type. Gemma-2-2B's layer there responds to nothing, the positive control included."

### W5. Must fix. "a direction loaded on adjacency, not on validity"

`direction_geometry_qwen3b_c2c_debiased_hyp.json`: 55% of the direction's squared norm lies in the span of the five adjacency class-mean differences (controls: mean 18%, largest 52%); cosine +0.69 with the direct-causation adjacency difference (largest control 0.55). "Loaded on adjacency" is supported. "Not on validity" is not: the direction is the validity class-mean difference, 45% of it lies outside the adjacency span, and the eighth audit's conclusion (Q1 d, f) is that the comparison "cannot separate" validity from adjacency.

Replacement: "about half of it is the adjacency contrast of the direct-causation hypotheses, so it cannot separate validity from adjacency".

### W6. Must fix. "Twelve independent audits"

`notes/` holds eleven reports that call themselves audits, numbered first to eleventh in their headers and in the log (tier1, tier2, debiased, round2_code, round2_results, consolidation, round3_code, round2_runs, settle_code, settle_code2, runs_b_d), and one "claims pass" (`claims_pass_14nov_raw.md`), which the log does not number. "Twelve" is right only if the claims pass is counted, and then the page's own references ("audit 8", "audit 11") use a numbering in which the twelfth has no number.

This report is a thirteenth review. A review of the power calculation is running. A running review should not be counted.

What the page should carry: the number of raw reports on file when it is built. Once this report is filed: "Eleven independent audits of code and results and two checks of the wording" (thirteen reviews). If the power-calculation review is filed before the build, add it.

### W7. Should fix. "pooled over hypothesis types" is ambiguous

The statistic that is at chance is the AUROC over all 2,400 items at once. The mean within-string AUROC also pools the five hypothesis types, and it is 0.549 (p 0.0005) for 7B and 0.473 (p 0.0025) for Gemma. Table `debiased_answers.md` prints 0.549 [0.526, 0.571] in the 7B row "all templates". Also, for 7B one pooled statistic excludes chance: cross-fitted best-threshold accuracy 0.525 [0.505, 0.543].

Replacement phrase: "ranked across all items at once (ignoring which hypothesis is asked), the models' answers are at chance by AUROC".

### W8. Should fix. "the 7B model's follow validity, weakly"

Supported on average, and beyond adjacency (0.589 [0.560, 0.620], above 0.5 in all five types). Not uniform: for "X directly causes Y", the largest type (14 strings, 700 items), the 7B answers go against validity (0.442 [0.401, 0.483], p 0.005).

The 53.6% compares two items that never share a hypothesis string (0 of 1,200 premises), which is the comparison the page calls insensitive. The same statistic is 52.0% for Qwen 3B. It is a fair measure of size and a poor discriminator.

Replacement: "the 7B model's follow validity on average, weakly, and against it for direct-causation hypotheses (0.442): AUROC within string ..., and the valid item of a premise gets the higher answer logit in 53.6% of premises (Qwen2.5-3B: 52.0%)."

### W9. Should fix. The count of 32 includes four comparisons at sites that respond to nothing

`summary/responding_sites`: 28 comparisons, 4 first ranks, 1.33 expected (P 0.042); debiased pool 14, 3, 0.67 (P 0.026). The page says elsewhere that Gemma's end-of-hypothesis layer responds to nothing and still counts its two comparisons in the denominator. The page also gives expected counts without the probabilities (0.064 and 0.038), and the aggregate mean rank without its p (0.001).

Replacement: "... in 4 of them, where chance gives 1.52 (4 of 28 and 1.33 without the two sites that respond to nothing); on the debiased pool in 3 of 16 (0.76)."

### W10. Should fix. "most probe fits ... stopped at the iteration limit"

ConvergenceWarning counts in `results/round2_logs_*/*controls*.log`, of 22 fits per stage: Qwen 3B 21, 0, 0, 0; Qwen 1.5B 21, 0, 0, 20; Gemma 21, 21, 21, 4. That is 129 of 264, about half; "most" holds for Gemma only (67 of 88). Of the four comparisons the page names, only Gemma's is affected: Qwen 1.5B's stage has no warning, and the other two are class-mean differences.

Replacement: "about half of the probe fits behind the GPU runs' steering directions stopped at the iteration limit (all those behind Gemma-2-2B's first rank, none behind Qwen2.5-1.5B's)."

### W11. Should fix. "A held-out test of the first two cases is running; how each outcome will be read was fixed beforehand."

Nothing in `results/` bears on this. From `notes/log.md` (07:56):
- the launch reported is Qwen 1.5B's; Qwen 3B's is third in the launch order, so "is running" is confirmed for one of the two;
- the mapping of outcomes to wording was recorded after the launch and before any output was seen;
- "held-out" means the direction refitted on the other half of the blocks, which the log says is not an independent draw (cosine 0.84 and 0.53);
- the threshold corrects for 24 comparisons. Over the 32 the page counts, the smallest p the test can give (1/501) is above 0.05/32.

The sentence will be stale the day a result arrives.

Replacement: "A test of the first two with the direction refitted on the other half of the data and 500 controls is under way; its reading was written down before any output was seen."

### W12. Should fix. Figure 12 against the text

The Gemma column is annotated "contrast +0.001 of 0.386; larger than 95% of controls" and "mean |gap| 0.285 → 0.284; larger than 80% of controls", with no mark that the positive control fails there. The page says this layer "responds to nothing". A reader of the figure alone sees a result at the edge of its controls. The table marks the site (◊); the figure does not. The Qwen 0.5B column says "larger than 95% of controls", which the page no longer mentions (W4).

Fix: mark the Gemma column in the figure ("site does not respond: positive control inside its controls"), or say so in the caption line.

Figures 5 and 13 agree with the text. Figure 13 shows the 7B probe between the lookup and the 4-gram classifier, which fits the replacement headline in W1 better than "no more".

### W13. Should fix (carried over from the sixth audit, m15). The 0.895 interval rests on 21 premises

`benchmark_baselines.json`: `n_clusters` 21. The eleventh audit calls such intervals "rough". Three words in the number line: "(21 premises)".

### W14. Optional. "follow adjacency of the queried pair"

The dependence has opposite signs by hypothesis type and by model (fig12, top row). "Depend on" is more exact than "follow".

### W15. Optional. Two unclear labels in "The four"

- "logits per unit step": one step is 11.5 natural SDs of the direction (`natural_scale/one_s_in_natural_sds`). Say "per step of 11 natural SDs", or drop the unit and keep "+0.023 at natural scale".
- The Qwen 0.5B case does not name its token (answer position).

### W16. Optional. Retraction row 10, "true of one model only"

The retracted sentence was about the gold probe's weight vector. For that direction it is true of two of four models (Qwen 0.5B ranks 6, 12, 17, 9; Qwen 3B 17, 16, 5, 4) and false for two (Qwen 1.5B, Gemma). Counting both gold directions on the debiased pool it is true of Qwen 0.5B only. Replacement: "false for three of the four models once both gold directions are counted".

### W17. Optional. The bug row

- The list has eight entries. The log and the audits record more defects than these (fourth audit F2 and F10; seventh audit, one critical and eight major; ninth and tenth audits of the settle script). "Eight" is a selection.
- I can tie listed bugs to: the ingest check (sample); own review (interval, discarded folder, two-GPU capture, figure script, second ingest overwrite); the fourth audit (fix not applied, per sixth audit m14); the sixth audit (first ingest overwrite, F5). I could not tie any of the eight to audits 5, 7 or 11.

Replacement: "Eight bugs that reached a result or a run (`notes/log.md`) | | Ingest check, own review, audits 4 and 6".

### W18. Optional. "one layer per model at the end of the hypothesis"

Steering used one layer per model and pool (the layer differs between pools: 13/11, 19/17, 21/15, 25/25). The adjacency test used two layers per token; the earlier one fails its positive control in every model, and Gemma has no responding end-of-hypothesis layer on the debiased pool. "At most one responding layer per model" is the accurate form.

### W19. Optional. Two small things

- The 7B probe interval is [0.762, 0.806] on the page (`probe_content`) and [0.760, 0.806] in tables `text_baselines.md` and `audited_verdict.md` (`confound`). Two bootstrap runs; harmless, visible to a reader who compares.
- "Independent audits": the raw reports describe "a fresh reviewer agent" given code and results. The page does not say that the auditors are reviewer agents commissioned by the author.

### W20. Should fix. "At the answer position the same shift does move the answer, alike for every hypothesis type"

Supported:
- contrast at the best answer-position layer is 98%, 64%, 69% and 56% of the benchmark (0.5B, 1.5B, Gemma, 3B);
- the effect matrix is 100%, 99%, 97% and 92% rank one;
- in the cells where target and source signs differ, the effect follows the source vector in all four models.

Two gaps. The shift is beyond all 20 controls in two of four models only (0.5B and 3B; 85% and 90% in the others). And "alike for every hypothesis type" can be read as "all five shifts push the same way", which is false (4, 4, 4 and 1 of 5 towards No).

Replacement: "At the answer position the same shift moves the answer (beyond all controls in two of four models), and a given shift acts the same way on items of every hypothesis type, so it locates the answer."

## Task 2. The three headline claims

**Claim 1: supported.** All eight cells (four models, balanced pool and CLadder) are negative with intervals excluding zero. The same holds in all five debiased-pool cells, 7B included.

**Claim 2: not supported as worded.**
- "As accurate as a lookup": holds for 1.5B, Gemma, 3B (differences -0.013, -0.002, 0.000) and within the interval for 7B (+0.018).
- "No more": contradicted for 7B (W1).
- "Pooled ... at chance": holds for all five models by the AUROC criterion, with the wording problem in W7.

**Claim 3: supported as an absence claim.**
- (i) The four comparisons beyond their controls are named and called undecided. None can survive correction.
- (ii) The aggregate is stated. Its explanation is over-specific (W3).
- (iii) The alignment file supports the answer-position reading in three of four models, on the debiased pool. It does not support "adjacency" as such, and it leaves the end-of-hypothesis ranking unexplained, including the Qwen 3B case.
- (iv) Adjacency, four models: at the end of the hypothesis nothing is beyond controls except Qwen 0.5B's uniform push towards No; at the answer position there is a rank-one answer shift. Nothing here contradicts the headline. The page's summary of it needs W4 and W20.
- "At the answer position that is the answers' dependence on adjacency read back, not a use of validity" is not a fair one-sentence statement for every model it covers: not Qwen 0.5B, not the original pool, and "adjacency" is inference.
- "Moves the answer by 11%, 6% and 1%" is the correct ratio and a misleading description (W4).

## Task 3. The 7B statement

| Required element | Present | Correct |
|---|---|---|
| Pooled AUROC at chance | yes, 0.514 [0.490, 0.536] | yes |
| Within-string AUROC 0.549 | yes, with interval | yes; recomputed 0.5491 [0.531, 0.568] |
| Valid item ranked higher in 53.6% of premises | yes | yes; recomputed 0.5358, interval [0.510, 0.562] |
| 4-bit quantisation as a limitation | yes, Run line and first limitation | yes |

Stronger than the files: "follow validity" without "on average" (W8); claim 2's "no more" (W1); the audit-11 clause (W2).

Weaker than the files: the page omits that the 7B answers follow validity beyond adjacency (0.589), the eleventh audit's main point.

Not on the page: the 7B forward pass ran in bfloat16 with batch noise of SD 0.17 in the logits (eleventh audit F6), which can only attenuate.

## Task 4. C1 to C9

| Item | Resolved | Correctly |
|---|---|---|
| C1 "six independent audits" | yes, "Twelve" | number needs settling (W6) |
| C2 "three of the eight cells" | yes, "four" | yes |
| C3 "pre-specified contrast" | yes, removed and entered in the retraction table | yes |
| C4 "in Qwen2.5-0.5B ... One model so far" | yes, four models | partly: two nulls are presented as effects and the 0.5B control comparison is dropped (W4) |
| C5 "stays within the range of matched controls" | yes, replaced by the 32-comparison count and retracted | yes, apart from "true of one model only" (W16) |
| C6 answer-position sentence | yes, rewritten for four models | partly: R² dropped, "alike for every hypothesis type" ambiguous (W20) |
| C7 figure and text inconsistent | yes, text covers all four columns | Gemma column annotation conflicts with "responds to nothing" (W12) |
| C8 limitations | yes: four models; the final-token-only clause removed; 7B result in place of "running"; non-convergence and multiplicity added | "most probe fits" overstated (W10); "one layer per model" loose (W18) |
| C9 retraction table | yes, three rows added | yes |

## Task 5. The owner's instruction for claim 3

| Instruction | Carried out |
|---|---|
| Keep the headline | yes, unchanged |
| Say that no intervention result survives correction for multiple comparisons | yes, for the four comparisons, which is the form the files support |
| Name all four beyond-control arms as undecided | yes |
| 4 of 32, 1.5 expected; 3 of 16 on the debiased pool | yes (1.52; 0.76) |

One thing the owner should know. Taken literally, "no intervention result survives correction" is not true of the aggregate: the mean-rank result has p 0.0011 (both directions together 0.0003), which survives any plausible correction. The page handles this correctly by limiting "None survives" to the four and stating the aggregate.

Nothing on the page contradicts the instruction. What goes beyond the evidence is W3, W4 and W5.

## Task 6. Omissions

In order of weight:
1. The 7B probe's increment over the strongest text-only classifier (W1).
2. The Qwen 0.5B adjacency contrast at the end of the hypothesis is outside its 20 controls (W4).
3. Responding-site counts for the steering comparisons (W9).
4. The answer-position account fails in Qwen 0.5B and was not measured on the original pool (W3).
5. 7B: against validity for direct-causation hypotheses; with validity beyond adjacency (W8).
6. 21 premises behind the 0.895 interval (W13).

Checked and found not to need a mention: SAE ablation, patching by template, cross-model comparison, behaviour on the original item sets. None conflicts with a sentence on the page.

Figures: fig5 and fig13 agree with the text; fig12's Gemma column does not (W12).

## Task 7. Cuts

The page is 1,164 words by `wc`. The fixes above add about 70. To land near 950, cut about 280:

| Cut | Words saved |
|---|---|
| Retraction row 13: replace the parenthetical list of bugs by "Eight bugs that reached a result or a run" | 30 |
| Move retraction rows 8 ("alphabetical order") and 9 ("splitting the class difference") to the report; neither sentence was ever on this page | 40 |
| Move retraction row 11 to the report (a draft wording; the caveat under claim 3 already says it) | 35 |
| Move retraction row 12 to the report (its content is claim 2's caveat and the audit-11 clause) | 28 |
| Merge rows 2+3 (audit 2), 4+5 (audit 3), 6+7 (adjacency experiments) | 20 |
| "The four, all undecided": one clause per case, no second parenthesis | 25 |
| Claim 1 "Checked by": "audits 1 and 6" | 15 |
| Claim 2 "Checked by": as in W2, short form | 10 |
| Drop "No intervention was run on the 7B model." from claim 3 (it is in the first limitation) | 9 |
| Drop "20 controls per comparison and 32 comparisons without adjustment" from the second limitation (it is in claim 3's number line) | 10 |
| Move the fifth limitation (3B CLadder draw) to the report; no 3B CLadder number is on the page | 17 |
| Move "Probe accuracies are three-seed ensembles" to the report | 5 |
| Drop the two "*Figure:* `figNN.png`" labels where the image follows directly; keep "(four of the eight cells)" | 8 |
| Drop '"at chance" holds for the pooled items only' if the headline says "ranked across all items at once" | 10 |
| Drop "(report 4.1)" | 2 |
| Total | about 265 |

Sentences that must stay for the page not to mislead: the 7B within-string sentence, the four named comparisons, the aggregate mean-rank sentence with its corrected explanation, '"No intervention shows" is not "unused"', and the third limitation (post hoc analyses).

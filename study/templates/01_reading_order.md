# Reading order: one week, five sessions

**How to use this file.** Each session takes 60 to 90 minutes. Read the items in the order listed. Section names are copied exactly from the files, so you can search for them. Every result number below is filled in from `results/*.json` when this file is built, so it always matches the report. Before each session, write one sentence saying what you expect to find. After it, do the self-test without looking.

**Four words you need from the first minute.**

- **Activation**: a vector of numbers the network computes for one token at one layer.
- **Probe**: a small classifier trained to predict a label from an activation. A *linear* probe is a single weighted sum.
- **Gold label**: the benchmark's correct answer for an item (valid or invalid; yes or no).
- **Baseline**: a simpler predictor of the same label that the probe is compared with. A *text-only* baseline sees the prompt text and no model.

**One warning about the current state of the documents.** Some tables and figures in `report/report.md` already contain rows or panels for Gemma-2-2B interventions and for Qwen2.5-3B. The report's text does not discuss them yet. Skip those rows this week. The topics that are still open are listed at the end of this file.

---

**Before session 1 (ten minutes):** read `RESULTS.md` once. It is the destination; the sessions are the route.

## Session 1. The question, the models, and what "at chance" means

**Read.**

1. `EXPLAINER.md`, "Start here" (all eight steps of "The story, in the order it happened").
2. `EXPLAINER.md`, "Pipeline".
3. `EXPLAINER.md`, "Concept primer", these entries: "Token, prompt, chat template"; "Logit and answer-token scoring"; "Residual stream"; "Activation cache"; "Logit lens"; "Confidence interval and the (cluster) bootstrap"; "Accuracy, macro-F1 and AUROC"; "Answer bias versus chance".
4. `report/report.md`, "1. Question and summary".
5. `report/report.md`, "2. Setup": "2.1 Benchmarks" and "2.2 Models, hardware and the cache" in full; skim "2.3 Analyses".
6. `report/report.md`, "3.1 Behaviour" (the first two tables and the paragraph that starts "Reading."), then "3.2 Logit lens".

**Ideas to come away with.**

- *The models do not solve the task.* A model can score 50% because its answers carry no information, or because they carry some but lean to one answer. Three measures separate the two: the Yes-rate, the AUROC of the answer score (the chance that a valid item gets a higher Yes-score than an invalid one; 0.5 means none), and accuracy after moving the threshold. Qwen2.5-0.5B says Yes to {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|pct}} of test items, yet its AUROC there is {{answer_bias.json|qwen0.5b/c2c_test/auroc|ci}}. After the variables are renamed it is {{answer_bias.json|qwen0.5b/c2c_test_renamed/auroc|ci}}, and on the debiased pool {{answer_bias.json|qwen0.5b/c2c_debiased/auroc|ci}}.
- *Accuracy misleads on an imbalanced set.* Only {{benchmark_baselines.json|corr2cause_description/test/frac_valid|pct1}} of Corr2Cause test items are valid. Answering "No" every time scores {{benchmark_baselines.json|corr2cause_official_splits/test/majority/accuracy|est}} accuracy and {{benchmark_baselines.json|corr2cause_official_splits/test/majority/macro_f1|est}} macro-F1 (the average of the F1 score of each class). That is why the report reads macro-F1 and AUROC.
- *Intervals are wide for a reason.* The 1,162 test items come from {{benchmark_baselines.json|corr2cause_description/test/n_distinct_premises|int}} distinct premises. Items that share a premise are not independent, so the bootstrap resamples whole premises ("cluster bootstrap").

**Figures.**

- `figures/fig1_behaviour.png`. Left panel: check that every macro-F1 bar sits at or near the dashed "always majority" line. Middle panel: check that the bars for Qwen2.5-1.5B and Gemma-2-2B are *below* the dashed line. Ask yourself why saying Yes more often lowers accuracy here.
- `figures/fig2_logit_lens.png`. In each panel, check that the four curves move together. A model that tracked the truth would separate the two colours.
- `figures/fig2b_lens_auroc.png`. Check that the curves for the three local models stay close to the dashed 0.5 line, and that the CLadder panel is flatter than the Corr2Cause panel.

**Self-test.**

1. A model answers "No" to every item of a set in which {{benchmark_baselines.json|corr2cause_description/test/frac_valid|pct1}} of items are valid. What is its accuracy? Can its AUROC still be above 0.5, and what would that mean?
2. Why does the report resample premises, not items, when it computes a confidence interval?

---

## Session 2. Probes, their controls, and the first retraction

**Read.**

1. `EXPLAINER.md`, "Concept primer": "Linear probe"; "Held-out evaluation, cross-validation and grouped folds"; "Baselines and controls for probes"; "Control task and selectivity (Hewitt & Liang, 2019)"; "Partial-input baselines (hypothesis-only, premise-only)".
2. `EXPLAINER.md`, "Decision log": D7, D11, D17, D18, D19.
3. `report/report.md`, "3.3 Probes and controls".
4. `report/report.md`, "3.4 Gold label vs model output: is anything represented beyond the text?". Read the first two tables and everything from "Result: the audited criterion is not met" to the end of the section. Skim the long tables in between.
5. `report/report.md`, "3.5 What the text alone gives away".
6. `report/benchmark_text_baselines.md`: under "Corr2Cause", the parts "Structure of the released splits" and "Text-only baselines on the official splits"; then "CLadder"; "What is and is not new here"; "Limits of this note".
7. `notes/log.md`: the "Probes." paragraph under "Tier 1 expectations (written Thu ~19:00, before any behavioural or probe result)"; "23:00 qwen1.5b Tier 1 first numbers and a post-hoc robustness analysis"; "00:05 Independent Tier 1 audit", findings 1 to 5 and the closing paragraph.

**Ideas to come away with.**

- *A high probe score proved nothing here.* On the balanced pool the probes score {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} (Qwen2.5-0.5B), {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} (Qwen2.5-1.5B) and {{confound_gemma2b_c2c_probe.json|all_items/gold_probe/accuracy|est}} (Gemma-2-2B). A classifier that sees only the hypothesis sentence scores {{confound_qwen1.5b_c2c_probe.json|all_items/hyp_prior/accuracy|est}} on the same items. A classifier on parsed text features scores {{confound_qwen1.5b_c2c_probe.json|all_items/structural/accuracy|est}}.
- *Why the hypothesis alone predicts the label.* The benchmark names variables so that causes get earlier letters. Of {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n|int}} distinct hypothesis sentences in the train split, {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n_never_valid|int}} are never valid. This is a statement about what a classifier that never sees the premise scores. It is not a statement about the benchmark's purpose.
- *A criterion can be passed for the wrong reason.* The criterion written before the experiment compared the probe with weak baselines on items the model got wrong. The audited criterion compares the probe with the strongest text-only baseline on all report-half items. It is not met in any cell.
- *CLadder tells the same story by another route.* A classifier on the quoted numbers and the polarity of the question scores {{benchmark_baselines.json|cladder_full_v1.5/baselines/numeric_polarity/accuracy|ci}}. The probes score {{confound_qwen0.5b_cladder.json|all_items/gold_probe/accuracy|est}} to {{confound_gemma2b_cladder.json|all_items/gold_probe/accuracy|est}}.

**Figures.**

- `figures/fig3_probes_c2c_probe.png`. Find the solid blue line (gold probe). Check that it is never clearly above the red dash-dot line (hypothesis-string prior) and always below the green one (parsed features). Check that the dotted grey line (shuffled labels) sits at 0.5 and that the dashed yellow line (control task) runs not far below the blue one. Note that the Qwen2.5-0.5B panel has no orange line: its answers do not vary, so there is nothing for a model-answer probe to predict.
- `figures/fig3_probes_cladder.png`. Check that the dashed yellow line is *above* the blue one. That is negative selectivity.
- `figures/fig4_selectivity_positions_c2c_probe.png` and `figures/fig4_selectivity_positions_cladder.png`. Left panels: note the sign of selectivity for each benchmark. Right panels: find the line for "end of premise / context". On Corr2Cause it should sit at chance. On CLadder it does not sit far below the other two; ask what that means for a position at which the question has not been read.
- `figures/fig5_confound_c2c_probe.png` and `figures/fig5_confound_cladder.png`. In each of the three panels, compare the blue bar (gold probe) with the green bar (parsed features) for every model. The claim needs blue above green by more than the whiskers. Check that it never is.

**Self-test.**

1. Name three controls that the original-pool probes passed, and the one comparison they failed. Say what each control rules out.
2. In one sentence each: what was the pre-registered criterion, and why can it be passed by a scorer that knows nothing about validity?

---

## Session 3. The debiased pool (first half of the audited core)

**Read.**

1. `EXPLAINER.md`, "Concept primer": "Target encoding, and how leave-one-out leaks the label"; "Debiasing an evaluation set"; "Confound controls versus reference baselines"; "Incremental predictive value"; "Stratified analysis".
2. `EXPLAINER.md`, "Decision log": D21 and D25.
3. `report/benchmark_text_baselines.md`, "Label-balanced pools (as used for probing)".
4. `report/report.md`, "3.6 A fairer test: the debiased Corr2Cause pool". Read "3.6.6 What this supports" first. Then read from the top of 3.6 through "3.6.1 Post-hoc reading: a ladder of baselines in two groups", "3.6.2 Does the probe add predictive power beyond the text models?", "3.6.3 Which parsed properties can be read from the activations?", "3.6.4 Is the probe's signal carried by one property, and how much is shared with text?" and "3.6.5 Do the model's answers express it?".
5. `notes/log.md`, in this order: "Fri 2026-11-13 10:45 — text baselines on the benchmarks (Step 2); debiased pool revised"; "Fri 2026-11-13 13:05 — debiased pool, probes for the two Qwen models (Step 3)"; "13:20 Result of the follow-up"; "14:03 Results of the extended analysis"; "Fri 2026-11-13 14:50 — Third independent audit"; "15:05 Revised debiased-pool analysis (after the third audit): my own numbers". Watch how the reading of the same number changes at each step.
6. `notes/audit_debiased_raw.md`: findings 1 to 5, then the last section, "The claim in the wording the reviewer would allow".

**Ideas to come away with.**

- *What the pool removes.* Every premise contributes one valid and one invalid hypothesis. Every hypothesis sentence is half valid in every fold. So a classifier that sees only the hypothesis, or only the premise, scores 0.50 by construction. Only the *combination* can predict the label.
- *A probe score needs a ladder, not one baseline.* For Qwen2.5-1.5B the probe scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}. The strongest control that connects nothing in the premise to the queried variables scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|est}}. A per-template lookup of the premise statements that name the queried pair scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|est}}. A hand-written parse scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}. For Qwen2.5-0.5B the final-token probe ({{probe_content_qwen0.5b_c2c_debiased.json|baselines/gold_probe/accuracy|est}}) is not above the no-binding controls.
- *"Does the probe know something the text model does not?" is a separate question.* Stacked with the hand parse, the probe adds {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_hand_parse/gain_accuracy|signedci}} accuracy. Stacked with the strongest text-only model it adds {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}}.
- *An average can hide opposite tendencies.* Pooled, the 1.5B model's answers have AUROC {{probe_content_qwen1.5b_c2c_debiased.json|answers/pooled_auroc_for_gold|ci}}. Within "X directly causes Y" hypotheses it is {{probe_content_qwen1.5b_c2c_debiased.json|answers/by_template/parent/within_string_auroc_for_gold|ci}}.

**Figures.** There is no figure for this section yet. The evidence is in five tables inside report Section 3.6 (also saved as `report/tables/debiased_ladder.md`, `debiased_incremental.md`, `debiased_properties.md`, `debiased_strata.md`, `debiased_answers.md`). Make your own figure: draw a vertical axis from 0.5 to 0.9, mark every scorer of the `debiased_ladder` table for Qwen2.5-1.5B, and mark the probe. Check three things on your drawing: the probe is above all of group (a); it sits among the lower rungs of group (b); the rungs of group (b) differ a lot among themselves, so "the text baseline" is not one number.

**Self-test.**

1. On this pool, why does "the probe beats a hypothesis-only classifier and whole-prompt TF-IDF" tell you nothing? What kind of control does tell you something?
2. The probe scores below the hand parse and still adds accuracy when stacked with it. Explain how both can be true, and why the gain disappears against a stronger text model.

---

## Session 4. Interventions: steering and patching (second half of the audited core)

**Read.**

1. `EXPLAINER.md`, "Concept primer": "Decodable is not the same as used"; "Steering vector", including the paragraph "Three traps, all found by the second audit"; "Directional ablation"; "Minimal pair"; "Activation patching"; "Random-donor control for patching"; "Attention heads and MLPs"; "Attribution patching".
2. `EXPLAINER.md`, "Decision log": D14, D15, D16, D24.
3. `report/report.md`, "4.1 Steering: what does adding a decodable direction do to the answer?". Read "Design", the matched-controls table, "What the data show" and "What may and may not be concluded". Skim the two long tables.
4. `report/report.md`, "4.2 Patching: where does a change of variable reach the answer position?" and "4.3 Cross-model comparison".
5. `notes/log.md`: "19:40 Tier 2 code written and smoke-tested; expectations"; "Fri 2026-11-13 11:50 — Independent Tier 2 audit"; "11:58 Expectations before the matched-control steering run"; "14:10 Matched-control steering runs".
6. `notes/audit_tier2_raw.md`: findings 1 to 5, 8, 10, 14, 15 and 18, then "Wording the reviewer would allow".

**Ideas to come away with.**

- *The unit of a push matters.* Vectors are scaled by s, the root-mean-square distance of an activation from the layer's mean activation. Along the gold probe's weight direction, one unit of s is about {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|int}} natural standard deviations for Qwen2.5-1.5B on the original pool and about {{steering_analysis_qwen1.5b_c2c_debiased.json|directions/gold_probe/alpha_1_in_own_sd|int}} on the debiased pool. "We pushed hard and little happened" and "we pushed a realistic amount" are different statements.
- *A control must be of the same kind as the thing tested.* The gold probe direction is compared with 20 probe directions fitted to shuffled labels. The share of those controls with a smaller slope, in absolute size, is {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} (original pool, 0.5B and 1.5B) and {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} (debiased pool). The report's wording is "within the range of label-free directions of the same kind". With twenty controls and one site the experiment "neither shows nor excludes a causal role".
- *Patching found a route, not a decision.* The models do not prefer the valid member of a pair: the mean difference in answer score between the two prompts is {{patching_qwen0.5b.json|kinds/hyp/behaviour_on_pairs/mean_delta|signedci}} for Qwen2.5-0.5B. So the heat-maps show where *any* prompt difference reaches the final token.

**Figures.**

- `figures/fig6_steering_c2c_probe.png`, `figures/fig6_steering_c2c_debiased.png`, `figures/fig6_steering_cladder.png`. Top panels: find the blue line (gold probe weights). Check its tilt: in the debiased-pool figure it goes *down* to the right in both models, a negative slope. Check that the grey dashed lines (three random directions) cover a similar range. Bottom panels: check that the blue bar is among the smallest on the log scale. Remember what is *not* in this figure: the 20 matched controls. They are in the table `steering_controls`.
- `figures/fig7_patching_hyp.png` and `figures/fig7_patching_prem.png`. Top row: follow the blue from the "diff tokens" row (early layers) to the "final" row (late layers). In the hyp figure, look for a short stop in the "hypothesis" and "template tail" rows in between. Bottom row: read the n in each panel title ({{patching_qwen0.5b.json|kinds/hyp/n_pairs_with_control|int}} and {{patching_qwen1.5b.json|kinds/hyp/n_pairs_with_control|int}} for the two Qwen models) and note that the row is not blank. That is why it is not used as a control.
- `figures/fig8_attribution_validation.png`. Check that the blue points (top ten components) lie on the diagonal. Then look at the grey points near the origin and ask whether they could tell you that no small component matters.
- `figures/fig9_cross_model.png`. Middle panel: check that every point is below the dashed zero line. Note how few models there are before reading any trend into it.

**Self-test.**

1. State the steering result in the audited wording. Then give two reasons why it does not show that the decoded information is unused.
2. Why does a recovery of 1.0 at the final token in late layers say nothing about where a causal judgement is made?

---

## Session 5. Rehearsal: audits, limits, and the defence

**Read.**

1. `report/report.md`, the paragraph "Pre-registration status." at the end of Section 1.
2. `report/report.md`, "6. Audits" (the three tables of findings and consequences) and "7. Limitations".
3. `EXPLAINER.md`, "Questions a professor would ask": "Tier 1 — behaviour, logit lens, probes" and "Tier 2 — steering, patching, cross-model". Cover each answer and say yours aloud first.
4. `study/02_questions.md`, levels 3 and 4.
5. `study/03_exercises.md`. Write each prediction before you run the command. Then read `study/04_exercise_answers.md`.
6. `YOUR_TURN.md`. Read both experiments. Experiment 1 needs no GPU; do it only when no other job is running on the laptop.
7. `report/report.md`, "8. Interpretation and what I'd do next". It is empty on purpose. Write half a page of your own.

**Ideas to come away with.**

- *Each audit removed a claim, and each time for the same reason.* A number was compared with a baseline, a control or a test that could not tell the two readings apart. Be able to name the three cases: weak text baselines on the original pool; controls at 0.50 by construction on the debiased pool; an incremental-value test that could not detect a two-point gain.
- *Know which analyses were fixed in advance and which were not.* The pre-registered parts are the wrong-items criterion, grouped folds, split-half selection and the cluster bootstrap. Everything in report Sections 3.6.1 to 3.6.6 and the steering controls is post hoc and labelled so.
- *Volunteer the limits.* All models are at chance. Probes are linear. Steering covers one layer and one token. Intervals are unadjusted across many cells.

**Figures.** Go through all files in `figures/` at thirty seconds each. For each one say aloud: what the two axes are, the one thing it shows, and the one thing it does not show. Check yourself against `EXPLAINER.md`, "How to read each figure".

**Self-test.**

1. For each of the three audits, name one claim that did not survive and the wording that replaced it.
2. Which single result in the project would you least like to be asked about, and what is your honest two-sentence answer?

---

## If you only have an hour

1. **10 minutes.** `EXPLAINER.md`, "Start here". This is the whole story in eight steps.
2. **10 minutes.** `report/report.md`, "1. Question and summary", including "Pre-registration status."
3. **15 minutes.** `report/report.md`, "3.6.6 What this supports". Then the first paragraph of "3.6 A fairer test: the debiased Corr2Cause pool" and the Qwen2.5-1.5B rows of the ladder table in 3.6.1.
4. **10 minutes.** `report/report.md`, in 4.1 the list "What may and may not be concluded"; in 4.2 the paragraph "What the data show."
5. **10 minutes.** `report/report.md`, "6. Audits": read only the three two-column tables.
6. **5 minutes.** Four figures: `fig3_probes_c2c_probe.png` (blue against the red and green lines), `fig5_confound_c2c_probe.png` (blue against green), `fig6_steering_c2c_debiased.png` (tilt of the blue line), `fig7_patching_hyp.png` (blue moving from the top row to the bottom row).

After that hour you should be able to say these five sentences from memory.

1. The models answer at or near chance on both benchmarks.
2. On the original item sets the probes score high, and text-only classifiers score as high or higher, so the probe score proves nothing there.
3. On the debiased pool, where no single input predicts the label, the 1.5B model's final-token probe scores {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|est}}: above controls that connect nothing to the queried variables, level with a lookup of the statements about the queried pair, below a hand-written parse.
4. Steering along the gold-probe direction moves the answer by an amount within the range of the control directions; this neither shows nor excludes a causal role.
5. Patching locates where a change in the prompt reaches the answer token, not where a judgement is made.

---

## Session 6 (added 13 November). Round 2: how an intervention result goes wrong, twice

*Read:* report section 4.2 from "Label-preserving pairs"; section 4.1 from "Steering at the end of the hypothesis"; section 4.4 in full; Appendix C (SAE); then `notes/log.md` entries of 13 November from 16:35 to the end, in order. EXPLAINER primer entries "Label-preserving minimal pair", "Where you intervene matters", "Observational versus interventional evidence for a feature"; questions 36 to 43.

*Ideas to come away with:*
- An intervention that "works" at a place where the output is already a linear function of the activation shows where the output is, not what it is computed from.
- A control has to be able to fail. For each control in section 4.4, say what it would have shown if the claim were false.
- Averages hide effects of opposite sign. Look at each hypothesis type before any summary number.

*Figures:* `fig10_patching_label_preserving.png` (compare the two |D| values in the legends, not the curves); `fig11_adjacency_final.png` and `fig12_adjacency_end_of_hypothesis.png` (are the dots outside the grey bars? In which panels? Does the middle row follow the top row?).

*Self-test:* Explain to someone why the first summary of the adjacency test ("a sixth of the dependence, beyond all controls") was wrong, without looking. Then answer study questions 45 to 47.

## Session 7 (added 14 November). One result that beats its controls: what is it worth?

*Read:* report section 4.1 from "Qwen2.5-1.5B and Qwen2.5-3B at both tokens" to "Three cautions on the table"; section 4.4, points 10 to 14 and the transfer summary table; `notes/audit_round2_runs_raw.md`, the part headed "Q1" (it answers four questions in order: is it real, is it one arm among many, does it replicate, does it survive the shuffle); `notes/claim3_proposed_rewording.md`; `notes/log.md`, the three entries of Saturday 14 November. EXPLAINER primer entries "First among 20 controls, and how many times you looked" and "What a shuffled control keeps and what it destroys"; questions 44 to 46.

*Ideas to come away with:*
- "Beyond all 20 controls" is p = 1/21 for one comparison. Count the comparisons before reading any of them.
- A result can be perfectly consistent across items and still be uncertain: here the uncertainty is in the draw of the direction, not in the items.
- A shuffled control tests against whatever the shuffle destroys. Know what else it destroys besides the thing you care about.
- More of the same control sharpens a percentile. An independent draw of the direction is a replication.
- Write the test and the decision rule down before the run that could embarrass you.

*Self-test:* Without looking, say how many first places among 12 comparisons chance gives, and what the probability of two or more is. Then do exercise 6, and answer study questions 49 to 52.

## Still to be added

- Gemma 2B: matched steering controls, label-preserving pairs, the adjacency test
- Qwen 7B behaviour and its probe comparison on the debiased pool
- The settle run for the Qwen 3B steering arm, if it is run

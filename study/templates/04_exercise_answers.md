# Answers to the five exercises

Read this only after you have written your predictions in `study/03_exercises.md`. For each exercise: the result, what it means, and the common wrong intuition it corrects. Every number is filled in from `results/*.json` when this file is built.

---

## Exercise 1. A classifier that never sees the premise, on the official test split

**Result.** AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}. Accuracy {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/accuracy|ci}}, against {{benchmark_baselines.json|corr2cause_official_splits/test/majority/accuracy|est}} for the always-"invalid" rule. It predicts "valid" for {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/frac_predicted_positive|pct}} of items, and its F1 on the valid class is {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/positive_f1|ci}} (the always-"invalid" rule has {{benchmark_baselines.json|corr2cause_official_splits/test/majority/positive_f1|est}}).

**What it means.** The hypothesis sentence alone ranks valid above invalid items well. Yet the accuracy is close to the majority rate. Both are true at once because few sentences are valid more often than not, so a classifier that must commit to "valid" or "invalid" still says "invalid" most of the time. The same information looks large on a balanced set and small on an imbalanced one when measured by accuracy. AUROC does not have that problem. The cause is how the benchmark names variables: causes get earlier letters, and {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n_never_valid|int}} of the {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n|int}} hypothesis sentences are never valid in the train split.

**The wrong intuitions.**

- *"Without the premise it cannot know anything, so AUROC is 0.5."* The premise is needed to decide an item. It is not needed to do better than chance on average.
- *"It scored far above chance on the balanced pool, so it will be far above the majority rule on the test split."* Accuracy depends on the base rate. Always compare with the right reference (chance on a balanced set, the majority rate on an imbalanced one) and report a threshold-free measure next to it.

**How to say it in front of a professor.** This is a measurement of what a classifier that never sees the premise scores. It matters for anything trained on in-distribution items, such as a probe. A model tested zero-shot has not seen these label statistics.

**Where it is in the repository.** `report/benchmark_text_baselines.md`, "Text-only baselines on the official splits"; `notes/log.md`, "Fri 2026-11-13 10:45", item 1, where the author's own prediction was right about the AUROC and about the accuracy.

---

## Exercise 2. A probe on a network that was never trained

**Result.** (a) {{confound_qwen1.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}}. That is not chance: the shuffled-label probes have mean {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A0_single_input/probe_on_shuffled_labels_null/mean|num}} and standard deviation {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A0_single_input/probe_on_shuffled_labels_null/sd|num}}. (b) Adjacency of the queried pair is readable from the untrained network at AUROC {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init_same_layer_auroc|ci}}. From the trained model it is {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained/auroc|ci}}.

**What it means.** An untrained network is not a blank. It is a fixed random function of the prompt, and its final-token vector is a random mixture of the tokens before it. A linear probe on that vector behaves like a text classifier with random features. On this pool it reaches about what one model per hypothesis template on premise-level counts reaches ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_counts/accuracy|est}}). Some properties are fully present in the input: "the premise has six variables" is read from the untrained network at AUROC {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/six_variables/at_gold_probe_layer/random_init_same_layer_auroc|est}}. So the bar for "the model has learned to compute something" is the random-init score, not 0.50. For Qwen2.5-1.5B the trained probe clears it by {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}}. For Qwen2.5-0.5B the gap is {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}}, which the third audit judged as no evidence at the final token.

**The wrong intuitions.**

- *"Untrained means chance."* Chance is what a probe gets when the *labels* carry no information (the shuffled-label control). The random-init control asks a different question: what do the input and the architecture give away without any learning?
- *"If the random-init probe is above chance, the pipeline must leak."* Not here. The shuffled-label control sits at chance, and the third audit reproduced the random-init scores.

**A caution.** Only one random initialisation was run. The report says so where it uses the number.

**Where it is in the repository.** `report/report.md`, table under "3.6 A fairer test: the debiased Corr2Cause pool", and "3.6.3 Which parsed properties can be read from the activations?"; `notes/audit_debiased_raw.md`, finding 2.

---

## Exercise 3. Where does the probe sit among text-only models?

**Result.** The probe scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}.

| Text-only model | Its accuracy | Probe minus this (paired) | Verdict |
|---|---|---|---|
| A. Per-template bag-of-words of the premise | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|ci}} | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/probe_minus_this_accuracy|signedci}} | probe clearly above |
| B. Per-template lookup of statements naming the pair | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|ci}} | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/probe_minus_this_accuracy|signedci}} | level |
| C. Hand-written parse | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|ci}} | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/probe_minus_this_accuracy|signedci}} | probe clearly below |
| D. Parse stacked with per-sentence 4-grams | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/strongest_text_only_parse_plus_4grams/accuracy|ci}} | {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/strongest_text_only_parse_plus_4grams/probe_minus_this_accuracy|signedci}} | probe clearly below |

**What it means.** The probe sits where binding starts. It is above every model that connects nothing in the premise to the queried variables. It is level with a lookup of what the premise says about the queried pair. It is below a parse and below a stronger text-only model. The audited wording is that the 1.5B model's final-token activations hold about what such a lookup holds. "Level with" is a statement about score. It does not show that the model performs a lookup.

**The wrong intuitions.**

- *"Whole-prompt TF-IDF is at chance here, so any probe well above it shows reasoning."* On this pool every single-input control, and every model that adds up premise and hypothesis evidence separately, is at 0.50 by construction. Beating them shows that the pool was built correctly. The informative controls are interaction models such as A.
- *"A network with over a billion parameters must hold more than a lookup table."* What a linear probe can read at one token is the question, and here it is no more than a lookup's worth.
- *"The text baseline is one number."* The four text models span from {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|est}} to {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/strongest_text_only_parse_plus_4grams/accuracy|est}}. The result is the probe's place on that ladder.

**Where it is in the repository.** `report/report.md`, "3.6.1 Post-hoc reading: a ladder of baselines in two groups"; `notes/audit_debiased_raw.md`, finding 2. The author's first version of this analysis listed only controls that were at 0.50 by construction; the third audit caught it.

---

## Exercise 4. What does the probe add to a parser that is better than it?

**Result.** (a) Added to the hand parse: {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_hand_parse/gain_accuracy|signedci}} accuracy (in AUROC: {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_hand_parse/gain_auroc|signedci}}). (b) Added to the strongest text-only model: {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}}.

**What it means.** The activations hold a small amount of validity information that this particular hand parse lacks, and nothing measurable that a stronger text-only model lacks. So the gain in (a) says something about the parse's gaps, not about information that is absent from the text.

**The wrong intuitions.**

- *"The probe is worse than the parse, so it cannot add anything."* A weaker predictor adds something whenever its errors differ from the stronger one's. Here they partly do.
- *"The probe is right on {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/probe_right_when_hand_parse_wrong|pct}} of the parse's errors, so the gain should be large."* The combined model does not know which items the parse gets wrong. It also has to pay for the items where the parse is right and the probe is wrong. With scores that correlate at r = {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/pearson_r_probe_vs_hand_parse_score|num2}}, most of the information is shared.
- *"A gain over the parse means the model knows something that is not in the text."* The gain disappears against model D. A claim of that kind is only as strong as the strongest text model tried.

**The lesson about method.** This number was reported three times. The first version was not nested and was withdrawn. The second was nested but added the probe as one column to gradient boosting; it gave {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/gbm_plus_one_column_insensitive/gain_accuracy|signedci}} and was read as "no gain". The third audit fed that test a made-up column of known value and showed that it reports about zero regardless. The version above is nested and recovers the known value. Before you trust a test that says "no difference", check that it can detect a difference when there is one.

**Where it is in the repository.** `report/report.md`, "3.6.2 Does the probe add predictive power beyond the text models?"; `notes/log.md`, "13:20", "14:03" (item B) and "14:50" (finding 1); `EXPLAINER.md`, "Incremental predictive value".

---

## Exercise 5. Pushing the model along the direction that decodes the right answer

**Result.** (a) The slope is negative: {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/slope_per_s|signedci}} logits per unit of s. Pushing towards "valid" makes the model lean towards No. The share of items with a positive slope is {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/frac_items_positive_slope|pct}}. (b) {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} of the control directions have a smaller absolute slope. The gold slope is {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_s|num2}} times the controls' median and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_max_abs_slope_per_s|num2}} of their maximum. (c) Yes. Per natural standard deviation, {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_own_sd|pct}} of the controls are smaller, and the gold direction is {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_own_sd|num2}} times their median.

**What it means.** Per unit of s, the gold direction moves the answer by an amount within the range of the control directions, which carry no label information. With twenty controls and one site, this neither shows nor excludes a causal role. (Added after the eighth audit: for probe directions the comparison per natural standard deviation is not matched. The real probe's direction has a larger natural spread than probes fitted to shuffled labels whatever it does causally, so a high percentile in that unit can come from the spread alone. Read (c) as arithmetic, not as evidence.) The answer to (c) is the one case, out of four, in which the gold direction stands out, and it is on the original pool, where the gold label is confounded with the hypothesis sentence. On the debiased pool the same comparison gives {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} per unit of s and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_own_sd|pct}} per natural standard deviation for this model. Keep the scale in mind too: one unit of s is about {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|int}} natural standard deviations along this direction. Moving an item across the whole natural gap between valid and invalid items would change the answer score by {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/effect_of_natural_class_gap_on_logit_diff|signed}} logits.

**The wrong intuitions.**

- *"A direction that decodes the label well is a direction the model uses, so pushing along it should move the answer most, and towards Yes."* A probe's weights say which combination of activations to *read*. They point into directions where activations vary little. Later layers need not respond to those directions. How well a direction decodes and how much the output responds to it are separate properties.
- *"The effect has the same sign for every item, so it must mean something."* A large push along almost any direction shifts the answer the same way for every item. Of the shuffled-label directions in this run, {{steering_controls_qwen1.5b_c2c_probe.json|families/shuffled_probe/frac_sign_consistent_directions|pct}} are sign-consistent in the same sense.
- *"Then the information is unused."* That does not follow either. The probe's weights and the difference of class means are two views of one discriminant (cosine {{steering_analysis_qwen1.5b_c2c_probe.json|geometry/cos_covariance_times_probe_vs_diffmeans|num2}} between the covariance times the probe weights and the difference of means). Only one layer and one token were tested.

**The lesson about method.** The author's first table showed pushes of one sign only, compared with three random directions, and concluded that the direction was "inert". Both signs, a control of the same kind, and two units were needed before any sentence about this experiment could be defended.

**Where it is in the repository.** `report/report.md`, "4.1 Steering: what does adding a decodable direction do to the answer?", points 1 to 3 under "What the data show" and the list "What may and may not be concluded"; `notes/log.md`, "11:58 Expectations before the matched-control steering run" and "14:10 Matched-control steering runs"; `notes/audit_tier2_raw.md`, finding 1.

---

## Exercise 6. How many "beyond all controls" does chance give?

**Result.** (a) {{steering_controls_multiplicity.json|summary/debiased_pool/expected_rank_1_by_chance|num2}}: each comparison has 1 chance in 21, and there are {{steering_controls_multiplicity.json|summary/debiased_pool/n_comparisons|int}}. (b) {{steering_controls_multiplicity.json|summary/debiased_pool/n_rank_1_in_absolute_slope|int}}. (c) {{steering_controls_multiplicity.json|summary/debiased_pool/p_at_least_this_many_rank_1|num}}. (d) Rank {{steering_controls_multiplicity.json|by_arm/qwen3b.c2c_debiased.final.gold_diffmeans/abs_rank_1_is_largest|int}} of 21. (e) {{steering_controls_multiplicity.json|summary/all/gold_diffmeans_slopes_positive|str}}.

**What it means.** Two first places where half of one is expected is not unusual (about one time in nine). Neither first place repeats in a neighbouring comparison. The one regularity that does repeat everywhere, the positive sign of the gold difference of class means, was noticed after looking and has a dull explanation: within a hypothesis string the gold label goes with adjacency, and the adjacency contrast of direct-causation questions pushes towards Yes in every model.

**The wrong intuitions.**

- *"It beat every control, so it is significant."* Against 20 controls the best possible result is p = 1/21. After 12 comparisons that is nothing.
- *"It is the same for all 150 items and in every block, so it is solid."* That removes one source of uncertainty, the items. The direction was fitted once and the 20 controls were drawn once; that is where the uncertainty is, and it is about two standard deviations of the control family.
- *"Then run more controls."* More controls on the same split sharpen one percentile of the same draw. A replication is a new draw of the direction: fit it on other data.

**The lesson about method.** The author's first note on this result said it went "against the plain reading of claim 3" and reached for a run with 100 controls. An auditor given only code and numbers counted the comparisons, checked the neighbours, asked what the direction was, and showed the planned remedy would not have decided anything.

**Where it is in the repository.** `report/report.md`, "4.1", from "End of the hypothesis, debiased pool: two first ranks"; `notes/audit_round2_runs_raw.md`, Q1; `notes/log.md`, Saturday 14 November; `notes/claim3_proposed_rewording.md`.

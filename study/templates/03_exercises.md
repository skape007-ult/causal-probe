# Five exercises: predict the result before you look it up

**Why do this.** The project log records an expectation before every experiment, and several of them were wrong. That is where most of the learning happened. These five exercises put you in the same position. For each one: read the setup, write a prediction and one reason, and only then run the command. The answers and what they mean are in `study/04_exercise_answers.md`. Do not open that file first.

**How to run the commands.** Run them from the repository root, the folder that contains `results/`. Each command only reads one small JSON file. It loads no model and is safe to run while other jobs are running.

**How to read the output.** Most values print as a small dictionary such as `{'est': …, 'lo': …, 'hi': …, 'n': …, 'n_groups': …}`. `est` is the estimate; `lo` and `hi` are the ends of the 95% confidence interval; `n` is the number of items; `n_groups` is the number of premises.

The numbers in the setups below are filled in from the result files when this file is built. None of them is the answer.

---

## Exercise 1. A classifier that never sees the premise, on the official test split

**Setup.** The Corr2Cause test split has 1,162 items; {{benchmark_baselines.json|corr2cause_description/test/frac_valid|pct1}} are valid. A rule that always answers "invalid" has accuracy {{benchmark_baselines.json|corr2cause_official_splits/test/majority/accuracy|est}} and AUROC 0.5. (AUROC is the chance that a valid item gets a higher score than an invalid one.)

Now take a classifier that sees only the hypothesis sentence, for example "A directly causes D.", and never the premise. Its score for a test item is the share of train-split items with exactly that sentence that were valid. On the label-balanced original pool, where chance is 0.50, this kind of classifier scores {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|est}} accuracy.

**Question.** On the official test split: (a) what is this classifier's AUROC? (b) what is its accuracy, and how far is that above the always-"invalid" rule?

**Your prediction:** AUROC ____ ; accuracy ____ ; because ____

**Look it up.** File `results/benchmark_baselines.json`. Key path: `corr2cause_official_splits` → `test` → `hypothesis_only` → `auroc`, and the same path ending in `accuracy`.

```bash
.venv/bin/python -c "import json; d=json.load(open('results/benchmark_baselines.json'))['corr2cause_official_splits']['test']['hypothesis_only']; print('auroc', d['auroc']); print('accuracy', d['accuracy'])"
```

---

## Exercise 2. A probe on a network that was never trained

**Setup.** The debiased pool is built so that a classifier that sees only the hypothesis, or only the premise, scores exactly 0.50. Probes trained on shuffled labels average {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A0_single_input/probe_on_shuffled_labels_null/mean|num}} there.

Take the Qwen2.5-1.5B architecture with random, untrained weights. Feed it the same prompts. At the final prompt token, train the same linear probe for the gold label, with the layer and the regularisation chosen in the same way as for the trained model.

**Question.** (a) What is this probe's held-out accuracy? Is it at chance? (b) From the same untrained network, how well can a probe read whether the two queried variables are adjacent in the premise's skeleton (AUROC; 0.5 is chance)?

**Your prediction:** (a) ____ ; (b) ____ ; because ____

**Look it up.** (a) File `results/confound_qwen1.5b_c2c_debiased.json`, key path `all_items` → `random_init_probe` → `accuracy`. (b) File `results/probe_content_qwen1.5b_c2c_debiased.json`, key path `properties` → `by_property` → `pair_adjacent_in_skeleton` → `at_gold_probe_layer` → `random_init_same_layer_auroc`.

```bash
.venv/bin/python -c "import json; print(json.load(open('results/confound_qwen1.5b_c2c_debiased.json'))['all_items']['random_init_probe']['accuracy'])"
```

```bash
.venv/bin/python -c "import json; print(json.load(open('results/probe_content_qwen1.5b_c2c_debiased.json'))['properties']['by_property']['pair_adjacent_in_skeleton']['at_gold_probe_layer']['random_init_same_layer_auroc'])"
```

---

## Exercise 3. Where does the probe sit among text-only models?

**Setup.** On the debiased pool, the final-token probe for the trained Qwen2.5-1.5B scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|est}} accuracy. A linear bag-of-words of the whole prompt scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A0_single_input/bag_of_words_whole_prompt_linear/accuracy|est}}, as it must on this pool.

Four text-only models, none of which uses a language model:

- **A.** One bag-of-words model of the premise per hypothesis template. It knows the type of question but connects nothing in the premise to the two variables the question names.
- **B.** One model per hypothesis template on the premise statements that name both queried variables: a lookup of what the premise says about that pair.
- **C.** A hand-written parse: gradient boosting on features such as "are the two variables correlated", "are they ever made independent", "how many conditioning sets separate them", "do they share neighbours".
- **D.** Model C stacked with a bag of premise 4-grams fitted separately for each hypothesis sentence.

**Question.** For each of A, B, C and D: is the probe clearly above it, level with it, or clearly below it? Give a number for "probe accuracy minus B".

**Your prediction:** A ____ ; B ____ (difference ____ ) ; C ____ ; D ____ ; because ____

**Look it up.** File `results/probe_content_qwen1.5b_c2c_debiased.json`. The key `baselines` → `summary` sorts the binding models into those the probe exceeds, those it cannot be told apart from, and those above it. The difference to B is at `baselines` → `B_binding_text_reference` → `template_x_statements_naming_the_pair` → `probe_minus_this_accuracy`. Model A is `baselines` → `A_no_binding_interaction` → `template_x_premise_bag_of_words`; C is `hand_parse_all_features` and D is `strongest_text_only_parse_plus_4grams`, both under `B_binding_text_reference`.

```bash
.venv/bin/python -c "import json; b=json.load(open('results/probe_content_qwen1.5b_c2c_debiased.json'))['baselines']; [print(k, v) for k, v in b['summary'].items()]; print('probe minus B', b['B_binding_text_reference']['template_x_statements_naming_the_pair']['probe_minus_this_accuracy'])"
```

---

## Exercise 4. What does the probe add to a parser that is better than it?

**Setup.** Same pool, same model. The probe scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|est}}; the hand-written parse (model C above) scores {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}. Their scores correlate at r = {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/pearson_r_probe_vs_hand_parse_score|num2}}. On the {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/n_hand_parse_wrong|int}} report-half items that the parse gets wrong, the probe is right on {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/probe_right_when_hand_parse_wrong|pct}}.

Now combine the two: a logistic regression on the parse's score and the probe's score. Both scores are produced by a cross-validation inside each training fold, so neither has seen the labels of the items it is scored on ("nested" stacking).

**Question.** (a) By how much does the combined model's accuracy exceed the parse alone? (b) The same, when the probe is added to the strongest text-only model (model D, {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/strongest_text_only_parse_plus_4grams/accuracy|est}}) instead?

**Your prediction:** (a) ____ ; (b) ____ ; because ____

**Look it up.** File `results/probe_content_qwen1.5b_c2c_debiased.json`. Key paths: `incremental_value` → `over_hand_parse` → `gain_accuracy`, and `incremental_value` → `over_strongest_text_only` → `gain_accuracy`.

```bash
.venv/bin/python -c "import json; v=json.load(open('results/probe_content_qwen1.5b_c2c_debiased.json'))['incremental_value']; print('over hand parse', v['over_hand_parse']['gain_accuracy']); print('over strongest text-only', v['over_strongest_text_only']['gain_accuracy'])"
```

---

## Exercise 5. Pushing the model along the direction that decodes the right answer

**Setup.** Qwen2.5-1.5B, original pool, layer {{steering_controls_qwen1.5b_c2c_probe.json|layer|int}}, final prompt token. The gold-label probe's weight vector is a direction in activation space. Projected onto it, report-half items separate into valid and invalid with AUROC {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/auroc_gold_report_half|num}}.

The experiment adds this direction to the residual stream, scaled to one unit of s (the root-mean-square distance of an activation from the layer's mean), once with a plus sign and once with a minus sign. The slope is the change in the Yes-minus-No answer score per unit added. A positive slope means that pushing towards "valid" makes the model lean towards Yes.

The controls are {{steering_controls_qwen1.5b_c2c_probe.json|families/shuffled_probe/abs_slope_per_s/n|int}} probe weight vectors fitted in the same way to shuffled labels. They decode nothing.

**Question.** (a) Is the gold direction's slope positive or negative? (b) What share of the control directions have a smaller slope, in absolute size, than the gold direction? (c) Does the answer to (b) change if every slope is expressed per natural standard deviation along its own direction?

**Your prediction:** (a) ____ ; (b) ____ ; (c) ____ ; because ____

**Look it up.** File `results/steering_controls_qwen1.5b_c2c_probe.json`. Key paths: `real` → `gold_probe` → `slope_per_s`; `real` → `gold_probe` → `versus` → `shuffled_probe` → `percentile_of_abs_slope_per_s` (the share of controls with a smaller absolute slope, as a fraction of 1); and the same path ending in `percentile_of_abs_slope_per_own_sd`.

```bash
.venv/bin/python -c "import json; g=json.load(open('results/steering_controls_qwen1.5b_c2c_probe.json'))['real']['gold_probe']; print('slope', g['slope_per_s']); print('share of controls smaller, per s', g['versus']['shuffled_probe']['percentile_of_abs_slope_per_s']); print('share of controls smaller, per natural SD', g['versus']['shuffled_probe']['percentile_of_abs_slope_per_own_sd'])"
```

---

When you have done all five, write down which of your predictions were wrong and, in one line each, what you had assumed. Then read `study/04_exercise_answers.md`.

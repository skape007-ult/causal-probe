# Do small LLMs represent causal answers they fail to give? A probing, steering and patching study

Technical report. Every number in a table comes from `results/*.json` via `scripts/12_tables.py`; the source file is named under each table. Figures are drawn from the same files by `scripts/06_figures.py`. For background on any method see `EXPLAINER.md`.

## 1. Question and summary

**Question.** Do small open-weight language models internally represent the correct answer to formal causal-reasoning questions they answer wrongly; is that representation used; and where is the answer formed?

**Summary of findings.**

1. *Behaviour.* Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-2-2B and Qwen2.5-3B are at or near chance on Corr2Cause and CLadder. Part of the picture is answer bias; the bias-free ranking score is at or near 0.5 as well. All four show a small signal on the original Corr2Cause pool (AUROC 0.56 to 0.59) that is gone on the debiased pool (0.49 to 0.51). Qwen2.5-3B is the only model above chance on CLadder, barely (accuracy {{behaviour_qwen3b.json|zero_shot/cladder/logit_scored/accuracy|ci}}) (Section 3.1).
2. *First probing result and its retraction.* Linear probes decode the gold label at {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} to {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} on a balanced Corr2Cause pool, and the pre-registered criterion for "represents but doesn't say" passed as coded in some cells. An independent audit showed the criterion's baselines were too weak: the hypothesis string alone predicts the label at {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|est}} on that pool. Against the strongest text-only baseline the probes lose in all six model × benchmark cells. The positive verdict was retracted (Section 3.4).
3. *Text-only baselines on the benchmarks.* On the official Corr2Cause test split a hypothesis-only classifier reaches AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}} and a parsed-feature classifier accuracy {{benchmark_baselines.json|corr2cause_official_splits/test/parsed_features/accuracy|ci}}; on CLadder, quoted numbers plus question polarity reach {{benchmark_baselines.json|cladder_full_v1.5/baselines/numeric_polarity/accuracy|ci}} on held-out stories. The Corr2Cause effect follows from the benchmark's published construction; the CLadder one is largely documented by prior work (Section 3.5 and `report/benchmark_text_baselines.md`).
4. *A fairer test.* On a debiased Corr2Cause pool, in which the hypothesis alone and the premise alone carry no label information in any fold, the gold probe scores {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} for Qwen2.5-0.5B and {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} for Qwen2.5-1.5B. For the 1.5B model this is above every interaction control that binds nothing to the queried variables (strongest: {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|est}}), level with a per-template lookup of the statements naming the queried pair ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|est}}), and below a hand-written parse ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}); it adds about two points to that parse and nothing to a stronger text-only model. For the 0.5B model the final-token probe is not above the no-binding controls. Pooled, both models answer at chance; within "X directly causes Y" hypotheses the 1.5B model's answers partly track validity. The audited criterion for "represents but doesn't say" is not met (Section 3.6; post hoc, audited).
5. *Steering.* At the probed layer and final token, adding the gold probe's weight direction shifts the answer by an amount within the range of probe directions fitted to shuffled labels, and a movement of natural size along it changes nothing measurable. With twenty controls and one site this neither shows nor excludes a causal role. The difference-of-class-means direction moves the answer, about as much as differences of means for shuffled labels (Section 4.1; rewritten after the second audit, wording tightened after the third).
6. *Patching.* A change of one variable in the prompt reaches the answer position at about 0.6 of the network's depth in both Qwen models. This locates the arrival of a prompt difference, not a causal judgement (Section 4.2).
7. *Sparse-autoencoder features (Gemma-2-2B, CLadder).* The public SAEs fit these activations poorly. The features most associated with a causal graph type are not specific to it, and removing them shifts the answer somewhat more than removing matched random features, equally for all graph types (Section 5).
8. *Remote runs.* Gemma-2-2B interventions and everything for Qwen2.5-3B ran on a Kaggle GPU and are included. A second round (Qwen2.5-7B behaviour; matched steering controls for Gemma and Qwen2.5-3B; steering at the end of the hypothesis; label-preserving patching pairs; a causal test of the adjacency information) is reported where its results have arrived and marked as pending where they have not.

**Pre-registration status.** Pre-registered before any probe was run: the wrong-items criterion (D11) with TF-IDF and template-prior baselines, grouped cross-validation, split-half selection, cluster bootstrap. Added after seeing results, each making the claim harder to pass or narrowing it: stronger text baselines (D18), the audited criterion (D19), the debiased pool (D21), the steering metric and matched controls (D24), the two-group reading and its three tests (D25). The pre-registered analysis is reported in full alongside.

## 2. Setup

### 2.1 Benchmarks

| | Corr2Cause (Jin et al., 2023a) | CLadder (Jin et al., 2023b) |
|---|---|---|
| Source | HF `causal-nlp/corr2cause`, default config | HF `causal-nlp/CLadder`, `full_v1.5_default` |
| Task | Premise: all correlations and (conditional) independences among 2–6 variables. Hypothesis: one causal claim. Label 1 iff the claim holds in every graph of the Markov equivalence class | Yes/no question about an association, intervention or counterfactual in a small described causal model |
| Size | train 205,734 / dev 1,076 / test 1,162 | 10,112 |
| Label balance | 18.5% / 12.9% / 15.5% positive (**imbalanced**) | 50.0% yes |
| Structure | 6 hypothesis templates; train has 2,338 distinct premises, test only 21 | rung 1/2/3; 10 query types; 10 graph types; 47 stories |
| Licence | No licence on the HF card; code repository MIT | No licence on the HF card; code repository MIT |

Item sets used (all deterministic given seed 0; `causal_probe.data.get_dataset`):

- **Corr2Cause test** (all 1,162 items): behaviour at the natural label distribution, and an external check of probes.
- **Corr2Cause probe pool** (2,400 items): 300 train premises × (4 valid + 4 invalid hypotheses). Exactly 50% positive, so chance is 50%. Disjoint from test premises (also up to variable relabelling; checked by the audit).
- **CLadder sample** (3,000 items): proportional stratified sample over label × query type (rung is a function of query type).
- **Renamed test** (1,162): the test set with variables A…F renamed Z…U (robustness check; the released perturbation files do not align with the released test set).
- **Few-shot / generation subsets**: 500 evenly spaced items of each of the two evaluation sets.

Imbalance is handled explicitly: macro-F1, AUROC and a majority-class baseline for behaviour; a balanced pool for probing. Items that share a premise (Corr2Cause) or story (CLadder) are never split between probe training and evaluation, and all confidence intervals resample premises/stories (cluster bootstrap, 2,000 resamples).

### 2.2 Models, hardware and the cache

Apple M3, 8 GB unified memory. Models: Qwen2.5-0.5B-Instruct (24 layers, width 896), Qwen2.5-1.5B-Instruct (28, 1536), Gemma-2-2B-it (26, 2304), each with its own chat template (single user turn; Qwen's default system prompt). Qwen2.5-3B does not fit on this machine; it and all Gemma interventions were run on a Kaggle GPU (two T4s, float16, the same code at a pinned commit; `KAGGLE.md`), and the exported activations at the final prompt token were analysed on the laptop with the same scripts. Gemma's answer logit differences agree between the two machines to {{cache_agreement_gemma2b.json|tags/c2c_probe/max_abs_diff|num}} (largest difference over the 2,400 pool items; correlation {{cache_agreement_gemma2b.json|tags/c2c_probe/pearson_r|str}}). Models run in float16 on MPS through plain Hugging Face modules with forward hooks; TransformerLens 4.4's bridge was verified to give the same logits (max difference 7e-5 on five prompts) and not used further. MPS float16 agrees with CPU float32 to 0.02 in the answer logit difference for the 0.5B model; for the two larger models a float32 reference does not fit in memory and the check is against CPU half precision (`results/env_check_*.json`).

Gemma only runs at usable speed if its 256k × 2304 embedding matrix is kept on disk (memory-mapped) and only the answer-token rows of the unembedding are used; its answer logit differences match the full model to 0.012 on the check prompts. Consequences: Gemma's cache has no full-vocabulary top-1 token and its generated-text check is first-token-only on 100 items per set.

**Prompt.** `instruction + premise + hypothesis` (Corr2Cause) or `instruction + context + question` (CLadder), the instruction asking for one word, Yes or No. The model's answer is read from the next-token logits at the final prompt token as logsumexp over {Yes, " Yes", yes, " yes", YES, " YES"} minus the same for No ("answer logit difference", LD).

**Cache.** One forward pass per item; the residual stream at every layer (index 0 = embeddings, index l = output of block l−1) is stored in float16 for three positions: the final prompt token, the last token of the hypothesis/question, and the last token of the premise/context. All Tier 1 analyses run from this cache on CPU. A randomly initialised copy of each model is cached on the probe sets as a control.

### 2.3 Analyses

**Behaviour.** Accuracy, macro-F1, AUROC of LD; majority and random baselines; zero-shot and 4-shot (demonstrations as chat turns); logit scoring vs greedy generated text; variable renaming.

**Logit lens.** Final norm + answer-token unembedding applied to the final-position residual at each layer.

**Probes.** L2 logistic regression on standardised activations, per layer, 5 premise/story-grouped folds × 3 seeds, at C ∈ {0.01, 0.1, 1}. Targets: gold label; the model's own answer (raw sign of LD where it varies; and a median split of LD, "calibrated"). Controls: shuffled training labels; a Hewitt & Liang control task (a fixed random label per hypothesis string / per graph × query type) with selectivity; the random-init model; text-only baselines.

**Text-only baselines** (same folds): TF-IDF (1–2 grams) of the whole prompt, C ∈ {0.1, 1, 10, 100}; template / type prior; and, added post hoc (decisions D18, D19): TF-IDF of the question alone; the label prior of the exact hypothesis string (Corr2Cause); gradient boosting on parsed features (Corr2Cause: premise statements bound to the hypothesis variables, e.g. skeleton adjacency, plus the hypothesis-string prior; CLadder: the quoted probabilities, their differences, polarity words, query and graph type). None of them applies causal inference rules.

**Selection and reporting.** Premises/stories are split in half once. The probe's (layer, C), the TF-IDF C and the choice of strongest text baseline are made on the selection half; all reported probe-vs-baseline numbers are on the report half.

**Criteria for "represents but doesn't say".** *Pre-registered (D11):* on report-half items the model answers wrongly, gold-probe accuracy > 0.5 and > TF-IDF and > template prior (95% CI lower bounds). *Audited (D19):* the gold probe beats the strongest text-only baseline on all report-half items (paired accuracy) and inside both calibrated model-answer strata (paired AUROC). The second was adopted after an independent audit showed the first can be passed by shortcut (Section 3.4).

**Steering (Tier 2).** At the selected layer, final token: add ±α·s·d for unit directions d ∈ {gold-probe weights, gold difference of class means, model-output probe weights, 3 random}, s = RMS distance of activations from their mean, α ∈ {¼, ½, 1, 2, 4, 8}; and directional ablation (set the component along d to its mean). Directions are fitted on the selection half and applied to 300 report-half items per dataset.

**Patching (Tier 2).** Token-aligned Corr2Cause minimal pairs with opposite gold labels: *hyp* (same premise, hypothesis variable changed) and *prem* (same hypothesis, two variable letters swapped throughout the premise). Residual-stream patching clean→corrupt over layer × token region, with a random-donor control; attribution patching over all heads and MLPs, validated by real patching of the top 10 and 10 random components. Effect size: pooled recovery Σ sign(D)·(LD_patched − LD_corrupt) / Σ|D| with D = LD_clean − LD_corrupt.

## 3. Tier 1 results

### 3.1 Behaviour

<!-- TABLE:behaviour_zero_shot -->

![behaviour](../figures/fig1_behaviour.png)

**No model answers above chance, and for different reasons.** On the Corr2Cause test set (15.5% valid) Qwen2.5-0.5B answers "No" to essentially everything (Yes-rate {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|num2}}), so its accuracy {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/accuracy|est}} is the majority rate and its macro-F1 {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/macro_f1|est}} is the majority baseline's. Qwen2.5-1.5B says Yes {{behaviour_qwen1.5b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|pct}} of the time and Gemma-2-2B {{behaviour_gemma2b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|pct}}; their accuracies ({{behaviour_qwen1.5b.json|zero_shot/c2c_test/logit_scored/accuracy|ci}}, {{behaviour_gemma2b.json|zero_shot/c2c_test/logit_scored/accuracy|ci}}) are *below* the majority rate because the extra Yes answers are mostly wrong. Qwen2.5-3B, run remotely, behaves like Qwen2.5-0.5B on Corr2Cause (Yes-rate {{behaviour_qwen3b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|num2}}, accuracy {{behaviour_qwen3b.json|zero_shot/c2c_test/logit_scored/accuracy|ci}}, below the majority rate). On CLadder (50% yes) the three local models are at 0.50 and Qwen2.5-3B at {{behaviour_qwen3b.json|zero_shot/cladder/logit_scored/accuracy|ci}}.

Accuracy alone cannot tell "no information" from "information hidden by a bias towards one answer". The next table separates them: the Yes-rate, the AUROC of the Yes−No logit difference (which ignores any constant bias), and the accuracy when the decision threshold is moved to its best position (chosen on one half of the premises or stories, scored on the other).

<!-- TABLE:answer_bias -->

Reading. On the original Corr2Cause sets the logit difference is weakly informative: AUROC {{answer_bias.json|qwen0.5b/c2c_test/auroc|ci}} for Qwen2.5-0.5B, {{answer_bias.json|qwen1.5b/c2c_test/auroc|ci}} for Qwen2.5-1.5B on test, and 0.56 to 0.59 on the balanced pool for all three, where a re-thresholded answer reaches about 0.57 accuracy. So the smallest model is biased rather than empty there. This weak signal does not survive on the debiased pool of Section 3.6 for any model ({{answer_bias.json|qwen0.5b/c2c_debiased/auroc|ci}}, {{answer_bias.json|qwen1.5b/c2c_debiased/auroc|ci}}, {{answer_bias.json|gemma2b/c2c_debiased/auroc|ci}} and {{answer_bias.json|qwen3b/c2c_debiased/auroc|ci}} for Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-2-2B and Qwen2.5-3B). Renaming the variables removes it for three of the four models (Qwen2.5-0.5B {{answer_bias.json|qwen0.5b/c2c_test_renamed/auroc|ci}}, Gemma-2-2B {{answer_bias.json|gemma2b/c2c_test_renamed/auroc|ci}}, Qwen2.5-3B {{answer_bias.json|qwen3b/c2c_test_renamed/auroc|ci}}) but not for Qwen2.5-1.5B ({{answer_bias.json|qwen1.5b/c2c_test_renamed/auroc|ci}}); an earlier version of this paragraph quoted the 0.5B model only. On CLadder the three local models are at chance by AUROC as well (0.50 to 0.52); Qwen2.5-3B is slightly above ({{answer_bias.json|qwen3b/cladder/auroc|ci}}). The behavioural picture is therefore: at chance on the task, with a small shortcut-driven signal on the original Corr2Cause items that is masked by answer bias.

Other checks. Four demonstrations change the bias, not the skill: Qwen2.5-0.5B flips to mostly Yes (accuracy change {{behaviour_qwen0.5b.json|few_shot/c2c_test_sub_fewshot/delta_accuracy_fewshot_minus_zeroshot|signedci}}), Gemma loses {{behaviour_gemma2b.json|few_shot/c2c_test_sub_fewshot/delta_accuracy_fewshot_minus_zeroshot|signedci}}. Scoring by answer-token logits and by generated text disagree on under 1% of items for the Qwen models; for Gemma on CLadder 13% of first tokens are neither Yes nor No (first-token-only check, 100 items). Renaming variables flips {{behaviour_qwen1.5b.json|robustness_variable_renaming/prediction_flip_frac|ci}} of Qwen2.5-1.5B's answers and {{behaviour_gemma2b.json|robustness_variable_renaming/prediction_flip_frac|ci}} of Gemma's.

Few-shot (500-item subsets, paired with zero-shot on the same items):

<!-- TABLE:behaviour_few_shot -->

Logit scoring vs generated text:

<!-- TABLE:behaviour_generation -->

Robustness to variable renaming (paired on the test set):

<!-- TABLE:robustness_renaming -->

By rung and query type (CLadder) and by template (Corr2Cause test):

<!-- TABLE:behaviour_by_cladder_rung -->

<!-- TABLE:behaviour_by_cladder_query_type -->

<!-- TABLE:behaviour_by_c2c_test_template -->

### 3.2 Logit lens

![lens](../figures/fig2_logit_lens.png)
![lens auroc](../figures/fig2b_lens_auroc.png)

The first figure shows the Yes−No logit difference the model would produce if it stopped at each layer, for gold-Yes and gold-No items separately. In every panel the curves move together: the lens shows a shared drift towards one answer, not a separation by the correct answer. The second figure removes the drift by asking only whether the lens value ranks gold-Yes above gold-No items (AUROC). On the Corr2Cause pool it never exceeds 0.60 at any layer for the three local models (the maxima are 0.59 to 0.60, reached in the second half of the network) and reaches 0.63 for Qwen2.5-3B; on CLadder it stays between 0.47 and 0.53 at every layer for the local models and reaches 0.54 for Qwen2.5-3B (`results/lens_*.json`, key `auroc_gold_by_layer`). The lens at the last layer reproduces the model's actual answer logits (maximum error and sign agreement are in `results/lens_*.json`). So the lens gives no sign of a correct answer that forms in the middle of the network and is then overwritten.

### 3.3 Probes and controls

<!-- TABLE:probes_summary -->

![probes c2c](../figures/fig3_probes_c2c_probe.png)
![probes cladder](../figures/fig3_probes_cladder.png)
![selectivity](../figures/fig4_selectivity_positions_c2c_probe.png)

Probe accuracy by token position:

<!-- TABLE:probe_positions -->

Probes decode the gold label from held-out premises at {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|ci}}, {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|ci}} and {{confound_gemma2b_c2c_probe.json|all_items/gold_probe/accuracy|ci}} on the Corr2Cause pool, and at {{confound_qwen0.5b_cladder.json|all_items/gold_probe/accuracy|ci}}, {{confound_qwen1.5b_cladder.json|all_items/gold_probe/accuracy|ci}} and {{confound_gemma2b_cladder.json|all_items/gold_probe/accuracy|ci}} on CLadder (report half; layer and regularisation chosen on the other half). The controls behave as controls should: shuffled labels give 0.50, and a randomly initialised model gives 0.50 to 0.62. Taken alone, this is the pattern that is usually read as "the model represents the answer".

Three observations already argue against that reading before any text baseline is considered. (1) Selectivity is small on Corr2Cause (+0.07 to +0.12) and negative on CLadder (about −0.2): the same probes memorise an arbitrary label per hypothesis string, or per graph and query type, almost as well as or better than they decode the gold label. The activations expose *which question is being asked* very well, and on Corr2Cause that alone predicts validity (Section 3.5). (2) On Corr2Cause the probe is better at the last token of the hypothesis than at the final prompt token where the answer is produced (final minus end-of-hypothesis is negative for all three models), and at the end of the premise, before the hypothesis has been read, it is at chance as it must be in a pool balanced within premise. (3) On CLadder most of the probe's small signal is present at the end of the context, before the question has been read: the gain from reading the question is {{confound_qwen0.5b_cladder.json|positions/premise_end/final_minus_this_accuracy|signedci}} for Qwen2.5-0.5B, {{confound_qwen1.5b_cladder.json|positions/premise_end/final_minus_this_accuracy|signedci}} for Qwen2.5-1.5B and {{confound_gemma2b_cladder.json|positions/premise_end/final_minus_this_accuracy|signedci}} for Gemma-2-2B.

### 3.4 Gold label vs model output: is anything represented beyond the text?

All scorers on the report half:

<!-- TABLE:text_baselines -->

Audited criterion (and, for comparison, the pre-registered one as coded):

<!-- TABLE:audited_verdict -->

![confound c2c](../figures/fig5_confound_c2c_probe.png)
![confound cladder](../figures/fig5_confound_cladder.png)

Pre-registered analysis in full (items the model gets wrong):

<!-- TABLE:confound_wrong_items -->

Within model-answer strata:

<!-- TABLE:confound_within_strata -->

The same under the original fixed regularisation (probe C = 0.01, TF-IDF C = 1):

<!-- TABLE:confound_fixedC_wrong_items -->

External check: probe trained on the pool, evaluated on the Corr2Cause test set (15.5% positive; AUROC):

<!-- TABLE:external_c2c_test -->

**Result: the audited criterion is not met in any of the six model × dataset cells.** The strongest text-only baseline is the parsed-feature classifier everywhere. On all report-half items the probe trails it by {{confound_qwen0.5b_c2c_probe.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}}, {{confound_qwen1.5b_c2c_probe.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} and {{confound_gemma2b_c2c_probe.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} on Corr2Cause, and by {{confound_qwen0.5b_cladder.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}}, {{confound_qwen1.5b_cladder.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} and {{confound_gemma2b_cladder.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} on CLadder. On Corr2Cause the hypothesis string alone scores {{confound_qwen0.5b_c2c_probe.json|all_items/hyp_prior/accuracy|est}}; the probe's paired difference to it is {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe_minus_hyp_prior/accuracy|signedci}} for Qwen2.5-0.5B, {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe_minus_hyp_prior/accuracy|signedci}} for Qwen2.5-1.5B and {{confound_gemma2b_c2c_probe.json|all_items/gold_probe_minus_hyp_prior/accuracy|signedci}} for Gemma-2-2B. Nothing in these data requires the models to represent more about the answer than a shallow reader of the prompt could compute.

**What was pre-registered, what passed, and what was retracted.** Before any probe was run, the decision log fixed this criterion (D11): on report-half items the model answers wrongly, gold-probe accuracy must exceed 0.5, TF-IDF and the template prior, by 95% CI lower bounds. As coded, that criterion is met for Qwen2.5-1.5B on CLadder and for Gemma-2-2B on CLadder, and in variants (median-split model answer, or the original fixed regularisation) also for Qwen2.5-1.5B and Gemma-2-2B on Corr2Cause; the full tables are above. When the first of these appeared (Qwen2.5-1.5B, Thursday night) I logged it as a positive result. The independent audit then showed why it does not support the claim, and I retracted it:

- the pre-registered baselines were too weak. TF-IDF of the whole prompt dilutes the hypothesis, and a six-template prior ignores which variables are named. The hypothesis string alone, and parsed features, are far stronger (table above);
- "accuracy above 0.5 on the items the model gets wrong" is not a test against chance when those items are mostly one class (for Qwen2.5-1.5B on Corr2Cause, 90% of wrong items are gold-Yes);
- with model answers almost independent of the gold label, "items the model gets wrong" is close to a random half of the data, so the design reduces to probe versus text baseline.

The stronger baselines (D18), the audited criterion (D19) and the debiased pool of Section 3.6 (D21) were all added **after** seeing results. They can only make the claim harder to pass, and they are reported next to the pre-registered analysis, not instead of it.

The external check agrees: a probe trained on the pool and applied to the natural-distribution test set ranks valid above invalid items well (AUROC in the table), but so does the hypothesis-string prior on the same items: the table's value is for a prior fitted on the 300-premise pool, like the probe; fitted on the whole train split it reaches {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}.

### 3.5 What the text alone gives away

The full analysis, with data versions, definitions and what is and is not already known, is in `report/benchmark_text_baselines.md` and is reproducible from one script (`scripts/14_benchmark_baselines.py`). The two tables that matter for this study:

<!-- TABLE:benchmark_c2c_pools -->

<!-- TABLE:benchmark_cladder -->

On the balanced Corr2Cause pool the hypothesis string alone predicts the label with {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|ci}} accuracy; this follows from how the benchmark names variables (causes get earlier letters), which makes 72 of the 120 hypothesis strings never valid. On CLadder, the quoted numbers and the polarity of the question predict {{benchmark_baselines.json|cladder_full_v1.5/baselines/numeric_polarity/accuracy|ci}} on held-out stories, a result largely anticipated by Clarke et al. (2026). These are properties of how text-only classifiers perform on the released data; for a probing study they set the bar a probe has to clear.

### 3.6 A fairer test: the debiased Corr2Cause pool

The original pool cannot separate "represents the answer" from "represents the hypothesis". The debiased pool (D21) removes every single-input route: 2,400 items in which each premise contributes one valid and one invalid hypothesis and each of the 48 usable hypothesis strings is half valid in every cross-validation fold. A classifier that sees only the hypothesis or only the premise scores exactly 0.50 by construction, and whole-prompt TF-IDF scores {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/full_text_tfidf/accuracy|ci}}. The parsed-feature classifier still reaches {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/parsed_features/accuracy|ci}}: it binds the premise's statements to the two variables named in the hypothesis, which is the first step of the task itself.

Gold-label probes on the debiased pool (report half; layer and regularisation chosen on the other half; hypothesis-only and premise-only classifiers are at 0.500 by construction):

| Model | Gold probe, final token | Random-init model's probe, final token | Whole-prompt TF-IDF | Parsed features | Probe − random-init | Probe − parsed features | Gold probe, end of hypothesis | Random-init, end of hypothesis |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_qwen0.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | {{confound_qwen0.5b_c2c_debiased.json|positions/content_end/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|positions/content_end/random_init_accuracy|ci}} |
| Qwen2.5-1.5B | {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_qwen1.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | {{confound_qwen1.5b_c2c_debiased.json|positions/content_end/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|positions/content_end/random_init_accuracy|ci}} |
| Gemma-2-2B (activations from Kaggle) | {{confound_gemma2b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_gemma2b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | not exported | not exported |
| Qwen2.5-3B (activations from Kaggle) | {{confound_qwen3b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_qwen3b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | not exported | not exported |

<sub>Source: `results/confound_<model>_c2c_debiased.json`. For the two models cached on Kaggle only the final-token activations were exported in the first run.</sub>

**Verdicts under the two criteria, unchanged.** The audited criterion (probe beats the strongest text-only baseline; D19) is **not met** for any of the four models: the probe trails the parsed-feature classifier by {{confound_qwen0.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} (Qwen2.5-0.5B), {{confound_qwen1.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} (Qwen2.5-1.5B), {{confound_gemma2b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} (Gemma-2-2B) and {{confound_qwen3b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} (Qwen2.5-3B). The three larger models land within a point and a half of one another (0.75 to 0.77): going from 1.5 to 3 billion parameters makes no visible difference. The original pre-registered criterion as coded (D11: against TF-IDF and the template prior, on wrongly answered items) is met for all four models on this pool; as explained in Section 3.4 it does not test the claim, and it is reported, not relied on.

#### 3.6.1 Post-hoc reading: a ladder of baselines in two groups

*Everything in 3.6.1 to 3.6.6 was designed after the debiased-pool probe results were seen (13 November; `scripts/05b_probe_content.py`) and revised after the third independent audit (`notes/audit_debiased_raw.md`). It is a labelled post-hoc analysis and changes neither criterion above.*

In this pool a classifier that sees only the hypothesis, or only the premise, is at 0.50 by construction, and so is any model that is additive in premise and hypothesis features, such as a linear bag-of-words of the whole prompt. Label information can only come from an *interaction* between hypothesis and premise. Listing the single-input controls as things the probe "beats" would therefore say nothing; they are shown as group (a0) to confirm the construction. The informative baselines form a ladder, ordered by how much they bind the premise to the hypothesis:

- **(a) Shallow confound controls**: interaction models that bind nothing to the queried variables. One model per hypothesis template on premise-level counts or on a bag-of-words of the premise; and a probe on a randomly initialised copy of the network.
- **(b) Structure-aware text references**: models that bind premise statements to the queried variables. Per template, counts of statements mentioning each queried variable separately; per template, a lookup of the statements naming the queried pair; per hypothesis string, a bag of premise n-grams (no parser; a separate model for each of the 48 strings can learn which statements matter for that string); the hand-written parse with all features; and the hand parse stacked with the per-string n-gram model, the strongest text-only model used here.

<!-- TABLE:debiased_ladder -->

For Qwen2.5-1.5B the final-token probe ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}} accuracy, AUROC {{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/auroc|ci}}) is above every control in group (a): the strongest is the per-template bag-of-words of the premise ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|ci}}; probe minus control {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/probe_minus_this_accuracy|signedci}}). Within group (b) it is indistinguishable from the per-template lookup of statements naming the pair ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|ci}}; difference {{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/probe_minus_this_accuracy|signedci}}) and from per-variable statement counts ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_per_variable_statement_counts/accuracy|ci}}), and below the per-string 4-gram model ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/per_string_premise_4grams/accuracy|ci}}), the hand parse ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|ci}}) and their stack ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/strongest_text_only_parse_plus_4grams/accuracy|ci}}). "Shallow" and "structure-aware" are ends of a continuum, and the probe sits where the binding starts: at the level of looking up what the premise says about the two queried variables.

Gemma-2-2B ({{probe_content_gemma2b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}) and Qwen2.5-3B ({{probe_content_qwen3b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}) sit in the same place on the ladder as Qwen2.5-1.5B: above the per-template bag-of-words of the premise (by {{probe_content_gemma2b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/probe_minus_this_accuracy|signedci}} and {{probe_content_qwen3b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/probe_minus_this_accuracy|signedci}}), indistinguishable from the per-template lookup of statements naming the pair ({{probe_content_gemma2b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/probe_minus_this_accuracy|signedci}} and {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/probe_minus_this_accuracy|signedci}}), and below the hand parse ({{probe_content_gemma2b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/probe_minus_this_accuracy|signedci}} and {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/probe_minus_this_accuracy|signedci}}).

For Qwen2.5-0.5B the final-token probe ({{probe_content_qwen0.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}) is **not** above group (a): it is level with the random-init probe ({{probe_content_qwen0.5b_c2c_debiased.json|baselines/A_no_binding_interaction/probe_on_random_init_model/accuracy|ci}}; difference {{probe_content_qwen0.5b_c2c_debiased.json|baselines/A_no_binding_interaction/probe_on_random_init_model/probe_minus_this_accuracy|signedci}}, one random initialisation) and no better than the per-template bag-of-words ({{probe_content_qwen0.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|ci}}; difference {{probe_content_qwen0.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/probe_minus_this_accuracy|signedci}}). At the final token there is no evidence of binding for this model. Evidence for it exists elsewhere: at the end of the hypothesis its probe scores {{confound_qwen0.5b_c2c_debiased.json|positions/content_end/accuracy|ci}}, and the adjacency probe below is well above its controls.

Robustness of the headline number:

<!-- TABLE:debiased_robustness -->

The reported accuracy is an ensemble over three fold assignments; single assignments are one to two points lower for Qwen2.5-1.5B. A probe trained on the selection half only (half the data) scores lower still. The shuffled-label null is centred on 0.50 with a standard deviation of about 0.013.

#### 3.6.2 Does the probe add predictive power beyond the text models?

Nested logistic stacking: for every outer fold, the text model's score and the probe's score for the training rows come from an inner cross-validation, a two- or three-parameter logistic regression is fitted on them, and it is applied to the test rows.

<!-- TABLE:debiased_incremental -->

For Qwen2.5-1.5B the probe adds {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_hand_parse/gain_accuracy|signedci}} accuracy (AUROC {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_hand_parse/gain_auroc|signedci}}) to the hand-written parse, and {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}} to the strongest text-only model. For Qwen2.5-0.5B: {{probe_content_qwen0.5b_c2c_debiased.json|incremental_value/over_hand_parse/gain_accuracy|signedci}} and {{probe_content_qwen0.5b_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}}. For Gemma-2-2B: {{probe_content_gemma2b_c2c_debiased.json|incremental_value/over_hand_parse/gain_accuracy|signedci}} over the hand parse and {{probe_content_gemma2b_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}} over the strongest text-only model; for Qwen2.5-3B: {{probe_content_qwen3b_c2c_debiased.json|incremental_value/over_hand_parse/gain_accuracy|signedci}} and {{probe_content_qwen3b_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}}. So the activations of the three larger models hold two to three points' worth of validity information that this particular hand parse lacks. Over the strongest text-only model the gain is zero for Qwen2.5-1.5B and about one point, with an interval that reaches zero, for Gemma-2-2B and Qwen2.5-3B. There is a mild trend with model size and no clear excess over text in any model.

*History of this number, for the record.* A first, non-nested stacking gave +0.023; I withdrew it for not being nested and replaced it with "add the probe score as one column to gradient boosting", which gave +0.006 [−0.015, +0.026] and which I reported as "no measurable gain". The third audit showed with a synthetic column of known value that this second test cannot detect a two-point gain, and that proper nesting reproduces the first number. The insensitive test's result is kept in the result file (`gbm_plus_one_column_insensitive`).

#### 3.6.3 Which parsed properties can be read from the activations?

A probe for each binary property a parser reads off the item, at the layer the gold probe uses, with a shuffled-label null (10 shuffles), the same probe on the random-init model, two text baselines for the same property, and the same probe at the end of the premise, before the hypothesis has been read.

<!-- TABLE:debiased_properties -->

"The queried pair is adjacent in the skeleton" is decodable from Qwen2.5-1.5B at AUROC {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained/auroc|ci}}, against {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init_same_layer_auroc|ci}} from the random-init copy at the same layer ({{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init_best_layer_auroc|num}} at its best layer), {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/text_no_binding_auroc|ci}} from text without binding, and {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained_at_end_of_premise_auroc|ci}} at the end of the premise, where the model cannot yet know which pair is asked about. A text model with per-variable statement counts and no pair lookup already reaches {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/text_per_variable_counts_auroc|ci}} for this property, so most of what the probe reads requires binding statements to each queried variable; the excess over that ({{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained_minus_text_per_variable_auroc|signedci}}) is what points to the pair being bound. For Qwen2.5-0.5B the same property is decodable at {{probe_content_qwen0.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained/auroc|ci}} (random-init {{probe_content_qwen0.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init_same_layer_auroc|ci}}; per-variable text {{probe_content_qwen0.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/text_per_variable_counts_auroc|ci}}). Premise-level counts are perfectly decodable from every source, including the random-init model and the end of the premise, and say nothing about computation. "The pair correlates" is nearly as decodable without any binding (it is true for 92% of items). Shuffled-label nulls for rare properties sit a little below 0.5; that is a property of shuffling an imbalanced label, confirmed over repeated shuffles.

For Gemma-2-2B and Qwen2.5-3B adjacency is decodable at AUROC {{probe_content_gemma2b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained/auroc|ci}} and {{probe_content_qwen3b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained/auroc|ci}}, against {{probe_content_gemma2b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init_same_layer_auroc|ci}} and {{probe_content_qwen3b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init_same_layer_auroc|ci}} from their random-init copies. The end-of-premise check is not available for these two models (final-token activations only).

#### 3.6.4 Is the probe's signal carried by one property, and how much is shared with text?

The gold probe's AUROC computed only over valid/invalid pairs that fall in the same stratum.

<!-- TABLE:debiased_strata -->

It does not fall inside strata of any single parsed property, which is expected because no single property predicts the label in this pool. It does fall inside strata of text-derived *scores*: for Qwen2.5-1.5B from {{probe_content_qwen1.5b_c2c_debiased.json|stratified/overall_probe_auroc|ci}} overall to {{probe_content_qwen1.5b_c2c_debiased.json|stratified/by_stratifier/hand_parse_score_quintiles/probe_auroc_within_strata|ci}} within quintiles of the hand-parse score and to {{probe_content_qwen1.5b_c2c_debiased.json|stratified/by_stratifier/parse_quintile_x_4gram_quintile/probe_auroc_within_strata|ci}} within 25 cells of two text scores; for Qwen2.5-0.5B to {{probe_content_qwen0.5b_c2c_debiased.json|stratified/by_stratifier/parse_quintile_x_4gram_quintile/probe_auroc_within_strata|ci}}; for Gemma-2-2B and Qwen2.5-3B to {{probe_content_gemma2b_c2c_debiased.json|stratified/by_stratifier/parse_quintile_x_4gram_quintile/probe_auroc_within_strata|ci}} and {{probe_content_qwen3b_c2c_debiased.json|stratified/by_stratifier/parse_quintile_x_4gram_quintile/probe_auroc_within_strata|ci}}. Most of the probe's signal is shared with what text models extract; a little more is left over in the two largest models than in Qwen2.5-1.5B.

#### 3.6.5 Do the model's answers express it?

Pooled over all items the answers are at chance (AUROC {{probe_content_qwen1.5b_c2c_debiased.json|answers/pooled_auroc_for_gold|ci}} for Qwen2.5-1.5B). The pooling hides structure: the hypothesis string alone explains {{probe_content_qwen1.5b_c2c_debiased.json|answers/frac_variance_of_answer_logit_explained_by_hypothesis_string|pct}} of the variance of that model's answer logit, so the fair comparison is between a valid and an invalid item with the *same* hypothesis string.

<!-- TABLE:debiased_answers -->

For "X directly causes Y" hypotheses, Qwen2.5-1.5B's answer logit does track validity ({{probe_content_qwen1.5b_c2c_debiased.json|answers/by_template/parent/within_string_auroc_for_gold|ci}}; similar in both halves of the data) and correlates with the probe's score ({{probe_content_qwen1.5b_c2c_debiased.json|answers/by_template/parent/corr_answer_logit_with_probe_score|signedci}}). For the two indirect-causation templates and for confounders it is weakly *anti*-correlated with validity, and for colliders flat. Summed over templates these cancel. So it is not accurate to say the model's answers ignore what the probe reads: for the simplest relation they use it in the right direction, and for others in the wrong one. This breakdown is post hoc over five templates; the direct-causation effect for Qwen2.5-1.5B is about five standard errors.

The other two models do not repeat that pattern, which is a reason not to read much into it. Qwen2.5-3B tracks validity for direct-causation hypotheses more weakly ({{probe_content_qwen3b_c2c_debiased.json|answers/by_template/parent/within_string_auroc_for_gold|ci}}) and runs against it for confounder hypotheses ({{probe_content_qwen3b_c2c_debiased.json|answers/by_template/has_confounder/within_string_auroc_for_gold|ci}}). Gemma-2-2B runs *against* validity for direct-causation hypotheses ({{probe_content_gemma2b_c2c_debiased.json|answers/by_template/parent/within_string_auroc_for_gold|ci}}; correlation of its answer logit with the probe's score {{probe_content_gemma2b_c2c_debiased.json|answers/by_template/parent/corr_answer_logit_with_probe_score|signedci}}). What all four models share is that the answer depends on whether the queried pair is adjacent, with a sign that differs by hypothesis type and by model; Section 4.4 tests whether that dependence runs through a linear encoding of adjacency.

#### 3.6.6 What this supports

On a Corr2Cause pool built so that the hypothesis alone and the premise alone are exactly uninformative:

- **Qwen2.5-1.5B.** A linear probe on the final-token activations predicts validity above every interaction control that binds nothing to the queried variables, at the level of a per-template lookup of the statements naming the queried pair, and below a parser-free per-hypothesis n-gram model and a hand-written parse. It adds about two points to the hand parse and nothing measurable to a stronger text-only model. Whether the queried pair is adjacent in the premise's skeleton is linearly readable, far better than from a random-init copy or from text without binding, and not before the hypothesis is read. The answers are at chance when pooled but track validity for direct-causation hypotheses and run against it for others.
- **Gemma-2-2B and Qwen2.5-3B.** The same picture as Qwen2.5-1.5B at the final token: probe accuracy 0.76 to 0.77, equal to the per-template pair lookup, five points below the hand parse, about three points of incremental value over the hand parse and at most one over the strongest text-only model. Their answers are at chance when pooled; by hypothesis type they differ from Qwen2.5-1.5B and from each other.
- **Qwen2.5-0.5B.** At the final token nothing beyond no-binding controls; some binding is visible at the end of the hypothesis and in the adjacency probe.
- **Not supported:** that any of the four models holds a "correct answer" it fails to give; that the activations contain validity information unavailable to a text-only classifier; that the information is unused.

The steering evidence is in Section 4.1 and the test of the adjacency encoding in Section 4.4.

## 4. Tier 2 results

This section was rewritten after the second independent audit (Section 6), which showed that my first summary of the steering results ("the gold-probe direction is inert, the model-output direction flips answers") did not survive scrutiny as stated. The analysis below uses the audit's metrics and adds the controls it asked for. Results for Gemma-2-2B and Qwen2.5-3B come from the Kaggle run and are added when it returns; that run used three random directions only, so the matched controls exist for the two local Qwen models.

### 4.1 Steering: what does adding a decodable direction do to the answer?

**Design.** At one layer (the gold probe's selected layer) and the final prompt token, add ±α·s·d for a unit direction d and read the Yes−No answer logit difference (LD). s is the RMS distance of that layer's activations from their mean, a norm over all dimensions. Directions are fitted on the selection half and applied to 300 report-half items (150 valid, 150 invalid). Three real directions: the gold-label probe's weight vector; the difference between the mean activation of valid and invalid items; and, as a positive control, the weight vector of a probe for the model's own (median-split) answer, which is fitted to the very quantity being measured. The headline number is the **slope**, (LD(+α) − LD(−α)) / 2α at α = 1, in logits per unit of s.

<!-- TABLE:steering -->

![steering c2c](../figures/fig6_steering_c2c_probe.png)
![steering debiased](../figures/fig6_steering_c2c_debiased.png)
![steering cladder](../figures/fig6_steering_cladder.png)

**Matched controls** (150 report-half items for Qwen2.5-0.5B, 100 for Qwen2.5-1.5B): 100 isotropic random directions; 20 probe weight vectors fitted to shuffled labels, the control of the same kind as a probe direction; 20 differences of class means for shuffled labels, the control of the same kind as a difference of means.

<!-- TABLE:steering_controls -->

**What the data show.**

1. *The gold-probe weight direction shifts the answer by an amount within the range of label-free directions of the same kind.* Slope, with its place among 20 probe directions fitted to shuffled labels: original pool {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/slope_per_s|signedci}} ({{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_max_abs_slope_per_s|num2}} of their maximum) for Qwen2.5-0.5B and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/slope_per_s|signedci}} ({{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_max_abs_slope_per_s|num2}} of their maximum) for Qwen2.5-1.5B; debiased pool {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/slope_per_s|signedci}} ({{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/ratio_to_max_abs_slope_per_s|num2}} of their maximum) and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/slope_per_s|signedci}} ({{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/ratio_to_max_abs_slope_per_s|num2}} of their maximum). The slope is small but not zero: it is negative in all four cases and has the same sign for nearly every item (pushing "towards valid" lowers the Yes logit; my first table showed one sign only and hid this). Twenty controls cannot exclude an effect two or three times the control median, so the statement is "within the range of", not "no more than".
2. *The comparison depends on the unit.* The shuffled-label directions have the same natural spread as the gold direction on the debiased pool, but on the original pool the gold direction is about three times wider. Per natural standard deviation, the gold direction for Qwen2.5-1.5B on the original pool exceeds all 20 shuffled controls ({{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/ratio_to_median_abs_slope_per_own_sd|num2}} times their median); in the other three cases it is at the {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_own_sd|pct}}, {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_own_sd|pct}} and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_own_sd|pct}} percentile. The original pool is the one in which the gold label is confounded with the hypothesis string.
3. *At natural scale the effect is negligible either way.* One unit of s is {{steering_analysis_qwen0.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|num2}} (Qwen2.5-0.5B) and {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|num2}} (Qwen2.5-1.5B) natural standard deviations along the gold-probe direction on the original pool, and {{steering_analysis_qwen0.5b_c2c_debiased.json|directions/gold_probe/alpha_1_in_own_sd|num2}} and {{steering_analysis_qwen1.5b_c2c_debiased.json|directions/gold_probe/alpha_1_in_own_sd|num2}} on the debiased pool. Moving an item across the whole natural gap between valid and invalid items along that direction would change LD by {{steering_analysis_qwen0.5b_c2c_probe.json|directions/gold_probe/effect_of_natural_class_gap_on_logit_diff|signed}} and {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/effect_of_natural_class_gap_on_logit_diff|signed}} logits (original pool), against a standard deviation of LD across items of {{steering_analysis_qwen0.5b_c2c_probe.json|baseline/logit_diff_sd|num2}} and {{steering_analysis_qwen1.5b_c2c_probe.json|baseline/logit_diff_sd|num2}}.
4. *The difference-of-means direction moves the answer, mostly not because it separates valid from invalid items.* Original pool: {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/slope_per_s|signedci}} ({{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_max_abs_slope_per_s|num2}} of their maximum) for Qwen2.5-0.5B and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_diffmeans/slope_per_s|signedci}} ({{steering_controls_qwen1.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_max_abs_slope_per_s|num2}} of their maximum) for Qwen2.5-1.5B; debiased pool: {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_diffmeans/slope_per_s|signedci}} ({{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_max_abs_slope_per_s|num2}} of their maximum) and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/slope_per_s|signedci}} ({{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} percentile of 20; {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_median_abs_slope_per_s|num2}} times their median, {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/ratio_to_max_abs_slope_per_s|num2}} of their maximum). A difference of class means lies in the high-variance part of activation space, and so does a difference of means for random labels. On the original pool it separates the model's own answer at least as well as the gold label (AUROC {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_diffmeans/auroc_model_answer_report_half|num}} against {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_diffmeans/auroc_gold_report_half|num}} for Qwen2.5-1.5B).
5. *The positive control works, with one qualification.* The model-answer probe direction exceeds all 100 isotropic and all 20 shuffled-label probe directions in every case. It does not have the largest slope of anything steered: for Qwen2.5-1.5B on the debiased pool the gold difference of means is larger ({{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/slope_per_s|signed}} against {{steering_controls_qwen1.5b_c2c_debiased.json|real/model_out/slope_per_s|signed}}), and the largest shuffled-label difference of means there has |slope| {{steering_controls_qwen1.5b_c2c_debiased.json|families/shuffled_diffmeans/abs_slope_per_s/max|num}}. Differences of means, for real or shuffled labels, point into the high-variance directions that the answer is most sensitive to; a probe direction is the fairer comparison for a probe direction. (An earlier version of this sentence said "largest slope everywhere"; that was wrong.)

**What may and may not be concluded.**

- May: adding the gold-probe direction at the probe's layer and the final token, in steps of one RMS activation radius ({{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|int}} to {{steering_analysis_qwen0.5b_c2c_debiased.json|directions/gold_probe/alpha_1_in_own_sd|int}} natural standard deviations along that direction, depending on model and pool), shifts the Yes−No logit by an amount inside the range of 20 shuffled-label probe directions. A movement of natural size changes the answer by less than 0.01 logits.
- May not: that the direction "is not read", or that the decoded information "is unused". With twenty controls and one site the experiment neither shows nor excludes a causal role. The probe's weights and the difference of class means are two views of one discriminant (the covariance times the probe weights has cosine {{steering_analysis_qwen1.5b_c2c_probe.json|geometry/cos_covariance_times_probe_vs_diffmeans|num2}} with the difference of means for Qwen2.5-1.5B). On the debiased pool the fitted direction is itself noisy (cosine {{steering_analysis_qwen1.5b_c2c_debiased.json|directions/gold_probe/split_half_cosine|num2}} between the two halves of the data). And the probe is stronger at the end of the hypothesis, which was not steered.
- What the behavioural data add (Section 3.6.5): the answers are at chance when pooled, but for direct-causation hypotheses they track validity and the probe's score. "Not expressed" holds only on average.
- CLadder: the steered gold directions barely decode the gold label there (AUROC {{steering_analysis_qwen0.5b_cladder.json|directions/gold_probe/auroc_gold_report_half|num}} and {{steering_analysis_qwen1.5b_cladder.json|directions/gold_probe/auroc_gold_report_half|num}}), so those rows are shown for completeness and no conclusion is drawn from them.

**Gemma-2-2B and Qwen2.5-3B (first Kaggle run).** Their rows in the first table come from the run that was designed before the second audit: three isotropic random directions and no matched controls. They can therefore not be placed "inside or outside the range of label-free directions", and I draw no conclusion from them yet. For the record: the gold-probe direction's slope is {{steering_analysis_gemma2b_c2c_probe.json|directions/gold_probe/slope_per_s_at_alpha_1|signedci}} (Gemma-2-2B) and {{steering_analysis_qwen3b_c2c_probe.json|directions/gold_probe/slope_per_s_at_alpha_1|signedci}} (Qwen2.5-3B) on the original pool and {{steering_analysis_gemma2b_c2c_debiased.json|directions/gold_probe/slope_per_s_at_alpha_1|signedci}} and {{steering_analysis_qwen3b_c2c_debiased.json|directions/gold_probe/slope_per_s_at_alpha_1|signedci}} on the debiased pool, while single isotropic directions in the same runs reach {{steering_analysis_gemma2b_c2c_debiased.json|directions/random_0/slope_per_s_at_alpha_1|signed}} (Gemma-2-2B) and {{steering_analysis_qwen3b_c2c_debiased.json|directions/random_0/slope_per_s_at_alpha_1|signed}} (Qwen2.5-3B). The matched controls for these two models are part of the second round of GPU runs and are added to the controls table when they arrive. Gemma's CLadder steering was run on Kaggle's draw of the CLadder sample (Section 7); its slopes are exact, and the direction-dependent columns for those rows use directions refitted on the laptop's draw.

**Steering at the end of the hypothesis (second round; so far Qwen2.5-0.5B).** The second audit pointed out that the probe is stronger at the last token of the hypothesis than at the final prompt token, and that only the latter had been steered. With directions fitted at the end of the hypothesis (layer chosen there on the selection half) and the answer still read at the final token, the gold-probe direction's slope for Qwen2.5-0.5B is {{steering_controls_qwen0.5b_c2c_debiased_hyp.json|real/gold_probe/slope_per_s|signedci}} on the debiased pool ({{steering_controls_qwen0.5b_c2c_debiased_hyp.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} of 20 shuffled-label probe directions have a smaller slope in absolute value) and {{steering_controls_qwen0.5b_c2c_probe_hyp.json|real/gold_probe/slope_per_s|signedci}} on the original pool ({{steering_controls_qwen0.5b_c2c_probe_hyp.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} have a smaller one; per natural standard deviation it exceeds all 20 there, as Qwen2.5-1.5B's did at the final token on the same pool). The model-answer direction exceeds all its controls on the debiased pool and {{steering_controls_qwen0.5b_c2c_probe_hyp.json|real/model_out/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} of them on the original pool. So for this model the earlier site gives the same reading as the final token: the gold-probe direction is within the range of its controls. The rows for both tokens are in the controls table above; the other models follow from the second round.

Flip rates and the earlier "accuracy when pushed towards gold" (now called controllability) are in the next table. They are poor measures for models that answer No to most items: a flip is a threshold on slope × α against the margin, and controllability uses the gold label to choose the sign of the push, so any direction with a consistent slope reaches 0 or 1. Ablating the component along the probe or random directions changes LD by amounts at the float16 noise floor; no ablation comparison is claimed.

<!-- TABLE:steering_flips -->

### 4.2 Patching: where does a change of variable reach the answer position?

**Design.** Token-aligned Corr2Cause minimal pairs with opposite gold labels: *hyp* pairs share the premise and differ in one variable of the hypothesis; *prem* pairs share the hypothesis and have two variable letters swapped throughout the premise. The residual stream of the "clean" run is copied into the "corrupt" run at one layer and one group of token positions. Pooled recovery = Σ sign(D)·(LD_patched − LD_corrupt) / Σ|D|, with D = LD_clean − LD_corrupt.

<!-- TABLE:patching_summary -->

![patching prem](../figures/fig7_patching_prem.png)
![patching hyp](../figures/fig7_patching_hyp.png)

**What the data show.** The models do not prefer the valid member of a pair: mean D is {{patching_qwen0.5b.json|kinds/hyp/behaviour_on_pairs/mean_delta|signedci}} (Qwen2.5-0.5B) and {{patching_qwen1.5b.json|kinds/hyp/behaviour_on_pairs/mean_delta|signedci}} (Qwen2.5-1.5B) for hyp pairs, and D is positive for about half of the pairs. Because recovery is normalised by each pair's own D, the heat-maps show where *any* difference between the two prompts is carried to the final token, not where a causal judgement is made. That transfer is sharp: recovery at the final token first reaches 0.5 at layer 15 of 24 for Qwen2.5-0.5B (relative depth 0.62) and at layer 17 or 18 of 28 for Qwen2.5-1.5B (every second layer was patched; relative depth 0.61 to 0.64). The two pair kinds share their clean prompts and are not independent replications.

Gemma-2-2B, patched on Kaggle with 40 pairs per kind, shows the same thing: mean D {{patching_gemma2b.json|kinds/hyp/behaviour_on_pairs/mean_delta|signedci}} for hyp pairs and {{patching_gemma2b.json|kinds/prem/behaviour_on_pairs/mean_delta|signedci}} for prem pairs, and recovery at the final token first reaching 0.5 at layer 16 of 26 (relative depth 0.62) for both kinds.

**The random-donor row is not a control.** Only 16 of 60 (Qwen2.5-0.5B) and 6 of 40 (Qwen2.5-1.5B) pairs have a donor prompt of equal length, half of the donors carry the same token as the clean prompt at the changed position, and the statistic is not zero under the null. It is shown and not used.

<!-- TABLE:patching_components -->

![attribution](../figures/fig8_attribution_validation.png)

**Components.** Attribution patching (one gradient) agrees with real patching for the ten highest-ranked components of each model and pair kind. Among ten random components the agreement is driven by one or two large ones; without them the two estimates can disagree in sign. So: accurate for the largest components, not a complete ranking. Component names such as L14H5 are block numbers; residual index 15 is the output of block 14.

**Not done.** A label-preserving minimal-pair control (two prompts that differ in a variable but have the same gold label) would separate "the answer changes" from "the prompt changes". It is the experiment this section most needs.

### 4.3 Cross-model comparison

<!-- TABLE:cross_model -->

![cross model](../figures/fig9_cross_model.png)

A descriptive comparison of three models from two families, not a scaling law. On the original item sets gold-probe accuracy is higher for the two larger models than for Qwen2.5-0.5B, and the probe trails the strongest text baseline for every model. The selected layer lies at relative depth 0.6 to 0.9. On the debiased pool (Section 3.6) the difference between Qwen2.5-0.5B and Qwen2.5-1.5B is larger and is the more informative comparison, since it is not inflated by the hypothesis-string cue.

## 5. Tier 3 results

### 5.1 Sparse-autoencoder features for causal graph type (Gemma-2-2B, CLadder)

**Question.** Does Gemma-2-2B have sparse features that mark the causal graph a CLadder question is about (confounding, collision, chain, fork, mediation), and does removing them change the answer more than removing other features?

**Setup.** Gemma Scope residual-stream SAEs (`gemma-scope-2b-pt-res-canonical`, 16k features) at blocks 6, 12 and 18, applied to activations at the last token of the context and at the final prompt token. Only CLadder's ten "nonsense" stories are used ({{sae_features.json|n_nonsense_items|int}} items in the sample), because every real-world story has a single graph type, so a "smoking" feature would pass for a "confounding" feature. For each graph type the ten features with the highest AUROC for that type against the rest are selected on stories 0–4 ({{sae_features.json|n_discovery|int}} items) and evaluated on stories 5–9 ({{sae_features.json|n_eval|int}} items). Ablation (block 12): the selected features' contributions are subtracted from the residual stream at every token and the answer logit difference is compared with the unablated run, against three sets of random features matched on mean activation. All of this ran on Kaggle, on Kaggle's draw of the CLadder sample (Section 7).

**Two caveats that limit everything below.**

1. *The SAEs fit these activations poorly.* They were trained on the base model `gemma-2-2b`; the activations are from the instruction-tuned `gemma-2-2b-it`, at two specific tokens. At the end of the context the reconstruction leaves {{sae_features.json|layers/6/positions/premise_end/reconstruction/fraction_variance_unexplained|num2}}, {{sae_features.json|layers/12/positions/premise_end/reconstruction/fraction_variance_unexplained|num2}} and {{sae_features.json|layers/18/positions/premise_end/reconstruction/fraction_variance_unexplained|num2}} of the variance across items unexplained (blocks 6, 12, 18), although the reconstruction points in nearly the right direction (cosine {{sae_features.json|layers/12/positions/premise_end/reconstruction/cosine_recon|num2}} at block 12). At the final prompt token the unexplained fraction is above one at every block: all items share that token and differ little there, and the SAE does not capture the differences. Features at the final token are listed in the table for completeness and I draw nothing from them.
2. *Three random feature sets are too few.* Their effects range from {{sae_ablation.json|structures/collision/target_structure/mean_abs_delta_random_each/0|num2}} to {{sae_ablation.json|structures/mediation/target_structure/mean_abs_delta_random_each/1|num2}} logits. This is the same weakness the Tier 2 audit found in the first steering table, and I did not fix it before the Kaggle run.

<!-- TABLE:sae_reconstruction -->

**Features.** Selected features generalise to held-out stories about as well as they fit the discovery stories, so the selection is not noise, but they are weak markers of graph type.

<!-- TABLE:sae_features -->

At the end of the context no selected feature reaches a held-out AUROC of 0.70 for confounding or mediation at any block. The best features are for chain (up to {{sae_features.json|layers/12/positions/premise_end/structures/chain/top_features/0/auroc_heldout_stories|num2}}), and the same feature is the best one for fork: at block 12 feature {{sae_features.json|layers/12/positions/premise_end/structures/chain/top_features/0/feature|str}} fires on {{sae_features.json|layers/12/positions/premise_end/structures/chain/top_features/0/firing_rate_target|pct}} of chain items and on {{sae_features.json|layers/12/positions/premise_end/structures/chain/top_features/0/firing_rate_other|pct}} of all other items, and its top-activating examples include collision and mediation items (`results/sae_top_examples.md`). It responds more strongly to contexts that describe the graph as a short list of "has a direct effect on" sentences without probabilities; chain and fork items are over-represented among those. I found no feature that is specific to one graph type.

**Ablation.**

<!-- TABLE:sae_ablation -->

Removing the ten selected features changes the answer logit difference by {{sae_ablation.json|structures/collision/target_structure/mean_abs_delta_top|est}} to {{sae_ablation.json|structures/chain/target_structure/mean_abs_delta_top|est}} logits on items of the features' own graph type. For four of the five graph types that is more than the mean of the three matched random sets (by 0.10 to 0.32 logits); for mediation it is less. The effect is not specific to the graph type: it is as large on items of other graph types (for example chain: {{sae_ablation.json|structures/chain/target_structure/mean_abs_delta_top|est}} on chain items, {{sae_ablation.json|structures/chain/other_structures/mean_abs_delta_top|est}} on others). The change is mostly a uniform shift of the logit difference in one direction (the signed column), that is, a change of answer bias. Accuracy before and after differs by at most eight points on 27 to 40 items, which is inside the noise for samples of that size.

**What may be concluded.** With SAEs that fit poorly, on one sample of nonsense-story items: there are features whose activation is moderately associated with chain- or fork-shaped contexts, they are not specific to a graph type, and removing them shifts the answer logit somewhat more than removing three sets of matched random features, equally for all graph types. This is not evidence of a "causal structure" feature that the answer depends on, and it is not evidence against one either: a dictionary that leaves two thirds of the across-item variance unexplained could miss it.

**Compared with what I expected** (logged before opening the results): I expected the features to track story or question type more than graph type (roughly right: they track how the context is worded) and the ablation effect to be below matched random features in at least half the cases (wrong: above in four of five). I expected no accuracy change beyond three points; the data cannot resolve three points.

**A reporting error of mine.** The result file's `top_minus_random_mean_abs_delta` interval subtracts the absolute value of the mean over random sets, not the mean of their absolute effects, and so overstates the gap (for confounding it reads +0.25 where the two means in the table differ by +0.15). The table shows the difference of the two means without an interval; the raw per-item values were not exported, so a correct interval needs a rerun.

LoRA fine-tuning (the optional second half of Tier 3) was dropped when the heavy runs moved to Kaggle (decision D20).

## 6. Audits

Each tier was reviewed by a fresh reviewer agent that received the code, the result files and the raw outputs, but not the author's conclusions. Raw reports: `notes/audit_tier1_raw.md`, `notes/audit_tier2_raw.md`, `notes/audit_debiased_raw.md`. Responses, finding by finding: `notes/log.md`.

**Tier 1 audit (Friday 00:05; Qwen models).** Confirmed: no fold leakage, no premise shared between pool and test, baselines fitted on training folds only, positions, layer indexing and sign conventions, all tables regenerate from the result files. Found:

| Finding | Consequence |
|---|---|
| The hypothesis string alone predicts the label at 0.86 in the balanced pool; my baselines (whole-prompt TF-IDF, six-template prior) were too weak | Hypothesis-string, question-only and parsed-feature baselines added; "audited criterion" = probe versus strongest text baseline; the one positive verdict was retracted |
| "Accuracy above 0.5 on items the model gets wrong" is not a chance test when those items are mostly one class | Balanced accuracy and class balance reported; criterion flagged not applicable outside 20–80% balance |
| With model answers nearly independent of the gold label, the wrong-items design reduces to probe versus text baseline | Said plainly; audited criterion is exactly that comparison |
| On CLadder the probe's signal is present before the question is read | Probe accuracy by token position added |
| The gold-signed logit-lens figure showed answer bias, not knowledge | Figure replaced; AUROC version is the headline |
| Stale figures and result files | Everything regenerated |

**Tier 2 audit (Friday 11:37; Qwen models).** Confirmed: every number reproduces from the raw outputs; selection and report halves are disjoint; layer indexing is consistent; the key-value-cache shortcut is equivalent to a full forward pass by code reading. The claim I had asked it to attack ("the gold-probe direction is inert while the model-output direction flips answers") **did not survive as stated**:

| Finding | Consequence |
|---|---|
| For Qwen2.5-1.5B on Corr2Cause the gold-probe direction has a consistent, wrong-signed effect that my table hid by showing one sign only | Re-analysis with signed slopes and both signs (`scripts/07b_steering_analysis.py`) |
| Three random directions cannot support "no more than random" | 100 isotropic and 2 × 20 label-shuffled control directions (`scripts/07c_steering_controls.py`) |
| Steps were tens to hundreds of the probe direction's natural SDs; `s` was mislabelled as a typical deviation | Natural-scale quantities reported; labels corrected |
| The probe weights and the difference of class means are two views of one discriminant; "weights inert" is not "information unused" | Wording |
| The difference-of-means effect is not specific to the gold label | Shuffled-label control; wording |
| CLadder steering rows are vacuous (the direction barely decodes gold) | Conclusions restricted to Corr2Cause |
| Qwen2.5-0.5B steering had been run at an outdated layer and regularisation | Rerun |
| Flip rate and "accuracy towards gold" are poor metrics for near-constant answerers | Signed slope is the headline; column renamed "controllability" |
| Ablation effects are at the float16 noise floor | No ablation claim; noise floor reported |
| Patching localises where a prompt difference reaches the final token, not where the answer is determined; the random-donor control is invalid as run | Section retitled and reworded; control removed from the argument |
| Attribution patching is validated for the largest components only | Wording |

**Third audit (Friday 14:45; the debiased-pool analysis and the steering wording).** Raw report: `notes/audit_debiased_raw.md`. Confirmed with its own code: the pool is exactly balanced for every hypothesis string and premise in all 30 training and test folds; the headline probe accuracy reproduces (0.752 [0.730, 0.775]); no selection contamination; the parser is correct; steering numbers reproduce. Four of the seven sentences I had proposed did not survive:

| Finding | Consequence |
|---|---|
| My test of what the probe adds to the parser (one extra column in gradient boosting) could not detect a two-point gain; I had concluded "nothing measurable" | Nested logistic stacking: about two points over the hand parse, nothing over a stronger text-only model (Section 3.6.2) |
| The "shallow confound controls" I listed are at 0.50 by construction in this pool and prove nothing | A ladder of interaction baselines; the probe is placed on it (Section 3.6.1) |
| "The answers do not express this information" is false below the pooled level | Per-template analysis of the answers (Section 3.6.5) |
| Twenty control directions support "within the range of", not "no more than"; the effect is not zero; per natural SD one case exceeds all controls | Section 4.1 reworded; per-natural-SD percentiles added to the table |
| For Qwen2.5-0.5B the final-token signal is absent, not "weaker" | Wording |
| Adjacency is largely decodable from per-variable statement counts; single shuffles are not a null distribution; near-duplicate properties; uninformative strata | Text baselines per property, end-of-premise check, repeated shuffles, merged properties, score strata (Sections 3.6.3, 3.6.4) |

**What the audits did not cover.** Gemma-2-2B's Tier 1 results were produced after the Tier 1 audit by the same, already-audited code and were reviewed by the author only. The new experiment code for the second round of GPU runs was reviewed by a fourth independent reviewer before those runs were launched (`notes/audit_round2_code_raw.md`); the results of the first Kaggle run (Gemma steering, patching and SAEs; Qwen2.5-3B) and of the second round are covered by that audit's follow-up. *(Findings and responses are added below as they arrive.)*

## 7. Limitations

- **Models and scale.** Three models of 0.5 to 2.6 billion parameters (plus one larger model run remotely), all at chance on both tasks. Nothing here says what a model that can do the task would show.
- **One probe family.** Linear probes (L2 logistic regression) at three token positions. A non-linear or multi-position read-out could find more; a negative probing result is weak evidence.
- **Post-hoc analysis.** The stronger text baselines, the audited criterion and the debiased pool were introduced after seeing results (decisions D18, D19, D21). They make the claim harder to pass, but they are not pre-registered.
- **Multiplicity.** Many model × dataset × criterion cells are reported with unadjusted 95% intervals.
- **Few clusters.** The Corr2Cause test set has 21 premises, CLadder 47 stories (24 in a report half); cluster-bootstrap intervals with so few clusters under-cover.
- **Numerics.** Activations are cached in float16 from float16 models; for the two larger local models no float32 reference fits in memory. Probe fits at C = 1 are stopped at 300 iterations (bias against the probe of under one point on the one fold checked).
- **Two machines.** Gemma's probes on the original item sets use activations cached on the laptop; its steering, patching, SAE results and its debiased-pool activations were computed on a Kaggle GPU. Agreement of the answer logits between the two is reported in `results/cache_agreement_gemma2b.json`.
- **Two draws of the CLadder sample.** The stratified sampler broke a four-way tie differently on the laptop and on Kaggle (decision D28), so Qwen2.5-3B's CLadder results and Gemma's CLadder steering and SAE results are computed on a different draw of 3,000 items (2,271 shared; same stratification). They are valid on their own but not item-for-item comparable with the laptop models' CLadder results. Nothing on Corr2Cause is affected. The sample is now pinned.
- **Remote caches hold the final token only.** For the larger model, and for Gemma on the debiased pool, the position comparison is unavailable.
- **Steering.** One layer (plus two fixed depths for some models), final token only; directions fitted on about 1,200 items; float16.
- **Patching.** 40 to 60 minimal pairs from at most five-variable premises; the models do not distinguish the pairs by gold label; every second layer for the 1.5B model.
- **Gemma generation check** is first-token-only on 100 items per set.
- **Labels as released.** Public reports of label errors in both benchmarks exist and were not checked.
- **CLadder few-shot demonstrations** come from stories that also occur among the evaluated items.
- **No fine-tuning tier.** The planned LoRA comparison was dropped for compute reasons (decision D20).

## 8. Interpretation and what I'd do next

*(Deliberately left empty: to be written by the project owner.)*

## References

- Jin, Z., Liu, J., Lyu, Z., Poff, S., Sachan, M., Mihalcea, R., Diab, M., Schölkopf, B. (2023a). Can Large Language Models Infer Causation from Correlation? arXiv:2306.05836 (ICLR 2024).
- Jin, Z., Chen, Y., Leeb, F., Gresele, L., Kamal, O., Lyu, Z., Blin, K., Gonzalez, F., Kleiman-Weiner, M., Sachan, M., Schölkopf, B. (2023b). CLadder: Assessing Causal Reasoning in Language Models. arXiv:2312.04350 (NeurIPS 2023).
- Hewitt, J., Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP.
- nostalgebraist (2020). Interpreting GPT: the logit lens.
- Lieberum, T. et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147.
- Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Nanda, N. (2023). Attribution Patching: Activation Patching At Industrial Scale.

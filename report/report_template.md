# Do small LLMs represent causal answers they fail to give? A probing, steering and patching study

Technical report. Every number in a table comes from `results/*.json` via `scripts/12_tables.py`; the source file is named under each table. Figures are drawn from the same files by `scripts/06_figures.py`. For background on any method see `EXPLAINER.md`.

## 1. Question and summary

<!-- SUMMARY -->

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

Apple M3, 8 GB unified memory. Models: Qwen2.5-0.5B-Instruct (24 layers, width 896), Qwen2.5-1.5B-Instruct (28, 1536), Gemma-2-2B-it (26, 2304), each with its own chat template (single user turn; Qwen's default system prompt). Qwen2.5-3B does not fit. Models run in float16 on MPS through plain Hugging Face modules with forward hooks; TransformerLens 4.4's bridge was verified to give the same logits (max difference 7e-5 on five prompts) and not used further. MPS float16 agrees with CPU float32 to 0.02 in the answer logit difference for the 0.5B model; for the two larger models a float32 reference does not fit in memory and the check is against CPU half precision (`results/env_check_*.json`).

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

**No model answers above chance, and for different reasons.** On the Corr2Cause test set (15.5% valid) Qwen2.5-0.5B answers "No" to essentially everything (Yes-rate {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|num2}}), so its accuracy {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/accuracy|est}} is the majority rate and its macro-F1 {{behaviour_qwen0.5b.json|zero_shot/c2c_test/logit_scored/macro_f1|est}} is the majority baseline's. Qwen2.5-1.5B says Yes {{behaviour_qwen1.5b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|pct}} of the time and Gemma-2-2B {{behaviour_gemma2b.json|zero_shot/c2c_test/logit_scored/frac_pred_yes|pct}}; their accuracies ({{behaviour_qwen1.5b.json|zero_shot/c2c_test/logit_scored/accuracy|ci}}, {{behaviour_gemma2b.json|zero_shot/c2c_test/logit_scored/accuracy|ci}}) are *below* the majority rate because the extra Yes answers are mostly wrong. On CLadder (50% yes) all three are at 0.50.

Accuracy alone cannot tell "no information" from "information hidden by a bias towards one answer". The next table separates them: the Yes-rate, the AUROC of the Yes−No logit difference (which ignores any constant bias), and the accuracy when the decision threshold is moved to its best position (chosen on one half of the premises or stories, scored on the other).

<!-- TABLE:answer_bias -->

Reading. On the original Corr2Cause sets the logit difference is weakly informative: AUROC {{answer_bias.json|qwen0.5b/c2c_test/auroc|ci}} for Qwen2.5-0.5B, {{answer_bias.json|qwen1.5b/c2c_test/auroc|ci}} for Qwen2.5-1.5B on test, and 0.56 to 0.59 on the balanced pool for all three, where a re-thresholded answer reaches about 0.57 accuracy. So the smallest model is biased rather than empty there. But this weak signal does not survive the two manipulations that remove surface cues: for Qwen2.5-0.5B the AUROC is {{answer_bias.json|qwen0.5b/c2c_test_renamed/auroc|ci}} after renaming the variables and {{answer_bias.json|qwen0.5b/c2c_debiased/auroc|ci}} on the debiased pool of Section 3.6. On CLadder every model is at chance by AUROC as well (0.50 to 0.52). The behavioural picture is therefore: at chance on the task, with a small shortcut-driven signal on the original Corr2Cause items that is masked by answer bias.

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

The first figure shows the Yes−No logit difference the model would produce if it stopped at each layer, for gold-Yes and gold-No items separately. In every panel the curves move together: the lens shows a shared drift towards one answer, not a separation by the correct answer. The second figure removes the drift by asking only whether the lens value ranks gold-Yes above gold-No items (AUROC). On the Corr2Cause pool it never exceeds 0.60 at any layer for any model (the maxima are 0.59 to 0.60, reached in the second half of the network); on CLadder it stays between 0.47 and 0.53 at every layer. The lens at the last layer reproduces the model's actual answer logits (maximum error and sign agreement are in `results/lens_*.json`). So the lens gives no sign of a correct answer that forms in the middle of the network and is then overwritten.

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

The external check agrees: a probe trained on the pool and applied to the natural-distribution test set ranks valid above invalid items well (AUROC in the table), but so does the hypothesis-string prior on the same items ({{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}).

### 3.5 What the text alone gives away

The full analysis, with data versions, definitions and what is and is not already known, is in `report/benchmark_text_baselines.md` and is reproducible from one script (`scripts/14_benchmark_baselines.py`). The two tables that matter for this study:

<!-- TABLE:benchmark_c2c_pools -->

<!-- TABLE:benchmark_cladder -->

On the balanced Corr2Cause pool the hypothesis string alone predicts the label with {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|ci}} accuracy; this follows from how the benchmark names variables (causes get earlier letters), which makes 72 of the 120 hypothesis strings never valid. On CLadder, the quoted numbers and the polarity of the question predict {{benchmark_baselines.json|cladder_full_v1.5/baselines/numeric_polarity/accuracy|ci}} on held-out stories, a result largely anticipated by Clarke et al. (2026). These are properties of how text-only classifiers perform on the released data; for a probing study they set the bar a probe has to clear.

### 3.6 A fairer test: the debiased Corr2Cause pool

The original pool cannot separate "represents the answer" from "represents the hypothesis". The debiased pool (D21) removes every single-input route: 2,400 items in which each premise contributes one valid and one invalid hypothesis and each of the 48 usable hypothesis strings is half valid in every cross-validation fold. A classifier that sees only the hypothesis or only the premise scores exactly 0.50 by construction, and whole-prompt TF-IDF scores {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/full_text_tfidf/accuracy|ci}}. The parsed-feature classifier still reaches {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/parsed_features/accuracy|ci}}: it binds the premise's statements to the two variables named in the hypothesis, which is the first step of the task itself.

<!-- T1-DEBIASED-TEXT -->

## 4. Tier 2 results

### 4.1 Steering and ablation: is the decodable direction used?

<!-- TABLE:steering -->

![steering c2c](../figures/fig6_steering_c2c_probe.png)
![steering cladder](../figures/fig6_steering_cladder.png)

<!-- T2-STEERING-TEXT -->

### 4.2 Patching: where is the answer determined?

<!-- TABLE:patching_summary -->

![patching prem](../figures/fig7_patching_prem.png)
![patching hyp](../figures/fig7_patching_hyp.png)

<!-- TABLE:patching_components -->

![attribution](../figures/fig8_attribution_validation.png)

<!-- T2-PATCHING-TEXT -->

### 4.3 Cross-model comparison

<!-- TABLE:cross_model -->

![cross model](../figures/fig9_cross_model.png)

<!-- T2-CROSS-TEXT -->

## 5. Tier 3 results

<!-- T3-TEXT -->

## 6. Audits

<!-- AUDITS -->

## 7. Limitations

<!-- LIMITATIONS -->

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

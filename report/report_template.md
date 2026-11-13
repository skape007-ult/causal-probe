# Do small LLMs represent causal answers they fail to give? A probing, steering and patching study

Technical report. Every number in a table comes from `results/*.json` via `scripts/12_tables.py`; the source file is named under each table. Figures are drawn from the same files by `scripts/06_figures.py`. For background on any method see `EXPLAINER.md`.

## 1. Question and summary

**Question.** Do small open-weight language models internally represent the correct answer to formal causal-reasoning questions they answer wrongly; is that representation used; and where is the answer formed?

**Summary of findings.**

1. *Behaviour.* Qwen2.5-0.5B, Qwen2.5-1.5B and Gemma-2-2B are at chance on Corr2Cause and CLadder. Part of the picture is answer bias; the bias-free ranking score is at or near 0.5 as well, and the small signal on the original Corr2Cause items disappears when surface cues are removed (Section 3.1).
2. *First probing result and its retraction.* Linear probes decode the gold label at {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} to {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} on a balanced Corr2Cause pool, and the pre-registered criterion for "represents but doesn't say" passed as coded in some cells. An independent audit showed the criterion's baselines were too weak: the hypothesis string alone predicts the label at {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|est}} on that pool. Against the strongest text-only baseline the probes lose in all six model × benchmark cells. The positive verdict was retracted (Section 3.4).
3. *Text-only baselines on the benchmarks.* On the official Corr2Cause test split a hypothesis-only classifier reaches AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}} and a parsed-feature classifier accuracy {{benchmark_baselines.json|corr2cause_official_splits/test/parsed_features/accuracy|ci}}; on CLadder, quoted numbers plus question polarity reach {{benchmark_baselines.json|cladder_full_v1.5/baselines/numeric_polarity/accuracy|ci}} on held-out stories. The Corr2Cause effect follows from the benchmark's published construction; the CLadder one is largely documented by prior work (Section 3.5 and `report/benchmark_text_baselines.md`).
4. *A fairer test.* On a debiased Corr2Cause pool, in which the hypothesis alone and the premise alone carry no label information in any fold, the gold probe scores {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} for Qwen2.5-0.5B and {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} for Qwen2.5-1.5B, with both models answering at chance. For the 1.5B model this is well above every shallow confound control (strongest: a probe on a randomly initialised copy, {{confound_qwen1.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|est}}), below a parsed-feature classifier ({{confound_qwen1.5b_c2c_debiased.json|all_items/structural/accuracy|est}}), and adds nothing measurable to it. Properties that require binding premise statements to the queried variables are readable from the trained model and not from a random one. The supported statement is that this model encodes part of the premise's structure that its answers do not express; the audited criterion for the stronger claim is not met (Section 3.6; post hoc).
5. *Steering.* At the probed layer, adding the gold probe's weight direction changes the answer no more than adding a probe direction fitted to shuffled labels, and a movement of natural size along it changes nothing. The difference-of-class-means direction does move the answer, but about as much as differences of means for shuffled labels. This does not establish that the decoded information is unused (Section 4.1; rewritten after the second audit).
6. *Patching.* A change of one variable in the prompt reaches the answer position at about 0.6 of the network's depth in both Qwen models. This locates the arrival of a prompt difference, not a causal judgement (Section 4.2).
7. *Remote runs.* Gemma-2-2B interventions and SAE features, and all analyses for Qwen2.5-3B, are pending the Kaggle run (Sections 3.6, 4, 5).

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

Gold-label probes on the debiased pool (report half; layer and regularisation chosen on the other half; hypothesis-only and premise-only classifiers are at 0.500 by construction):

| Model | Gold probe, final token | Random-init model's probe, final token | Whole-prompt TF-IDF | Parsed features | Probe − random-init | Probe − parsed features | Gold probe, end of hypothesis | Random-init, end of hypothesis |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_qwen0.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | {{confound_qwen0.5b_c2c_debiased.json|positions/content_end/accuracy|ci}} | {{confound_qwen0.5b_c2c_debiased.json|positions/content_end/random_init_accuracy|ci}} |
| Qwen2.5-1.5B | {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_qwen1.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | {{confound_qwen1.5b_c2c_debiased.json|positions/content_end/accuracy|ci}} | {{confound_qwen1.5b_c2c_debiased.json|positions/content_end/random_init_accuracy|ci}} |
| Gemma-2-2B (activations from Kaggle) | {{confound_gemma2b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_gemma2b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | {{confound_gemma2b_c2c_debiased.json|positions/content_end/accuracy|ci}} | {{confound_gemma2b_c2c_debiased.json|positions/content_end/random_init_accuracy|ci}} |
| Qwen2.5-3B (activations from Kaggle) | {{confound_qwen3b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/random_init_probe/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/tfidf/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/structural/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|all_items/gold_probe_minus_random_init_probe/accuracy|signedci}} | {{confound_qwen3b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} | {{confound_qwen3b_c2c_debiased.json|positions/content_end/accuracy|ci}} | {{confound_qwen3b_c2c_debiased.json|positions/content_end/random_init_accuracy|ci}} |

<sub>Source: `results/confound_<model>_c2c_debiased.json`. "(pending: …)" marks runs that have not finished.</sub>

**Verdicts under the two criteria, unchanged.** The audited criterion (probe beats the strongest text-only baseline; D19) is **not met** for either model: the parsed-feature classifier beats the probe by {{confound_qwen0.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} (Qwen2.5-0.5B) and {{confound_qwen1.5b_c2c_debiased.json|verdict/audited/all_items_probe_minus_best_text_accuracy|signedci}} (Qwen2.5-1.5B). The original pre-registered criterion as coded (D11: against TF-IDF and the template prior, on wrongly answered items) is met for both models on this pool; as explained in Section 3.4 it does not test the claim, and it is reported, not relied on.

#### 3.6.1 Post-hoc reading: two kinds of baseline

*Everything in 3.6.1 to 3.6.4 was designed after the debiased-pool probe results were seen (13 November; `scripts/05b_probe_content.py`, `notes/log.md`). It is a labelled post-hoc analysis and changes neither criterion above.*

The text baselines are of two different kinds, and comparing a probe with them answers two different questions.

- **(a) Shallow confound controls**: bag-of-words of the prompt, the hypothesis-string prior, the premise alone, a probe on a randomly initialised copy of the model, and the probe trained on shuffled labels. Beating these shows that the probe is not exploiting surface statistics of the prompt, the architecture alone, or a leak.
- **(b) Structure-aware reference**: the parsed-feature classifier. It binds the premise's statements to the two queried variables, which approximates solving the task from the text. It is not a confound; it marks how far explicit structure extraction gets.

<!-- TABLE:debiased_two_groups -->

For Qwen2.5-1.5B the probe ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}) is above every shallow control, the strongest being the random-init probe ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/shallow_confound_controls/probe_on_random_init_model/accuracy|ci}}; probe minus control {{probe_content_qwen1.5b_c2c_debiased.json|baselines/shallow_confound_controls/probe_on_random_init_model/probe_minus_this_accuracy|signedci}}), and below the structure-aware reference ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/structure_aware_reference/parsed_features/accuracy|ci}}). For Qwen2.5-0.5B the probe ({{probe_content_qwen0.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}) exceeds the random-init probe by only {{probe_content_qwen0.5b_c2c_debiased.json|baselines/shallow_confound_controls/probe_on_random_init_model/probe_minus_this_accuracy|signedci}}. Both models' own answers on this pool are at chance (AUROC {{answer_bias.json|qwen0.5b/c2c_debiased/auroc|ci}} and {{answer_bias.json|qwen1.5b/c2c_debiased/auroc|ci}}). For Qwen2.5-1.5B, probe accuracy stays at the random-init level through the first half of the network and rises between layers 14 and 22 (`results/probes_qwen1.5b_c2c_debiased_C1.json`).

#### 3.6.2 Does the probe add predictive power beyond the parsed features?

Gradient boosting on the parsed features alone, against the same model with the probe's score added as one column. The probe score is produced by nested cross-validation, so that no label from a test fold reaches any feature.

<!-- TABLE:debiased_incremental -->

No measurable gain: {{probe_content_qwen1.5b_c2c_debiased.json|incremental_value/gain_accuracy|signedci}} accuracy for Qwen2.5-1.5B and {{probe_content_qwen0.5b_c2c_debiased.json|incremental_value/gain_accuracy|signedci}} for Qwen2.5-0.5B. What the probe reads is, as far as this test can tell, contained in what the parser extracts. (A cruder version of this test, a logistic regression on the two out-of-fold scores, had suggested a two-point gain; it was not nested and is withdrawn, see `notes/log.md`.)

#### 3.6.3 Which parsed properties can be read from the activations?

A probe for each binary property a parser reads off the item, at the layer the gold probe uses, with a shuffled-label control and the same probe on the random-init model.

<!-- TABLE:debiased_properties -->

Properties of the *queried pair* that require binding premise statements to the two hypothesis variables are decodable from the trained models far better than from random-init copies. For Qwen2.5-1.5B, "the pair is adjacent in the skeleton" has AUROC {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/trained/auroc|ci}} against {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/pair_adjacent_in_skeleton/at_gold_probe_layer/random_init/auroc|ci}} at random init, and "a common neighbour lies in a separating set" {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/common_neighbour_in_a_separating_set/at_gold_probe_layer/trained/auroc|ci}} against {{probe_content_qwen1.5b_c2c_debiased.json|properties/by_property/common_neighbour_in_a_separating_set/at_gold_probe_layer/random_init/auroc|ci}}. Premise-level counts (how many statements, six variables) are perfectly decodable from the random-init model as well, so they reflect prompt length and say nothing about computation. Shuffled-label controls are at 0.5 except where the property is rare (a small-sample effect of shuffling an imbalanced label).

#### 3.6.4 Is the probe's signal carried by one or two properties?

The gold probe's AUROC inside strata of each strong property, weighted over strata, next to its unstratified AUROC. If one property carried the signal, the AUROC inside its strata would fall towards 0.5.

<!-- TABLE:debiased_strata -->

It does not fall for any single property: for Qwen2.5-1.5B {{probe_content_qwen1.5b_c2c_debiased.json|stratified/by_stratifier/pair_adjacent_in_skeleton/pair_weighted_within_stratum_probe_auroc|ci}} inside adjacency strata against {{probe_content_qwen1.5b_c2c_debiased.json|stratified/overall_probe_auroc|ci}} overall. This matches the structure of the pool, in which no single property predicts the label (last column of the previous table: 0.51 to 0.55) and validity depends on their combination with the hypothesis template. Inside the joint strata of template × adjacency × common neighbour, which together predict the label at {{probe_content_qwen1.5b_c2c_debiased.json|stratified/by_stratifier/template_x_adjacent_x_common_neighbour/gold_accuracy_from_stratifier_alone|num}}, the probe's AUROC is {{probe_content_qwen1.5b_c2c_debiased.json|stratified/by_stratifier/template_x_adjacent_x_common_neighbour/pair_weighted_within_stratum_probe_auroc|ci}}; inside the parsed classifier's two prediction strata it is {{probe_content_qwen1.5b_c2c_debiased.json|stratified/by_stratifier/parsed_classifier_prediction/pair_weighted_within_stratum_probe_auroc|ci}}. The probe is reading a combination of structural properties, not one of them.

#### 3.6.5 What this supports

For Qwen2.5-1.5B on the debiased pool: the activations encode part of the premise's structure as it bears on the queried pair, beyond every shallow control, built up between layers 14 and 22; less than a hand-written parse of the prompt, and nothing measurable beyond that parse; not reducible to one or two parsed properties; and not expressed in the model's answers, which are at chance. For Qwen2.5-0.5B the same pattern is much weaker and, at the final token, barely distinguishable from a random network. Whether the encoded information is *used* by the model is a causal question; the steering evidence on it is in Section 4.1. Gemma-2-2B and Qwen2.5-3B are added when their activations arrive from the Kaggle run. All of 3.6.1 to 3.6.4 is post hoc and is the subject of the third independent audit.

## 4. Tier 2 results

This section was rewritten after the second independent audit (Section 6), which showed that my first summary of the steering results ("the gold-probe direction is inert, the model-output direction flips answers") did not survive scrutiny as stated. The analysis below uses the audit's metrics and adds the controls it asked for. Results for Gemma-2-2B and Qwen2.5-3B come from the Kaggle run and are added when it returns; that run used three random directions only, so the matched controls exist for the two local Qwen models.

### 4.1 Steering: what does adding a decodable direction do to the answer?

**Design.** At one layer (the gold probe's selected layer) and the final prompt token, add ±α·s·d for a unit direction d and read the Yes−No answer logit difference (LD). s is the RMS distance of that layer's activations from their mean, a norm over all dimensions. Directions are fitted on the selection half and applied to 300 report-half items (150 valid, 150 invalid). Three real directions: the gold-label probe's weight vector; the difference between the mean activation of valid and invalid items; and, as a positive control, the weight vector of a probe for the model's own (median-split) answer, which is fitted to the very quantity being measured. The headline number is the **slope**, (LD(+α) − LD(−α)) / 2α at α = 1, in logits per unit of s.

<!-- TABLE:steering -->

![steering c2c](../figures/fig6_steering_c2c_probe.png)
![steering debiased](../figures/fig6_steering_c2c_debiased.png)

**Matched controls** (150 report-half items for Qwen2.5-0.5B, 100 for Qwen2.5-1.5B): 100 isotropic random directions; 20 probe weight vectors fitted to shuffled labels, the control of the same kind as a probe direction; 20 differences of class means for shuffled labels, the control of the same kind as a difference of means.

<!-- TABLE:steering_controls -->

**What the data show.**

1. *The gold-probe weight direction does no more than label-free directions of the same kind.* On the original pool its slope is {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/slope_per_s|signedci}} (percentile {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label probe directions; their median |slope| {{steering_controls_qwen0.5b_c2c_probe.json|families/shuffled_probe/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen0.5b_c2c_probe.json|families/shuffled_probe/abs_slope_per_s/max|num}}) for Qwen2.5-0.5B and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/slope_per_s|signedci}} (percentile {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label probe directions; their median |slope| {{steering_controls_qwen1.5b_c2c_probe.json|families/shuffled_probe/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen1.5b_c2c_probe.json|families/shuffled_probe/abs_slope_per_s/max|num}}) for Qwen2.5-1.5B. On the debiased pool: {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/slope_per_s|signedci}} (percentile {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label probe directions; their median |slope| {{steering_controls_qwen0.5b_c2c_debiased.json|families/shuffled_probe/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen0.5b_c2c_debiased.json|families/shuffled_probe/abs_slope_per_s/max|num}}) and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/slope_per_s|signedci}} (percentile {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label probe directions; their median |slope| {{steering_controls_qwen1.5b_c2c_debiased.json|families/shuffled_probe/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen1.5b_c2c_debiased.json|families/shuffled_probe/abs_slope_per_s/max|num}}). In no case is it beyond the 95th percentile of its matched controls. For Qwen2.5-1.5B on the original pool the effect is consistent in sign across items and *opposite* to the gold label (pushing "towards valid" lowers the Yes logit); my first table showed one sign only and hid this. It is within what shuffled-label probe directions do, so it is not evidence of a gold-specific mechanism in either direction.
2. *At natural scale the probe direction does essentially nothing.* A probe's weight vector points into low-variance directions: one unit of s is {{steering_analysis_qwen0.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|num2}} (Qwen2.5-0.5B) and {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/alpha_1_in_own_sd|num2}} (Qwen2.5-1.5B) natural standard deviations along it. Moving an item across the whole natural gap between valid and invalid items along that direction would change LD by {{steering_analysis_qwen0.5b_c2c_probe.json|directions/gold_probe/effect_of_natural_class_gap_on_logit_diff|signed}} and {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_probe/effect_of_natural_class_gap_on_logit_diff|signed}} logits, against a standard deviation of LD across items of {{steering_analysis_qwen0.5b_c2c_probe.json|baseline/logit_diff_sd|num2}} and {{steering_analysis_qwen1.5b_c2c_probe.json|baseline/logit_diff_sd|num2}}.
3. *The difference-of-means direction moves the answer, but not specifically because it separates valid from invalid items.* Its slope is {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/slope_per_s|signedci}} (percentile {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label differences of means; their median |slope| {{steering_controls_qwen0.5b_c2c_probe.json|families/shuffled_diffmeans/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen0.5b_c2c_probe.json|families/shuffled_diffmeans/abs_slope_per_s/max|num}}) for Qwen2.5-0.5B and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_diffmeans/slope_per_s|signedci}} (percentile {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label differences of means; their median |slope| {{steering_controls_qwen1.5b_c2c_probe.json|families/shuffled_diffmeans/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen1.5b_c2c_probe.json|families/shuffled_diffmeans/abs_slope_per_s/max|num}}) for Qwen2.5-1.5B on the original pool; on the debiased pool {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_diffmeans/slope_per_s|signedci}} (percentile {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label differences of means; their median |slope| {{steering_controls_qwen0.5b_c2c_debiased.json|families/shuffled_diffmeans/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen0.5b_c2c_debiased.json|families/shuffled_diffmeans/abs_slope_per_s/max|num}}) and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/slope_per_s|signedci}} (percentile {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_diffmeans/versus/shuffled_diffmeans/percentile_of_abs_slope_per_s|pct}} among 20 shuffled-label differences of means; their median |slope| {{steering_controls_qwen1.5b_c2c_debiased.json|families/shuffled_diffmeans/abs_slope_per_s/median|num}}, maximum {{steering_controls_qwen1.5b_c2c_debiased.json|families/shuffled_diffmeans/abs_slope_per_s/max|num}}). A difference of class means lies in the high-variance part of activation space, and so does a difference of means for random labels. On the original pool this direction also separates the model's own answer at least as well as the gold label (AUROC {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_diffmeans/auroc_model_answer_report_half|num}} against {{steering_analysis_qwen1.5b_c2c_probe.json|directions/gold_diffmeans/auroc_gold_report_half|num}} for Qwen2.5-1.5B). Only for Qwen2.5-0.5B on the original pool does it exceed all 20 shuffled controls, and that pool is the one in which the gold label is confounded with the hypothesis string.
4. *The positive control works.* The model-answer direction has the largest slope everywhere and exceeds every control, as it must.

**What may and may not be concluded.**

- May: at the probed layer and the final token, later layers do not read the gold probe's weight direction more than they read a label-free direction of the same kind, and a movement of natural size along it leaves the answer unchanged.
- May not: "the decoded information is unused". The probe's weights and the difference of class means are two views of one discriminant (the covariance times the probe weights has cosine {{steering_analysis_qwen1.5b_c2c_probe.json|geometry/cos_covariance_times_probe_vs_diffmeans|num2}} with the difference of means for Qwen2.5-1.5B); one site and one position were tested; and the activation pattern that does move the answer was not gold-specific.
- The most direct evidence that the information of Section 3.6 is not used is behavioural: on the debiased pool the answers are at chance while the probe decodes validity at {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe/accuracy|num2}} and {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|num2}}.
- CLadder: the steered gold directions barely decode the gold label there (AUROC {{steering_analysis_qwen0.5b_cladder.json|directions/gold_probe/auroc_gold_report_half|num}} and {{steering_analysis_qwen1.5b_cladder.json|directions/gold_probe/auroc_gold_report_half|num}}), so those rows are shown for completeness and no conclusion is drawn from them.

Flip rates and the earlier "accuracy when pushed towards gold" (now called controllability) are in the next table. They are poor measures for models that answer No to most items: a flip is a threshold on slope × α against the margin, and controllability uses the gold label to choose the sign of the push, so any direction with a consistent slope reaches 0 or 1. Ablating the component along the probe or random directions changes LD by amounts at the float16 noise floor; no ablation comparison is claimed.

<!-- TABLE:steering_flips -->

### 4.2 Patching: where does a change of variable reach the answer position?

**Design.** Token-aligned Corr2Cause minimal pairs with opposite gold labels: *hyp* pairs share the premise and differ in one variable of the hypothesis; *prem* pairs share the hypothesis and have two variable letters swapped throughout the premise. The residual stream of the "clean" run is copied into the "corrupt" run at one layer and one group of token positions. Pooled recovery = Σ sign(D)·(LD_patched − LD_corrupt) / Σ|D|, with D = LD_clean − LD_corrupt.

<!-- TABLE:patching_summary -->

![patching prem](../figures/fig7_patching_prem.png)
![patching hyp](../figures/fig7_patching_hyp.png)

**What the data show.** The models do not prefer the valid member of a pair: mean D is {{patching_qwen0.5b.json|kinds/hyp/behaviour_on_pairs/mean_delta|signedci}} (Qwen2.5-0.5B) and {{patching_qwen1.5b.json|kinds/hyp/behaviour_on_pairs/mean_delta|signedci}} (Qwen2.5-1.5B) for hyp pairs, and D is positive for about half of the pairs. Because recovery is normalised by each pair's own D, the heat-maps show where *any* difference between the two prompts is carried to the final token, not where a causal judgement is made. That transfer is sharp: recovery at the final token first reaches 0.5 at layer 15 of 24 for Qwen2.5-0.5B (relative depth 0.62) and at layer 17 or 18 of 28 for Qwen2.5-1.5B (every second layer was patched; relative depth 0.61 to 0.64). The two pair kinds share their clean prompts and are not independent replications.

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

<!-- T3-TEXT -->

## 6. Audits

Each tier was reviewed by a fresh reviewer agent that received the code, the result files and the raw outputs, but not the author's conclusions. Raw reports: `notes/audit_tier1_raw.md`, `notes/audit_tier2_raw.md`. Responses, finding by finding: `notes/log.md`.

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

**What the audits did not cover.** Gemma-2-2B's Tier 1 results were produced after the Tier 1 audit by the same, already-audited code and were reviewed by the author only. Results from the Kaggle run (Gemma steering, patching and SAEs; the larger model) are covered by a third audit after they arrive. *(Pending.)*

## 7. Limitations

- **Models and scale.** Three models of 0.5 to 2.6 billion parameters (plus one larger model run remotely), all at chance on both tasks. Nothing here says what a model that can do the task would show.
- **One probe family.** Linear probes (L2 logistic regression) at three token positions. A non-linear or multi-position read-out could find more; a negative probing result is weak evidence.
- **Post-hoc analysis.** The stronger text baselines, the audited criterion and the debiased pool were introduced after seeing results (decisions D18, D19, D21). They make the claim harder to pass, but they are not pre-registered.
- **Multiplicity.** Many model × dataset × criterion cells are reported with unadjusted 95% intervals.
- **Few clusters.** The Corr2Cause test set has 21 premises, CLadder 47 stories (24 in a report half); cluster-bootstrap intervals with so few clusters under-cover.
- **Numerics.** Activations are cached in float16 from float16 models; for the two larger local models no float32 reference fits in memory. Probe fits at C = 1 are stopped at 300 iterations (bias against the probe of under one point on the one fold checked).
- **Two machines.** Gemma's probes on the original item sets use activations cached on the laptop; its steering, patching, SAE results and its debiased-pool activations were computed on a Kaggle GPU. Agreement of the answer logits between the two is reported in `results/cache_agreement_gemma2b.json`.
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

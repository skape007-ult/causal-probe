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

<!-- T1-BEHAVIOUR-TEXT -->

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

<!-- T1-LENS-TEXT -->

### 3.3 Probes and controls

<!-- TABLE:probes_summary -->

![probes c2c](../figures/fig3_probes_c2c_probe.png)
![probes cladder](../figures/fig3_probes_cladder.png)
![selectivity](../figures/fig4_selectivity_positions_c2c_probe.png)

Probe accuracy by token position:

<!-- TABLE:probe_positions -->

<!-- T1-PROBES-TEXT -->

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

<!-- T1-CONFOUND-TEXT -->

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

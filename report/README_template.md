# causal-probe

Do small open-weight language models internally represent the correct answer to formal causal-reasoning questions that they answer wrongly? A probing, steering and patching study on Corr2Cause and CLadder, run on an 8 GB laptop plus free Kaggle and Colab GPU sessions.

**Status:** work in progress until Sunday 15 November 2026, 14:00 (Toronto). Sections marked *pending* are filled in as runs finish.

## The question

1. Is the correct answer linearly decodable from a model's activations even when the model answers wrongly?
2. Is that representation causally used, and where is the answer determined?
3. What do sparse-autoencoder features show? *(Tier 3.)*

## The short answer

- **Behaviour.** Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-2-2B and Qwen2.5-3B answer both benchmarks at or near chance. Part of that is answer bias, but the bias-free ranking score is also near 0.5 (table below), and exactly at chance on the debiased item set.
- **Probes.** Linear probes read the gold label at {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} to {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} accuracy on a balanced Corr2Cause pool. This is **not** evidence of a hidden correct answer: the hypothesis string alone, with no premise and no model, predicts the label at {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|est}}, and a parsed-feature text classifier at {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/parsed_features/accuracy|est}}. Against the strongest text-only baseline the probe loses in all six model × benchmark cells.
- **A fair test (post hoc, audited).** On a debiased Corr2Cause pool, where neither the hypothesis alone nor the premise alone carries label information in any fold, only an interaction between the two can predict the label. The final-token probes of Qwen2.5-1.5B, Gemma-2-2B and Qwen2.5-3B score {{probe_content_qwen1.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}, {{probe_content_gemma2b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}} and {{probe_content_qwen3b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}: above every interaction control that binds nothing to the queried variables (strongest: {{probe_content_qwen1.5b_c2c_debiased.json|baselines/A_no_binding_interaction/template_x_premise_bag_of_words/accuracy|est}}), level with a per-template lookup of the premise statements that name the queried pair ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|est}}), and below a hand-written parse ({{probe_content_qwen1.5b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}). They add two to three points to that parse and at most one, with an interval reaching zero, to the strongest text-only model. Qwen2.5-0.5B's probe ({{probe_content_qwen0.5b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}) is not above the no-binding controls. **The audited criterion for "represents the answer but doesn't say it" is not met for any model.**
- **The answers.** Pooled, all four models answer the debiased pool at chance. Their answers do depend on whether the two queried variables are adjacent in the premise, with a sign that differs by hypothesis type and by model.
- **Steering (rewritten after a second audit, tightened after a third).** Adding the gold-label probe's weight direction, at the final token or at the end of the hypothesis, shifts the answer by an amount within the range of 20 probe directions fitted to shuffled labels (Qwen models on the laptop; matched controls for Gemma-2-2B and Qwen2.5-3B come with the second round of GPU runs). This neither shows nor excludes a causal role, and a later experiment showed why it could not: a probe's weight vector is not where the information lives.
- **Patching.** A change of one variable in the prompt reaches the answer position at about 60% of the network's depth (three models). The answer logit moves as much when the edit leaves the correct answer unchanged as when it changes it (Qwen2.5-0.5B so far), so this locates where a prompt difference arrives, not where a causal judgement is made.
- **Is the adjacency information used? (second round; Qwen2.5-0.5B so far.)** Erasing the linear encoding of adjacency at the last token of the hypothesis removes {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|erasure/contrast_reduction_of_dependence/value|num}} of the {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|erasure/contrast_reduction_of_dependence/benchmark_full_removal_on_evaluation_items|num}} logits by which the answers depend on adjacency, where 20 equally large control erasures remove between {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|erasure/contrast_reduction_of_dependence/vs_controls_norm_matched/control_min|signed}} and {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|erasure/contrast_reduction_of_dependence/vs_controls_norm_matched/control_max|signed}}. So a small part of what a probe reads there is passed on to the answer. Near the output the whole dependence is present in linear form, and a transfer check shows that this is each hypothesis type's own answer shift, not a shared adjacency feature. The first version of this experiment could not answer the question and was replaced after an independent code review, before it ran at scale.
- **SAE features (Gemma-2-2B, CLadder).** The public dictionaries fit these activations poorly. The features most associated with a type of causal graph are not specific to it; removing them shifts the answer somewhat more than removing matched random features, equally for all graph types.
- **Still running as this is written:** Qwen2.5-7B behaviour; matched steering controls, label-preserving patching and the adjacency test for Qwen2.5-1.5B, Qwen2.5-3B and Gemma-2-2B. The report marks what has and has not arrived.

## What was pre-registered, what passed, what was retracted

- **Pre-registered** (written in the decision log before any probe was run, D11): on report-half items the model answers wrongly, gold-probe accuracy must exceed 0.5, a TF-IDF baseline and a template prior, by 95% confidence-interval lower bounds. Layer selection on one half of the premises, reporting on the other.
- **Passed as coded.** In the current result files that criterion is met for Qwen2.5-1.5B and Gemma-2-2B on CLadder, in variants also on Corr2Cause, and for all four models on the debiased pool. One cell (Qwen2.5-1.5B on Corr2Cause, as first run) was logged as a positive result on Thursday night.
- **Retracted** after an independent audit of code and results on Friday: the pre-registered baselines were too weak. The audit showed that the hypothesis string alone predicts {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|pct}} of labels in the pool, and that "accuracy above 0.5 on wrongly answered items" is not a chance test when those items are mostly one class.
- **Added after seeing results** (and therefore not pre-registered): the hypothesis-string, question-only and parsed-feature baselines (D18); the audited criterion, probe versus strongest text-only baseline (D19); the debiased pool (D21); the ladder of baselines on it (D25); matched steering controls (D24); the adjacency test, its second version and its transfer arm (D27, `notes/log.md` 13 November). Each can only make the claim harder to pass. The pre-registered analysis is still reported in full next to them.

## Headline tables

Behaviour, separating bias from chance:

<!-- TABLE:answer_bias -->

Probe versus strongest text-only baseline (report half; "Corr2Cause debiased pool" rows appear once those runs finish):

<!-- TABLE:audited_verdict -->

What the text alone gives away on Corr2Cause (official splits):

<!-- TABLE:benchmark_c2c_official -->

## Key figures

![behaviour](figures/fig1_behaviour.png)
![probes](figures/fig3_probes_c2c_probe.png)
![probe vs text](figures/fig5_confound_c2c_probe.png)
![adjacency, end of hypothesis](figures/fig12_adjacency_end_of_hypothesis.png)

How to read each figure: `EXPLAINER.md`, section "How to read each figure".

## What this shows / what it does not

*Shows:*
- For these four models, on these two benchmarks, a linear probe's above-chance accuracy for the gold label is fully accounted for by what text-only classifiers extract from the prompt.
- On the released Corr2Cause data, a classifier that sees only the hypothesis ranks valid above invalid test items with AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}; this follows from the benchmark's variable-naming construction (details and prior work: `report/benchmark_text_baselines.md`).

*Does not show:*
- That these models contain no causal-reasoning structure at all: a linear probe at three token positions is a limited instrument, and absence of evidence here is not evidence of absence.
- Anything about models that can do the task. The largest model analysed here has 3 billion parameters and answers at chance.
- That any of these models "uses an adjacency feature": the one positive causal result is a modest erasure effect at one layer of the smallest model.
- That the benchmarks are unsuitable for their purpose. Zero-shot evaluation, which the benchmarks were designed for, cannot exploit a label prior it has never seen. The finding concerns classifiers trained on in-distribution items, which includes probes.

## Documents

| File | What it is |
|---|---|
| `EXPLAINER.md` | Start here. Concept primer, decision log, how to read each figure, questions a professor would ask |
| `report/report.md` | Method, tables and figures per tier |
| `report/benchmark_text_baselines.md` | Stand-alone note on text-only baselines for the two benchmarks |
| `notes/log.md` | Dated lab notebook: expectation before each experiment, comparison after |
| `notes/upstream_issues.md` | Bugs and documentation gaps found on the way, with minimal reproductions (not filed) |
| `notes/prior_work.md` | Prior-work check for the benchmark note: what was opened, what was found, the queries |
| `notes/audit_*_raw.md` | The raw reports of the independent audits (Tier 1, Tier 2, debiased pool, round-2 code) |
| `study/` | Reading order, 40 graded questions with answers, five predict-then-check exercises |
| `KAGGLE.md` | How to run the GPU part (first run and round 2) |
| `YOUR_TURN.md` | Two small experiments left for the reader |

## Limitations

*(Full list in the report; filled in at packaging.)*

## Reproduction

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt -e .
```

```bash
.venv/bin/python scripts/download_data.py
```

```bash
.venv/bin/python scripts/14_benchmark_baselines.py
```

The last command reproduces every number in `report/benchmark_text_baselines.md` and needs no model. The model-dependent parts need the activation cache (`scripts/build_cache.sh` for the two Qwen models; `KAGGLE.md` for Gemma and the larger model), after which `scripts/run_all.sh` regenerates every number and figure. *(run_all.sh: pending.)*

## Citations

- Jin et al. (2024). Can Large Language Models Infer Causation from Correlation? ICLR 2024. arXiv:2306.05836.
- Jin et al. (2023). CLadder: Assessing Causal Reasoning in Language Models. NeurIPS 2023. arXiv:2312.04350.
- Clarke et al. (2026). Evaluating Causal Reasoning in LLMs: A Case Study in the Challenges of Procedural Generation. arXiv:2610.38100.
- Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019.
- Poliak et al. (2018). Hypothesis Only Baselines in Natural Language Inference. *SEM 2018.
- Gururangan et al. (2018). Annotation Artifacts in Natural Language Inference Data. NAACL-HLT 2018.
- Lieberum et al. (2024). Gemma Scope. arXiv:2408.05147.

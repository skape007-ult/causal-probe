# causal-probe

Do small open-weight language models internally represent the correct answer to formal causal-reasoning questions that they answer wrongly? A probing, steering and patching study on Corr2Cause and CLadder, run on an 8 GB laptop plus one free Kaggle GPU session.

**Status:** work in progress until Sunday 15 November 2026, 14:00 (Toronto). Sections marked *pending* are filled in as runs finish.

## The question

1. Is the correct answer linearly decodable from a model's activations even when the model answers wrongly?
2. Is that representation causally used, and where is the answer determined?
3. What do sparse-autoencoder features show? *(Tier 3; pending Kaggle run.)*

## The short answer

- **Behaviour.** Qwen2.5-0.5B, Qwen2.5-1.5B and Gemma-2-2B answer both benchmarks at chance. Part of that is answer bias, but the bias-free ranking score is also near 0.5 (table below).
- **Probes.** Linear probes read the gold label at {{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} to {{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} accuracy on a balanced Corr2Cause pool. This is **not** evidence of a hidden correct answer: the hypothesis string alone, with no premise and no model, predicts the label at {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|est}}, and a parsed-feature text classifier at {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/parsed_features/accuracy|est}}. Against the strongest text-only baseline the probe loses in all six model × benchmark cells.
- **A fair test.** On a debiased Corr2Cause pool where neither the hypothesis alone nor the premise alone carries any label information, the gold probe scores {{confound_qwen0.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} for Qwen2.5-0.5B (a randomly initialised copy: {{confound_qwen0.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|est}}) and {{confound_qwen1.5b_c2c_debiased.json|all_items/gold_probe/accuracy|ci}} for Qwen2.5-1.5B (random copy: {{confound_qwen1.5b_c2c_debiased.json|all_items/random_init_probe/accuracy|est}}), while both models answer at chance. A parsed-feature text classifier scores {{confound_qwen1.5b_c2c_debiased.json|all_items/structural/accuracy|ci}}, so the audited criterion is still not met; but the 1.5B model does carry information about the answer that it does not use. Probes for parsed properties show what it is: whether the two queried variables are adjacent in the premise's skeleton is readable from its activations at {{probe_content_qwen1.5b_c2c_debiased.json|properties/adjacent/probe/final/accuracy|est}}. Gemma-2-2B and Qwen2.5-3B: *pending the Kaggle run*.
- **Steering (two Qwen models; rewritten after a second audit).** Adding the gold-label probe's weight direction at the probed layer changes the answer no more than adding a probe direction fitted to shuffled labels: its slope is at the {{steering_controls_qwen0.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} and {{steering_controls_qwen1.5b_c2c_probe.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} percentile of 20 such controls on the original pool, and at the {{steering_controls_qwen0.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} and {{steering_controls_qwen1.5b_c2c_debiased.json|real/gold_probe/versus/shuffled_probe/percentile_of_abs_slope_per_s|pct}} percentile on the debiased pool. This shows that this direction is not read at this site; it does not show that the decoded information is unused. The direct evidence for "not expressed" is that the answers are at chance.
- **Patching.** A change of one variable in the prompt reaches the answer position at about 60% of the network's depth. Because the models do not prefer the valid prompt of a pair, this locates where a prompt difference arrives, not where a causal judgement is made.
- **Gemma-2-2B interventions, a larger model, and SAE features:** *pending the Kaggle run.*

## What was pre-registered, what passed, what was retracted

- **Pre-registered** (written in the decision log before any probe was run, D11): on report-half items the model answers wrongly, gold-probe accuracy must exceed 0.5, a TF-IDF baseline and a template prior, by 95% confidence-interval lower bounds. Layer selection on one half of the premises, reporting on the other.
- **Passed as coded.** That criterion is met for Qwen2.5-1.5B on CLadder and for Gemma-2-2B on CLadder, and in variants also on Corr2Cause. The first of these was logged as a positive result on Thursday night.
- **Retracted** after an independent audit of code and results on Friday: the pre-registered baselines were too weak. The audit showed that the hypothesis string alone predicts {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|pct}} of labels in the pool, and that "accuracy above 0.5 on wrongly answered items" is not a chance test when those items are mostly one class.
- **Added after seeing results** (and therefore not pre-registered): the hypothesis-string, question-only and parsed-feature baselines (D18); the audited criterion, probe versus strongest text-only baseline (D19); the debiased pool (D21). Each can only make the claim harder to pass. The pre-registered analysis is still reported in full next to them.

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

How to read each figure: `EXPLAINER.md`, section "How to read each figure".

## What this shows / what it does not

*Shows:*
- For these three models, on these two benchmarks, a linear probe's above-chance accuracy for the gold label is fully accounted for by what text-only classifiers extract from the prompt.
- On the released Corr2Cause data, a classifier that sees only the hypothesis ranks valid above invalid test items with AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}; this follows from the benchmark's variable-naming construction (details and prior work: `report/benchmark_text_baselines.md`).

*Does not show:*
- That these models contain no causal-reasoning structure at all: a linear probe at three token positions is a limited instrument, and absence of evidence here is not evidence of absence.
- Anything about larger models. *(One larger model is being run; pending.)*
- That the benchmarks are unsuitable for their purpose. Zero-shot evaluation, which the benchmarks were designed for, cannot exploit a label prior it has never seen. The finding concerns classifiers trained on in-distribution items, which includes probes.

## Documents

| File | What it is |
|---|---|
| `EXPLAINER.md` | Start here. Concept primer, decision log, how to read each figure, questions a professor would ask |
| `report/report.md` | Method, tables and figures per tier |
| `report/benchmark_text_baselines.md` | Stand-alone note on text-only baselines for the two benchmarks |
| `notes/log.md` | Dated lab notebook: expectation before each experiment, comparison after |
| `notes/upstream_issues.md` | Bugs and documentation gaps found on the way, with minimal reproductions (not filed) |
| `KAGGLE.md` | How to run the GPU part |
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

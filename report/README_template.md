# causal-probe

Do small open-weight language models internally represent the correct answer to formal causal-reasoning questions that they answer wrongly? A probing, steering and patching study on Corr2Cause and CLadder, run on an 8 GB laptop plus free Kaggle and Colab GPU sessions.

**Status:** work in progress until Sunday 15 November 2026, 14:00 (Toronto). The results page below was replaced on Saturday 14 November at 11:29, after fifteen reviews; every sentence of it was checked against the result files by a reviewer before it was entered (`notes/audit_results_wording_raw.md`). Three GPU runs launched on 14 November are not in it yet: a held-out test of the Qwen2.5-3B steering case, patching on about 240 pairs, and Gemma-2-9B's answers and probes.

<!-- INCLUDE:RESULTS -->

The same page as a file: `RESULTS.md`. How the story unfolded, in plain language: `EXPLAINER.md`, "Start here".

## Further results (not headline claims)

- **Behaviour.** The four smaller models answer both benchmarks at or near chance; part of that is answer bias, and the bias-free ranking score is near 0.5 as well (table below). Qwen2.5-7B (4-bit) is at chance on the debiased pool when all items are ranked at once and follows validity weakly inside a hypothesis string (results page, claim 2).
- **Patching.** A change of one variable in the prompt reaches the answer position at 60 to 70% of the network's depth (four models). When the edit leaves the correct answer unchanged the answer logit moves by an amount of the same order as when it changes it (four models); in Qwen2.5-0.5B it moves as much, and its sign follows where the queried variable stands in the premise, not validity (a pattern the two larger models do not reproduce): this locates where a prompt difference arrives, not where a judgement is made.
- **In the report's appendix:** logit lens, cross-model comparison, sparse-autoencoder features, attribution patching of components.

## What was pre-registered, what passed, what was retracted

- **Pre-registered** (written in the decision log before any probe was run, D11): on report-half items the model answers wrongly, gold-probe accuracy must exceed 0.5, a TF-IDF baseline and a template prior, by 95% confidence-interval lower bounds. Layer selection on one half of the premises, reporting on the other.
- **Passed as coded.** In the current result files that criterion is met for Qwen2.5-1.5B and Gemma-2-2B on CLadder, in variants also on Corr2Cause, and for all four models on the debiased pool. One cell (Qwen2.5-1.5B on Corr2Cause, as first run) was logged as a positive result on Thursday night.
- **Retracted** after an independent audit of code and results on Friday: the pre-registered baselines were too weak. The audit showed that the hypothesis string alone predicts {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|pct}} of labels in the pool, and that "accuracy above 0.5 on wrongly answered items" is not a chance test when those items are mostly one class.
- **Added after seeing results** (and therefore not pre-registered): the hypothesis-string, question-only and parsed-feature baselines (D18); the audited criterion, probe versus strongest text-only baseline (D19); the debiased pool (D21); the ladder of baselines on it (D25); matched steering controls (D24); the adjacency test, its second version and its transfer arm (D27, `notes/log.md` 13 November). Each of D18 to D25 can only make the claim harder to pass; the adjacency test is exploratory, and one of its changes made a positive reading easier (report 4.4, "Status"). The pre-registered analysis is still reported in full next to them.

## Headline tables

Behaviour, separating bias from chance:

<!-- TABLE:answer_bias -->

Probe versus strongest text-only baseline (report half):

<!-- TABLE:audited_verdict -->

What the text alone gives away on Corr2Cause (official splits):

<!-- TABLE:benchmark_c2c_official -->

## Other figures

![behaviour](figures/fig1_behaviour.png)
![probes](figures/fig3_probes_c2c_probe.png)

How to read each figure: `EXPLAINER.md`, section "How to read each figure".

## What this shows / what it does not

*Shows:*
- For these models, on these two benchmarks, a linear probe's above-chance accuracy for the gold label is almost entirely accounted for by what text-only classifiers extract from the prompt. On the debiased pool no probe's gain in accuracy over a text-only stack with one tuned setting is demonstrable for any of five models; small gains in AUROC (a few thousandths) are present from Qwen2.5-1.5B upward (report Section 3.6).
- On the released Corr2Cause data, a classifier that sees only the hypothesis ranks valid above invalid test items with AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}; this follows from the benchmark's variable-naming construction (details and prior work: `report/benchmark_text_baselines.md`).

*Does not show:*
- That these models contain no causal-reasoning structure at all: a linear probe at three token positions is a limited instrument, and absence of evidence here is not evidence of absence.
- Anything about models that can do the task. The largest model analysed here has 3 billion parameters and answers at or barely above chance.
- That any of these models "uses an adjacency feature": the one clear causal effect is at the answer position, where it shows only that the answer is there.
- That the benchmarks are unsuitable for their purpose. Zero-shot evaluation, which the benchmarks were designed for, cannot exploit a label prior it has never seen. The finding concerns classifiers trained on in-distribution items, which includes probes.

## Documents

| File | What it is |
|---|---|
| `RESULTS.md` | The one-page results shown at the top of this README |
| `EXPLAINER.md` | Start here. Concept primer, decision log, how to read each figure, questions a professor would ask |
| `report/report.md` | Method, tables and figures per tier |
| `report/benchmark_text_baselines.md` | Stand-alone note on text-only baselines for the two benchmarks |
| `notes/log.md` | Dated lab notebook: expectation before each experiment, comparison after |
| `notes/upstream_issues.md` | Bugs and documentation gaps found on the way, with minimal reproductions (not filed) |
| `notes/prior_work.md` | Prior-work check for the benchmark note: what was opened, what was found, the queries |
| `notes/audit_*_raw.md` | The raw reports of the independent audits (fifteen), and a claims pass |
| `study/` | Reading order, 52 graded questions with answers, six predict-then-check exercises |
| `KAGGLE.md` | How to run the GPU part (first run and round 2) |
| `YOUR_TURN.md` | Two small experiments left for the reader |

## Limitations

The short list is in the results page at the top; the full list is Section 7 of `report/report.md`.

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

The last command reproduces every number in `report/benchmark_text_baselines.md` and needs no model. The model-dependent parts need the activation cache (`scripts/build_cache.sh` for the two small Qwen models; `KAGGLE.md` for Gemma, Qwen2.5-3B and the second-round interventions of Qwen2.5-1.5B), after which `scripts/run_all.sh` regenerates every number and figure. *(`run_all.sh` exists but has not been run end to end since the second round; that check is part of packaging.)*

## Citations

- Jin et al. (2024). Can Large Language Models Infer Causation from Correlation? ICLR 2024. arXiv:2306.05836.
- Jin et al. (2023). CLadder: Assessing Causal Reasoning in Language Models. NeurIPS 2023. arXiv:2312.04350.
- Clarke et al. (2026). Evaluating Causal Reasoning in LLMs: A Case Study in the Challenges of Procedural Generation. arXiv:2610.38100.
- Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019.
- Poliak et al. (2018). Hypothesis Only Baselines in Natural Language Inference. *SEM 2018.
- Gururangan et al. (2018). Annotation Artifacts in Natural Language Inference Data. NAACL-HLT 2018.
- Lieberum et al. (2024). Gemma Scope. arXiv:2408.05147.

# Work log

All times America/Toronto. Format per experiment: **Expectation** (written before), **Surprise would be**, **Result** (after), **Comparison**.

## Thu 2026-11-12

### 17:11 Start. Environment
- Apple M3, 8 cores, **8 GB RAM**, 252 GB free disk, MPS available. Python 3.12 venv via uv. torch 2.15.0, transformers 5.22.2, transformer-lens 4.4.0, sae-lens 6.55.4.
- HF login present (gemma-2-2b-it accessible). GitHub remote `origin` already configured.
- 8 GB RAM is the binding constraint. Qwen2.5-3B in float16 is ~6.2 GB of weights alone: excluded (decision log D2).

### 17:15 Datasets read (cards, GitHub READMEs, paper abstracts, raw CSVs)
- **Corr2Cause** (Jin et al. 2023, arXiv 2306.05836; ICLR 2024). HF `causal-nlp/corr2cause`; card is literally "TODO"; code repo MIT. Task: premise = all (conditional) independence / correlation statements among 2–6 variables; hypothesis = one causal claim; label 1 if the hypothesis holds in every graph of the Markov equivalence class.
  - train 205,734 (18.5% positive), dev 1,076 (12.9%), test 1,162 (15.5% = 180 positive). **Imbalanced.**
  - Six hypothesis templates: parent, child, non-parent ancestor, non-child descendant, has_collider, has_confounder.
  - Train has only 2,338 distinct premises (≈88 hypotheses each); **test has only 21 distinct premises**, dev 22. Items sharing a premise are not independent → bootstrap must resample premises, probes must split by premise.
  - No premise overlap train↔test. Dev↔train share 2 premises (42 dev items) → logged in upstream_issues.md; we use dev only as the source of few-shot demonstrations.
  - The `child` template has 0 positives in train (34,291 items), dev and test, whereas `parent` has 35% positives. Logged as an upstream question.
- **CLadder** (Jin et al. 2023, arXiv 2312.04350; NeurIPS 2023). HF `causal-nlp/CLadder`, split `full_v1.5_default`: 10,112 yes/no questions, exactly 50/50. Card empty; code repo MIT. Fields: prompt, label, rung (1/2/3), query_type (10), graph_id (10), story_id (47), question_property.
  - Label is balanced within every query type. Rung is a function of query type.

### 17:20 MPS verification, qwen0.5b
- **Expectation:** MPS float32 matches CPU float32 to ~1e-3 in logits; float16 within ~0.1; float16 several times faster than CPU.
- **Surprise would be:** NaNs in half precision, or top-1 disagreement.
- **Result** (`results/env_check_qwen0.5b.json`): MPS fp32 max |Δlogit| 1.2e-4; fp16 0.050 (answer logit-diff gap 0.023); bf16 0.40 (gap 0.089); top-1 agreement 100% in all; no NaNs. Batched (left-padded) vs single-item passes agree to the same tolerance. Speed at ~209 tokens/prompt: CPU fp32 2.7 items/s, MPS fp16 8.5 items/s, MPS bf16 8.2; MPS fp32 0.8 items/s with 8.2 GB allocated (memory thrash).
- TransformerLens 4.4 `TransformerBridge` vs plain HF (CPU fp32, 5 prompts): max |Δlogit| 7e-5.
- **Comparison:** as expected, except MPS float32 being *slower* than CPU, which I did not predict (memory pressure on an 8 GB machine). → float16 on MPS (D3).

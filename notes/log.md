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

### 17:30–19:00 MPS verification for the larger models; the Gemma memory saga
- **qwen1.5b** (`results/env_check_qwen1.5b.json`): a float32 CPU reference (6.2 GB) does not fit in 8 GB (killed after thrashing), so the reference is CPU **float16**: this checks the device, not the precision. MPS fp16 vs CPU fp16: max |Δlogit| 0.0625, answer-diff gap 0.031, top-1 agree 100%, no NaN. 2.2 items/s. bf16 again ~4–6× less accurate. Precision adequacy of fp16 rests on the 0.5B fp32 comparison.
- **gemma2b** (`results/env_check_gemma2b.json`, CPU reference in bf16 because CPU fp16 was even slower): MPS fp16 vs CPU bf16 gap 0.27 in answer logit diff, but the bf16 reference disagrees with *itself* by 0.25 between batched and single runs, so the gap is reference noise; MPS fp16 batched-vs-single gap is 0.016; top-1 agree 100%; no NaN. **Limitation: no float32 reference for the two larger models.**
- **Expectation (before):** Gemma (5.2 GB fp16) would fit and run at ~1 item/s. **Surprise:** it ran at 0.07–0.2 items/s.
- Diagnosis, in order: (1) `model(...).logits` unembeds every position (B×T×256k) → fixed by unembedding the last position only; still slow. (2) Body alone is fast (0.75 s for 440 tokens) with random inputs; slow only in the real pipeline. (3) Cause is memory pressure: weights 5.3 GB on an 8 GB unified-memory machine; anything else touching memory evicts weights. (4) Fix: keep Gemma's 256k×2304 embedding/unembedding matrix (1.18 GB) as a float16 memory-map on disk, look up input rows on CPU, compute only the 12 answer-token logits per item, batch size 1, release the MPS allocator cache every batch → **1.0–1.1 items/s at 4.4 GB**. Answer logit diffs match the full model on the 5 check prompts to 0.012.
- Cost of the fix: for Gemma the cache has no full-vocab top-1 token, and generated-text scoring is first-token only on 100 items per set (each full-vocab decode step costs ~6 s).
- HF quirk found while testing: `hidden_states[-1]` is post-final-norm in transformers 5.22 (U6). Our cache uses block hooks and is unaffected; a test pins both facts.

### 19:00 Cache build started (qwen0.5b first)

## Tier 1 expectations (written Thu ~19:00, before any behavioural or probe result)
**Behaviour.**
- Expect: on Corr2Cause test all three models near or below the majority baseline in accuracy (84.5%) and macro-F1 not far above the random baseline (~0.45–0.55); likely strong Yes- or No-bias. On CLadder, accuracy 50–60%. Few-shot helps a little (≤5 points). Logit scoring and generated text agree on >90% of items. Variable renaming flips 5–20% of predictions with little change in accuracy (accuracy is near chance anyway).
- Surprise: any model > 0.70 macro-F1 on Corr2Cause or > 70% on CLadder zero-shot; few-shot hurting substantially; logit/generation disagreement > 20%.

**Logit lens.**
- Expect: answer logit difference ≈ 0 (uninformative) until the last third of layers, then moving to the model's final answer. Gold-signed curves for correct and incorrect items separate only late; incorrect items do **not** show a transient lean towards the right answer.
- Surprise: incorrect items leaning towards the gold answer at middle layers and being overwritten later.

**Probes.**
- Expect: Corr2Cause balanced pool: gold probe 60–75% at the best layer; TF-IDF also well above chance (55–70%) because hypothesis templates carry label priors; random-init probe near TF-IDF; shuffled ≈ 50%; control-task accuracy moderately high (the hypothesis identity is easy to read), so selectivity is small. CLadder: gold probe 55–65%; TF-IDF ≈ 50–55%.
- Expect model-output probe ≫ gold probe (90%+), since the model's answer is by construction in the final-position residual.
- **Headline prediction:** "represents but doesn't say" will *fail* the pre-registered criterion for most model × dataset cells: where the gold probe beats chance on model-wrong items, the text baselines will too. My credence that the criterion passes in at least one of six cells: ~35%.
- Surprise: gold probe > 85% anywhere; or the criterion passing in most cells; or the random-init model matching the trained one.
- Cross-model: decodability rises modestly with size; best layer in the middle-to-late range for all.

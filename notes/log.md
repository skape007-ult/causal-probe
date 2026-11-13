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

### 19:10 First look at qwen0.5b; one analysis added after seeing results (logged as required)
- qwen0.5b answers "No" on 100% of Corr2Cause test, 99.6% of the pool and 96% of CLadder. So (i) the raw model-output probe target is degenerate (skipped when the minority answer is < 5%), and (ii) "items the model gets wrong" = "items whose gold label is Yes", so accuracy-on-wrong-items is just recall on positives.
- **Change made after seeing this (not a hyperparameter):** added a second, clearly labelled "calibrated" version of the model's answer = answer logit difference above/below its own median (constant bias removed), with its own probe target and its own strata in `05_confound.py`. The pre-registered verdict (D11) is unchanged and still computed on the raw answer; the calibrated verdict is reported next to it. Nothing about the probes themselves (C, layers, folds) was changed.
- Smoke-run on qwen0.5b/c2c_probe (random-init cache not yet built): gold probe 0.760 at best layer vs TF-IDF 0.750, type prior 0.692, control task 0.644, shuffled 0.498. In line with the logged expectation (text baseline matches the probe).

### 19:30 qwen0.5b Tier 1 complete (cache 24.6 min trained + 9.5 min random-init)
Numbers from `results/*_qwen0.5b*.json` (full Tier 1 comparison against expectations is written once all three models are in).
- Behaviour: Corr2Cause test acc 0.845 = majority baseline exactly (predicts "No" for 100%); macro-F1 0.458; AUROC of its logit diff 0.589. CLadder acc 0.500 (96% "No").
- Probes, Corr2Cause pool: gold 0.760 @ L15, TF-IDF 0.750, type prior 0.692, control task 0.644, random-init 0.588, shuffled 0.498. CLadder: gold 0.548 @ L22, TF-IDF 0.515, control task 0.756 (selectivity negative), random-init 0.505.
- Criterion: not met on either dataset (probe ≈ TF-IDF on model-wrong items).

### 19:40 Tier 2 code written and smoke-tested; expectations
**Honesty note:** the two smoke tests below (40 items / 8 pairs, qwen0.5b) ran before I wrote these expectations, so for qwen0.5b they are not blind. They are blind for qwen1.5b and gemma2b.
- Smoke, steering (n=40, L15, Corr2Cause): gold-probe direction flips 0% of answers at every scale up to 8 (random directions: up to 33%); diff-of-means flips 95–100% from scale 2; model-output direction flips 97–100% from scale 1.
- Smoke, patching (8 pairs): the model's logit diff barely distinguishes clean from corrupt (mean D ≈ 0.13 logits) but pooled recovery at the final position jumps from 0.11 (layer 14) to 0.80 (layer 15); attribution vs real patching r ≈ 0.99 on 10 components.

**Expectations for the full runs.**
- Steering. What I would have said before the smoke test: gold-probe direction has a moderate causal effect, weaker than the model-output direction, stronger than random. After the smoke test I expect instead: gold-probe direction ≈ no effect (≤ random) in all three models; diff-of-means and model-output directions flip most answers at scale ≥ 2; ablating the gold-probe direction changes the logit diff by less than ablating the model-output direction. *Surprise:* the gold-probe direction flipping answers in the larger models; or accuracy improving when steering all items in the same direction (it cannot, by construction, if the direction is just "say Yes").
- Why the probe direction may be inert: logistic regression weights are discriminative, w ∝ Σ⁻¹(μ₁−μ₀); they point along low-variance directions that separate classes, which need not be directions the downstream layers read. Diff-of-means has no Σ⁻¹ and usually steers better (known from the literature on steering/"refusal direction"-style work).
- Patching. Expect: (i) minimal-pair sensitivity D small but positive on average for the larger models (they are near chance), (ii) information moves from the differing tokens to the final position in the middle third of layers, at similar *relative* depth across models, (iii) random-donor control recovery ≈ 0 at the differing tokens and at the final position early, but possibly non-zero late at the final position (any clean run carries a generic "answer" state), (iv) a handful of heads + MLPs carry most of the attribution, attribution-vs-real correlation > 0.8. *Surprise:* recovery spread evenly over layers; negative D on average (model prefers the wrong member of the pair); attribution failing validation.

### 19:55 Lesson: one GPU job at a time
Running 0.5B patching (which also held ~1 GB of recorded clean residuals in RAM) next to the 1.5B cache build stalled both for 13 minutes (memory free 6%). Killed the patching run; `08_patching.py` now recomputes clean runs on demand instead of storing them. Rule from now on: GPU jobs strictly sequential.

### 19:50 qwen0.5b steering (full, n=300 per dataset) — `results/steering_qwen0.5b_*.json`
- **Result:** gold-probe direction: flip rate ≤ 0.03 at every scale on Corr2Cause and ≤ 0.04 on CLadder, i.e. *below* random directions (0.33 / 0.64 at α=8). Ablating it changes the logit diff by 0.005 (random: 0.007). Model-output direction flips 100% (C2C) / 93% (CLadder) at α=1. Diff-of-means flips 96% at α=2 on C2C; on CLadder it pushes the *other* way (accuracy toward gold falls to 0.01), i.e. the gold-Yes mean differs from the gold-No mean along a direction the model reads as "No".
- Geometry: the gold-probe direction has projection std 0.026 s (random direction: ~0.03 s), cosine 0.05 with the model-output direction. It is a low-variance direction.
- **Comparison:** matches the (post-smoke-test) expectation. The CLadder diff-of-means sign reversal was not expected.

### 19:57 Probe regularisation: sensitivity check fired the D7 trigger → change of method (logged per the rules)
- `results/probe_sensitivity_qwen0.5b.json`: Corr2Cause gold-probe accuracy at layer 15 by C = 1e-4 / 1e-3 / 1e-2 / 1e-1 / 1: 0.604 / 0.687 / **0.763** / 0.787 / 0.798. My fixed C = 0.01 under-fits; and the TF-IDF baseline was run at C = 1. That handicaps the probe relative to the baseline and could flip the probe-vs-text conclusion (D7 named exactly this as the condition for changing the decision). CLadder is flat (0.52–0.54) at every C.
- **What changed (final form, 20:20):** probes are run at C ∈ {0.01, 0.1, 1} and TF-IDF at C ∈ {0.1, 1, 10}; `05_confound.py` chooses (layer, C) for each probe target, and C for TF-IDF, by accuracy on the *selection half* of premises/stories and reports on the *report half*. Same procedure for the random-init and model-output probes. (First attempt was full nested CV inside each training fold via `LogisticRegressionCV`; after 20+ minutes it had not finished the first position for the smallest model, so it was abandoned for this cheaper scheme, which keeps the same guarantee: no reported item influences any choice.)
- **Why this is not "tuning until it works":** the selection is automatic, uses only the selection half, is applied identically to the baseline and the controls, and the original fixed-C results are kept and reported side by side (`results/confound_*_fixedC.json`, flag `--fixed-c`; and the per-C curves in `results/probes_*_C*.json`). The direction of the change favours the "represents but doesn't say" hypothesis, so both versions will be shown wherever the verdict is discussed.
- **Expectation for the nested run:** Corr2Cause gold probe ≈ 0.79–0.80 for qwen0.5b; TF-IDF may also rise with C=10; control-task accuracy rises too (more capacity) so selectivity does not improve much. Verdict could become True on Corr2Cause if TF-IDF stays ~0.75–0.77. Surprise: probe ≥ 0.85, or TF-IDF overtaking the probe.

### 22:25 Scheduling change
Gemma's cache build made no progress for 10 minutes while the 1.5B probes ran on CPU (memory pressure again). Stopped Gemma (resumable), and wrote `scripts/queue_night.sh`: Qwen work first, then Gemma alone.

### 22:55 qwen0.5b patching (60 pairs per kind) — `results/patching_qwen0.5b.json`
- Behaviour on the pairs: mean D = LD(clean) − LD(corrupt) is 0.009 (hyp) and 0.023 (prem) logits; D > 0 in 50% / 58% of pairs. The model does not prefer the correct member of a minimal pair.
- Yet the *location* of whatever distinguishes the two prompts is sharp: pooled recovery at the final token is ≤ 0.08 up to layer 14 and 0.81–0.87 at layer 15, ~1.0 from layer 20. Attribution patching agrees with real patching on 20 components (r ≈ 1.00); the top components are head L14H5 and MLPs 23, 12, 9, 14.
- **Comparison with expectation:** (i) D small: yes, in fact indistinguishable from zero for this model; (ii) mid-depth transfer to the final token: yes (layer 15 of 24 = 0.62); (iv) attribution validated: yes. Not expected: such a clean single-layer step.
- Caveat written now so it is not forgotten: because D ≈ 0 on average and its sign is a coin flip, "recovery" here localises where *prompt-specific* information (which variable is asked about / how the premise is labelled) enters the final token, not where a *correct causal judgement* is computed. The model has none to localise.

### 23:00 qwen1.5b Tier 1 first numbers and a post-hoc robustness analysis (logged because it was added after seeing results)
- qwen1.5b behaviour: Corr2Cause test acc 0.773 (below the 0.845 majority baseline; says Yes on 10%), macro-F1 0.489; CLadder acc 0.497 (Yes on 64%).
- qwen1.5b Corr2Cause pool: **the pre-registered criterion is met** — on model-wrong items the gold probe scores 0.892 [0.866, 0.919] vs TF-IDF 0.847 and type prior 0.717 (selected C); 0.848 vs 0.790 under the original fixed C. This contradicts my headline prediction for this cell (I gave ~35% to *any* cell passing).
- Because this is the exciting direction, I asked what a sceptic would ask next: is TF-IDF a fair stand-in for "what the text gives away"? Bag-of-bigrams cannot bind a premise statement to the two variables named in the hypothesis. So I added a **stronger text-only baseline** (`probes.structural_features`, `probes.structural_oof`): for Corr2Cause, parsed premise statements bound to the hypothesis variables (are they correlated? adjacent in the skeleton? how many separating sets? shared neighbours inside/outside separating sets?) + template, fed to gradient boosting; for CLadder, the quoted probabilities, their differences, polarity words of the question, query and graph type. No causal inference rules are applied in either.
- **Expectation before running it:** structural baseline on the Corr2Cause pool ≈ 0.80–0.85, i.e. close to the 1.5B probe; the criterion probably does *not* survive it. CLadder numeric baseline 0.55–0.65.
- **Result:** Corr2Cause structural baseline 0.83 accuracy / 0.92 AUROC (grouped CV). CLadder numeric baseline **0.77 accuracy / 0.85 AUROC on held-out stories**, with `backadj` and `collider_bias` at 100% — far above my expectation, above every model (≈0.50) and above every probe (≤0.60). So on CLadder shallow text features carry much more label information than the models' activations linearly expose. (Dataset shortcut noted as U9.)
- This baseline is reported as a **secondary, post-hoc** check (`verdict.post_hoc_structural_baseline`), next to the unchanged pre-registered verdict. It can only weaken the exciting claim, never create it.

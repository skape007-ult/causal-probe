# EXPLAINER

Written for a smart first-year student who has not met these methods before. Read "Start here", then dip into the primer whenever a word is unfamiliar.

Contents: [Start here](#start-here) · [Pipeline](#pipeline) · [Concept primer](#concept-primer) · [Decision log](#decision-log) · [How to read each figure](#how-to-read-each-figure) · [Questions a professor would ask](#questions-a-professor-would-ask)

---

## Start here

<!-- START-HERE: updated as results arrive -->
**The question.** Small language models are bad at formal causal reasoning puzzles such as "given these correlations, must A cause B?". When such a model answers wrongly, is the right answer nevertheless sitting inside it, readable from its internal numbers? If it is, does the model actually use that information?

**What was done.** Three small open chat models (Qwen2.5 0.5B and 1.5B, Gemma-2 2B) were run on two benchmarks, Corr2Cause and CLadder, and their internal activations were saved. A simple classifier (a "probe") was then trained to read the correct answer out of those activations.

**The story, in the order it happened.**

1. *The models cannot do the task.* All three answer at chance. The smallest says "No" to almost everything; the other two say "Yes" more often and are wrong more often. Checked three ways (accuracy, a bias-free ranking score, and accuracy after re-thresholding), the answers carry at most a faint signal.
2. *First apparent result: the answer is in there.* Probes read the correct answer from the activations with 81 to 87% accuracy on Corr2Cause and 56 to 63% on CLadder, on questions about premises the probe had never seen. The usual sanity checks passed. By the criterion written down before the experiment, one model "represents the answer but doesn't say it". I logged that as a positive result.
3. *The audit.* A fresh reviewer, given the code and the numbers but not my conclusions, found the hole: on Corr2Cause, the *wording of the question alone*, without the premise and without any model, predicts the answer 86% of the time. The benchmark names variables so that causes come earlier in the alphabet, so "D directly causes C" is never true. A probe that merely recognises which question is being asked scores as if it knew the answer.
4. *Retraction.* Against text-only baselines that exploit this (and a similar one for CLadder, where the quoted numbers plus the wording of the question predict 80% and more), the probes lose in every model and both benchmarks. The positive result was withdrawn. The stronger baselines were added after seeing results, and the write-up says so.
5. *Making the benchmark finding solid.* The same baselines were run on the official splits with one reproducible script, the papers were read to separate what is already known from what is not, and a neutral note was written (`report/benchmark_text_baselines.md`).
6. *A fair test.* A new set of Corr2Cause items was built in which neither the question alone nor the premise alone says anything about the answer, in any training or test fold. Result so far (two models; two more are on a Kaggle GPU): the smallest model's probe scores 63%, barely above a probe on a randomly initialised copy of the same network (60%). The 1.5B model's probe scores 75%, clearly above its random copy (58%), while the model's actual answers stay at chance. So that model does hold something about the answer that it does not say. But a hand-written text parser that only performs the first step of the textbook method (look up what the premise says about the two variables in the question) scores 82%, and the model's activations turn out to contain exactly that lookup (readable at 93%). The fair description is: *partial computation that is not expressed*, not a hidden correct answer.
7. *Is the readable information used?* My first answer was "no: pushing the activations along the probe's direction does nothing, while pushing along the direction of the model's own answer flips it". A second independent audit took that apart. I had shown one sign of the push only, which hid a consistent effect in one model; I had compared with three random directions, far too few; and the pushes were a hundred times larger than anything that occurs naturally along the probe's direction. Redone with the right measure (how much the answer moves per unit pushed, both signs) and the right controls (directions built the same way from shuffled labels), the careful statement is: the probe's direction moves the answer no more than a label-free direction of the same kind, and a push of natural size does nothing. That is weaker than "unused", and the report says why. The plainest evidence that the information is not used is simply that the model's answers stay at chance.
8. *Where is the answer formed?* Swapping activations between two prompts that differ in one variable shows the difference travelling from the changed word to the final position at about 60% of the network's depth. The audit pointed out that this is where a *change in the prompt* arrives, not where a *causal judgement* is made, because the models do not prefer the correct prompt of a pair. The section is retitled accordingly.

**The one-sentence version (so far).** What first looked like a hidden correct answer was the probe reading the wording of the question; once that route is closed, the smallest model shows almost nothing and a larger one shows a first step of the reasoning that it carries internally and never uses. The general lesson: a probe's score means nothing until it is compared with what the text alone gives away, on items where the cheap routes are closed.
<!-- /START-HERE -->

---

## Pipeline

```mermaid
flowchart TD
    A[Raw benchmarks, pinned revisions<br/>download_data.py] --> B[Loaders, item sets<br/>original pool, debiased pool, CLadder sample<br/>data.py]
    A --> T[Text-only baselines on the benchmarks<br/>14_benchmark_baselines.py, no model]
    B --> C[Prompt + chat template<br/>prompts.py]
    C --> D[One forward pass per item<br/>laptop GPU: Qwen 0.5B, 1.5B, Gemma<br/>Kaggle GPU: Gemma debiased pool, Qwen 3B<br/>cache.py]
    D --> E[(Activation cache, float16<br/>residual stream, all layers<br/>+ answer logits)]
    E --> F[Behaviour and answer bias<br/>02, 15]
    E --> G[Logit lens<br/>03]
    E --> H[Probes + controls<br/>04]
    H --> I[Probe versus text baselines<br/>05: pre-registered and audited criteria]
    I --> I2[Post-hoc on the debiased pool<br/>05b: two baseline groups, incremental value,<br/>property probes, strata]
    I --> J[Steering, needs the model<br/>07 raw runs, 07b slopes, 07c matched controls]
    C --> K[Minimal pairs + activation patching<br/>08, needs the model]
    E --> L[SAE features and ablation<br/>11, Gemma only, Kaggle]
    F & G & I & I2 & J & K & L & T --> R[results/*.json] --> U[12 tables, 06 figures, 13 documents]
```

Independent audits sit between each tier's results and its write-up (`notes/audit_*_raw.md`).

The expensive step (D) runs once per model. Steps F–I never touch the model again.

---

## Concept primer

In dependency order. Each entry: what it is, a tiny example, where it lives in the code.

### Token, prompt, chat template
A model reads text as a sequence of **tokens** (word pieces, each an integer id). An instruct model was trained on conversations wrapped in special marker tokens; the **chat template** is the recipe that wraps your message in those markers. Skip it and the model sees text unlike anything it was trained to answer. *Example:* for Qwen, the user text `Is 2+2=4?` becomes `<|im_start|>user\nIs 2+2=4?<|im_end|>\n<|im_start|>assistant\n`; the model's answer is whatever it predicts next. *Code:* `prompts.render`, `prompts.encode_with_positions`.

### Logit and answer-token scoring
For the next token the model outputs one number per vocabulary entry, a **logit**; larger means more likely. We score a yes/no question by comparing the logits of "Yes"-like tokens against "No"-like tokens right after the prompt. Because "Yes", " yes" and "YES" are different tokens, we pool the variants (log-sum-exp) and take the **answer logit difference** = pooled Yes minus pooled No; positive means the model says Yes. *Example:* logits Yes=11.2, No=12.0 → difference −0.8 → the model answers No. *Code:* `models.answer_token_ids`, `models.answer_logit_diff`.

### Residual stream
A transformer keeps, for every token position, a running vector (896 numbers in Qwen-0.5B, 2304 in Gemma-2B). Each layer reads this vector and *adds* its contribution back. That running sum is the **residual stream**; it is the model's working memory at that position. *Example:* after layer 10 at the last prompt token, the residual stream is the embedding plus everything layers 0–10 added. *Code:* `cache.ResidCollector` (captures it), cache layout in `cache.py` docstring.

### Activation cache
Running a model is slow; analysing saved numbers is fast. We run each prompt once and save the residual stream at every layer for three positions: the **final** prompt token (where the answer is produced), the **end of the hypothesis/question**, and the **end of the premise**. *Example:* one Gemma item is 27 layers × 3 positions × 2304 numbers × 2 bytes ≈ 373 KB. *Code:* `cache.build_cache`, `cache.load_cache`.

### Logit lens
Normally only the last layer's residual stream is turned into logits. The **logit lens** applies that same final step (normalise, then multiply by the unembedding matrix) to an *earlier* layer, asking "if the model had to answer right now, what would it say?". It shows when the answer takes shape. *Example:* if the Yes−No difference is ≈0 up to layer 15 and then swings to −3, the decision is written into the stream after layer 15. *Code:* `lens.lens_logit_diff`, `scripts/03_logit_lens.py`.

### Linear probe
A **probe** is a small classifier trained to predict some property from an activation vector. A *linear* probe (logistic regression) can only draw a flat boundary, so if it succeeds the property is "linearly decodable": readable by one weighted sum. *Example:* take the layer-12 vectors of 2,400 puzzles, fit logistic regression to predict the correct answer, test on puzzles it has not seen. *Code:* `probes.fit_predict`, `probes.oof_scores`, `scripts/04_probes.py`.

### Held-out evaluation, cross-validation and grouped folds
A probe must be tested on data it was not trained on. In k-fold **cross-validation** the data is cut in k parts; each part is tested by a probe trained on the others. Items here come in families (88 hypotheses share a Corr2Cause premise; many questions share a CLadder story), and family members look alike, so a whole family must sit on one side: **grouped** folds. *Example:* all 8 pooled questions about premise #17 are either all training or all test. *Code:* `data.grouped_folds`.

### Baselines and controls for probes
A high probe score can have boring causes. Controls rule them out one at a time. **Shuffled labels:** train on scrambled labels; should fall to 50%, otherwise the pipeline leaks. **Text baseline (TF-IDF):** word counts of the prompt, same classifier; if it matches the probe, the model did no work. **Type prior:** predict from the question type alone. **Random-init model:** same architecture with untrained weights; whatever a probe reads there is in the input, not learned. *Code:* `probes.shuffled_labels`, `probes.tfidf_oof`, `probes.query_only_oof`, `models.load_model(random_init=True)`.

### Control task and selectivity (Hewitt & Liang, 2019)
A probe with enough capacity can memorise. A **control task** gives each input *type* a fixed random label (here: each hypothesis string, or each graph × query-type pair). Nothing about causality predicts these labels; only memorising type→label does. **Selectivity** = accuracy on the real task minus accuracy on the control task. High selectivity means the probe is reading structure rather than memorising. *Example:* real 80%, control 55% → selectivity 25 points. *Code:* `probes.control_task_labels`.

### Confidence interval and the (cluster) bootstrap
An accuracy from 1,000 items is an estimate; a **95% confidence interval** is a range that would cover the true value in about 95% of repetitions of the study. The **bootstrap** builds it by resampling the items with replacement many times and looking at the spread of the recomputed statistic. Because items sharing a premise or story are not independent, we resample whole premises/stories (**cluster** bootstrap), which gives honestly wider intervals. *Example:* accuracy 0.56 [0.48, 0.63]: we cannot rule out chance (0.50). *Code:* `stats.bootstrap_ci`.

### Accuracy, macro-F1 and AUROC
**Accuracy** is the share answered correctly; with 85% "No" labels, always saying "No" scores 85%, so accuracy alone misleads. **F1** for a class balances precision and recall; **macro-F1** averages F1 over both classes, so ignoring the rare class is punished. **AUROC** asks whether a score ranks positives above negatives (0.5 = no better than chance, 1 = perfect), independent of any threshold. *Code:* `stats.macro_f1`, `stats.auroc`, `stats.baselines`.

### Answer bias versus chance
A model can score 50% for two different reasons: its answers carry no information, or they carry information but lean heavily towards one answer. To tell them apart we report three things. The **Yes-rate** (how often it says Yes). The **AUROC of the answer score**, which only asks whether true-Yes items get higher scores than true-No items and so ignores any constant lean. And the **accuracy at the best threshold**: move the cut-off between Yes and No to wherever helps most, choosing it on one half of the data and scoring on the other. *Example:* a model that says "No" to everything has accuracy equal to the share of No labels, yet if its Yes-score is a bit higher on true-Yes items its AUROC is above 0.5: biased, not empty. *Code:* `scripts/15_answer_bias.py`.

### Partial-input baselines (hypothesis-only, premise-only)
Each benchmark item has parts: a premise and a hypothesis. A **partial-input baseline** is a classifier that is shown only one part. If it beats chance, the label leaks through that part alone, and anything that can read that part (including a probe on a model's activations) can score well without solving the task. The idea comes from natural language inference, where "hypothesis-only" classifiers exposed annotation artefacts (Poliak et al., 2018; Gururangan et al., 2018). *Example:* in Corr2Cause, "D directly causes C." is never valid whatever the premise says, because variables are named in causal order; a classifier that has only memorised each hypothesis string's track record reaches 86% on a balanced pool. *Code:* `probes.hyp_prior_oof`, `scripts/14_benchmark_baselines.py`.

### Target encoding, and how leave-one-out leaks the label
**Target encoding** turns a category (here: the hypothesis string) into a number: the average label of the training items in that category. It must never include the item's own label. A tempting shortcut is **leave-one-out**: average over all *other* items in the category. That leaks. If a category has 4 valid and 4 invalid items, a valid item sees 3/7 = 0.43 and an invalid item sees 4/7 = 0.57, so the feature is *lower exactly when the label is 1*. A flexible model learns "slightly below average means valid", scores brilliantly in training and collapses on new data. The fix is **out-of-fold** encoding: compute each item's feature from other folds only, with whole premises held out together. *Example:* my first parsed-feature baseline used leave-one-out and collapsed on held-out data for exactly this reason; a test now guards against it. The same mechanism later made a hypothesis-only classifier score *below* chance on the first debiased pool. *Code:* `probes._oof_prior`, `tests/test_baselines.py`.

### Debiasing an evaluation set
If a shortcut exists, one fix is to build an item set where the shortcut carries no information. Here: every premise contributes one valid and one invalid hypothesis, every hypothesis string is used equally often as valid and as invalid, and the set is cut into blocks that each satisfy both rules, so that every cross-validation fold does too. Then any classifier looking at the hypothesis alone or the premise alone gets exactly 50%, and only the *combination*, the actual task, can help. *Example:* in the debiased pool "A directly causes D." appears 25 times as valid and 25 times as invalid, five and five in each fold. *Caveat:* balancing only the whole pool is not enough; see the previous entry and decision D21. *Code:* `data.c2c_debiased_pool`, `data.grouped_folds`, `tests/test_data.py`.

### Confound controls versus reference baselines
Baselines for a probe come in two kinds, and it matters which one you are comparing with. A **confound control** asks "could the probe be scoring for a boring reason?": word statistics of the prompt (bag-of-words), the question alone (hypothesis prior), the premise alone, the network architecture without training (random-init probe), or a leak in the pipeline (shuffled labels). A probe has to beat these before its score means anything. A **reference baseline** is different: it is a deliberate attempt to do the task from the text, here a hand-written parser that looks up what the premise says about the two variables in the question. It is not something to "rule out"; it is a yardstick for how much of the task explicit structure-reading achieves. *Example:* on the debiased pool the 1.5B model's probe scores 0.75. Against the confound controls (at most 0.58) that is a real signal. Against the reference (0.82) it is less than a parser gets. Both statements are true and they answer different questions: "is there something there?" and "how much?". *Code:* `scripts/05b_probe_content.py`, table `debiased_two_groups`.

### Incremental predictive value
Two predictors can each be good and still carry the *same* information. To ask whether predictor B knows anything A does not, fit one model with A alone and another with A and B together, and compare them on held-out data. The difference is B's **incremental value**. Two rules keep it honest. First, B must be produced without ever seeing the labels of the items it is evaluated on; when B is itself a trained probe this needs **nested** cross-validation (a cross-validation inside each training fold). Second, report the difference with a confidence interval computed on the same items for both models (a *paired* comparison). *Example:* parsed features alone 0.816; parsed features plus the probe's score 0.822; gain +0.006 with an interval from −0.015 to +0.026. The interval includes zero: no evidence that the activations hold anything the parser does not. A first, non-nested attempt had suggested +0.023, which shows why the nesting matters. *Code:* `nested_probe_feature` in `scripts/05b_probe_content.py`, table `debiased_incremental`.

### Stratified analysis
To check whether an effect is "really" due to some other variable, split the data into groups (**strata**) in which that variable is constant, and measure the effect inside each group. If the effect vanishes inside the groups, the variable explained it; if it survives, it did not. *Example:* is the probe just reading "are the two queried variables adjacent"? Inside the items where they are adjacent, and separately inside the items where they are not, the probe still ranks valid above invalid items almost as well as overall (AUROC 0.82 against 0.83), so adjacency alone is not what it reads. Two cautions. Inside a stratum the labels are usually not 50/50, so use a measure that does not care about the base rate (AUROC), not accuracy. And strata must not be too small or too many: with dozens of tiny groups everything looks noisy. *Code:* part D of `scripts/05b_probe_content.py`, table `debiased_strata`.

### Decodable is not the same as used
A probe finding information in a layer shows the information is *present*. It does not show the model *uses* it: the information could be a by-product that no later layer reads. To claim use, you must intervene (change the activation and watch the output) rather than just observe. The next four entries are interventions. *Example:* a thermometer on a car's exhaust lets you decode engine speed, but warming the exhaust does not make the car go faster. *Code:* observational = `04_probes.py`; interventional = `07_steering.py`, `08_patching.py`.

### Steering vector
Pick a direction in activation space (a vector), add a multiple of it to the residual stream at some layer and position during the forward pass, and see how the output changes. We summarise the change by a **slope**: how many units the Yes−No logit difference moves per unit of vector added, measured with both signs (+ and −) so that a constant offset cancels. *Example:* adding +1 unit moves the logit difference from −1.1 to −0.2 and adding −1 unit moves it to −2.0; the slope is (−0.2 − (−2.0)) / 2 = 0.9. We test three directions: the gold-label probe's weight vector, the difference between the mean activation of gold-Yes and gold-No items, and a probe for the model's own answer (a positive control: it is fitted to the very quantity we measure). *Code:* `steer.AddAtLast`, `steer.steered_logit_diffs`, `scripts/07_steering.py`, `scripts/07b_steering_analysis.py`.

**Three traps, all found by the second audit.** (1) *Units.* We scale vectors by s, the typical distance of an activation from the average activation. That is a length over all ~1,000 dimensions together. Along any single direction the natural spread is far smaller, and along a probe's weight vector smaller still (probes exploit quiet directions). One unit of s can be a hundred natural standard deviations along the probe direction, so "we pushed hard and nothing happened" and "we pushed a realistic amount" are very different statements; we report both scales. (2) *Controls must be of the same kind.* A difference of class means lies in the loud, high-variance part of the space; random directions drawn uniformly do not. The fair comparison for a real direction is the same construction applied to shuffled labels. (3) *A weight vector is not the feature.* The probe's weights and the difference of class means are two views of one discriminant: the first says which combination of activations to read, the second says how activations actually differ between classes. Later layers responding to the second but not the first does not mean the information is unused. *Code:* `scripts/07c_steering_controls.py`.

```mermaid
flowchart LR
    P[prompt] --> L1[layers 0 .. L*-1]
    L1 --> R((residual stream<br/>final token))
    V[alpha x direction] -->|add| R
    R --> L2[layers L* .. end] --> O[Yes - No logit difference]
```

### Directional ablation
The opposite of steering: instead of adding a direction, remove the activation's own component along it (set it to the dataset average). If the model relied on that component, its answer should move. *Example:* an item sits 3 units along the direction and the average is 1; ablation subtracts 2 units of the direction. *Code:* the `ablate` block in `scripts/07_steering.py`.

### Minimal pair
Two prompts that differ in as little as possible but have different correct answers. Any difference in the model's behaviour between them must come from that small difference. We use two kinds, both with identical token counts: **hyp** (same premise, hypothesis asks about a different variable) and **prem** (same hypothesis, two variable letters swapped throughout the premise, i.e. a relabelled causal graph). *Example:* "…Hypothesis: A directly causes B." (Yes) vs "…Hypothesis: A directly causes C." (No). *Code:* `patch.build_pairs`, `patch.encode_pair`.

### Activation patching
Run the model on the clean prompt and save its activations. Run it on the corrupt prompt, but at one chosen place (a layer and a token region) overwrite the activation with the clean one. If the answer moves towards the clean answer, that place carries the information that distinguishes the two prompts. Sweeping over layers and regions maps where the answer-relevant information lives and when it moves. *Example:* overwriting the final token's residual stream at layer 15 restores 80% of the clean-vs-corrupt logit difference, at layer 14 only 11%: the information arrives at the final token around layer 15. *Code:* `patch.ResidPatcher`, `patch.resid_patch_grid`, `scripts/08_patching.py`.

```mermaid
flowchart LR
    C[clean prompt] --> CA[(clean activations)]
    K[corrupt prompt] --> F[forward pass]
    CA -->|overwrite layer l, region r| F
    F --> O[LD patched]
    O --> M["recovery = (LD patched - LD corrupt) / (LD clean - LD corrupt)"]
```

### Random-donor control for patching
Overwriting an activation with *anything* might disturb the answer. As a control we patch in activations from the clean run of an unrelated premise of the same token length. If that "recovers" as much as the true clean run, the effect is not specific to the pair's causal content. *Code:* the `ctrl` grid in `scripts/08_patching.py`.

### Attention heads and MLPs
Each layer has two kinds of component that write into the residual stream: several **attention heads** (each moves information between token positions) and one **MLP** (which transforms information at a position). Patching one head or MLP at a time asks which specific components matter. *Code:* `patch.ComponentIO`.

### Attribution patching
Patching every head and MLP separately needs one forward pass each (hundreds). **Attribution patching** approximates all of them at once with one backward pass: effect ≈ (clean activation − corrupt activation) × gradient of the output with respect to that activation. It is a first-order (straight-line) approximation, so it can be wrong; we therefore re-check the top-ranked components and some random ones with real patching and report the correlation. *Example:* head L14H5: attribution 0.21, real patching 0.20. *Code:* `patch.record_components`, Part B of `scripts/08_patching.py`.

### Sparse autoencoder (SAE) and features
A residual-stream vector has a few thousand numbers, and each number means nothing by itself: concepts are spread across many of them. A **sparse autoencoder** is a separately trained network that rewrites an activation vector as a sum of a few "features" out of a very large dictionary (here 16,384 per layer), each feature being a fixed direction with a non-negative strength. "Sparse" means only a few dozen features are active for any one input. The hope is that single features correspond to recognisable things. *Example:* an activation might be rewritten as 3.1 × feature 2041 + 0.8 × feature 977 + … + a small leftover (the reconstruction error); if feature 2041 fires on exactly the prompts that describe a collider, it is a candidate "collider feature". We use Gemma Scope, public SAEs trained by Google DeepMind on Gemma-2-2B (the base model; we apply them to the instruction-tuned variant, and report how well they reconstruct our activations). *Code:* `scripts/11_sae.py`, `load_sae`, `encode`.

**How features are selected and tested here.** CLadder stories about real topics each use one causal graph type, so a feature that fires on "smoking" would look like a "confounding" feature. Only the ten made-up-word stories are used, because each occurs with every graph type. Features are *chosen* on five of those stories (which features best separate one graph type from the rest) and *evaluated* on the other five, against the best features found for shuffled graph labels. Reconstruction quality is reported relative to the activation's own size; a measure relative to the variation between prompts is meaningless at the final token, where every prompt has the same token.

### Feature ablation
To test whether a feature matters for the answer, subtract its contribution from the residual stream during the forward pass (at every token position, at the SAE's layer) and see whether the Yes−No logit difference changes. The control is to ablate the same number of *other* features that are equally active on these prompts; removing anything perturbs the model a little. *Code:* `FeatureAblator` in `scripts/11_sae.py`.

```mermaid
flowchart LR
    P[prompt] --> L1[layers up to the SAE layer]
    L1 --> R((residual stream<br/>every token))
    R --> S[SAE encode] --> F[strengths of the chosen features]
    F -->|"subtract strength x feature direction"| R
    R --> L2[remaining layers] --> O[Yes - No logit difference]
```

*(A LoRA fine-tuning tier was planned and dropped for compute reasons; see decision D20.)*

---

## Decision log

Each entry was written when the decision was made. Format: decision · alternatives · why · what would have changed it.

**D1. Plain Hugging Face models with forward hooks; TransformerLens only as a cross-check.** (Thu 17:20)
*Alternatives:* TransformerLens for everything. *Why:* the installed TransformerLens is 4.4, where `HookedTransformer` no longer exists; its replacement `TransformerBridge` wraps the HF model and forces "eager" attention, which is slower and uses more memory on an 8 GB machine. HF hooks give the same residual stream with no extra cost. We verified the bridge's logits equal HF's on 5 prompts (max difference 7e-5, `results/env_check_qwen0.5b.json`). *Would change it:* a mismatch between the bridge and HF, or needing TL-only utilities.

**D2. Models: Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-2-2B (all instruct). No 3B model.** (Thu 17:20)
*Alternatives:* add Qwen2.5-3B. *Why:* the Mac has 8 GB RAM in total, shared between CPU and GPU. 3B weights in float16 are ≈6.2 GB, leaving too little for activations and the OS; it would swap for hours. Gemma-2-2B (≈5.2 GB) is the largest that plausibly fits and is needed for the SAE tier. *Would change it:* 16 GB of RAM.

**D3. Run models in float16 on MPS.** (Thu 17:22)
*Alternatives:* float32 on MPS; float32 on CPU; bfloat16. *Why:* measured on Qwen-0.5B: MPS float16 is 8.5 items/s vs CPU float32 2.7 and MPS float32 0.8 (it thrashes memory). Float16 changes logits by at most 0.05 and the answer logit difference by 0.023, never the top-1 token; bfloat16 is 4× less accurate at the same speed. *Would change it:* NaNs or answer flips in float16 for a larger model (checked per model in `results/env_check_*.json`).

**D4. Subsample sizes and the cache arithmetic.** (Thu 17:25)
Cached sets: Corr2Cause test (all 1,162), a balanced Corr2Cause probe pool from train (300 premises × 8 hypotheses = 2,400, half positive), CLadder (3,000 stratified by label × query type), the renamed test set (1,162) and two 500-item few-shot sets: 8,724 items, plus 5,400 for the random-init control. Bytes per item = (layers+1) × 3 positions × width × 2: Qwen-0.5B 25×3×896×2 = 134 KB; Qwen-1.5B 29×3×1536×2 = 267 KB; Gemma 27×3×2304×2 = 373 KB. Totals ≈ 1.9 + 3.8 + 5.3 = 11 GB against 252 GB free. Disk is not the limit; forward-pass time is (≈3 M tokens per model). *Alternatives:* cache all 10k CLadder items / more train premises. *Why not:* each extra thousand items costs Gemma ≈10–15 minutes and the probe CIs are already driven by the number of premises/stories, not items. *Would change it:* CIs too wide to separate probe from baseline.

**D5. Probe pool for Corr2Cause is balanced and sampled by premise; the natural-distribution test set is kept for behaviour and an external probe check.** (Thu 17:25)
*Alternatives:* probe directly on the 1,162 test items. *Why:* test has only 21 distinct premises and 15.5% positives; a probe trained there would have ~180 positives from ~21 families. Sampling 300 train premises with 4 positive + 4 negative hypotheses each gives chance = 50% and 300 independent families. Imbalance is handled explicitly: balanced pool for probing, macro-F1/AUROC and majority baselines for behaviour. *Would change it:* nothing foreseeable.

**D6. Prompt format and positions.** (Thu 17:25)
Instruction first, then premise, then hypothesis, single user turn, default system prompt of each template; the assistant turn is left empty so the next token is the answer. *Alternatives:* instruction last; prefilled "Answer:". *Why:* with the query last, "end of user text" coincides with "end of hypothesis", giving a principled second position; no prefill keeps to the documented template. Positions are found from character offsets, taking the token that covers the last character of the span (tokenizers merge ".\n" into one token). *Would change it:* a model that mostly emits non-answer tokens first (measured: `top1_not_an_answer_token`).

**D7. Probe = logistic regression on standardised activations, L2 with C = 0.01 fixed in advance, 5 grouped folds × 3 seeds.** (Thu 17:30)
*Alternatives:* tune C per layer by inner cross-validation; PCA first; MLP probes. *Why:* width (896–2304) is comparable to the number of training items (≈1,900), so strong regularisation is needed; fixing it in advance removes any temptation to tune until something works, and keeps ~3,000 fits per model affordable. Linear only, because the question is about *linear* decodability. A sensitivity check at other C values is reported, not used for selection. *Would change it:* the sensitivity check showing conclusions flip with C.

**D8. Control task adapted from word types to item types.** (Thu 17:30)
Hewitt & Liang assign a random label per word type. Our unit is a whole prompt, so the type is the hypothesis string (Corr2Cause) or graph × query type (CLadder). *Alternatives:* a random label per premise/story (unlearnable under grouped folds, so it would trivially give 50% and flatter selectivity); a random label per item (same problem). *Why:* types recur across premises/stories, so a memorising probe *can* succeed, which is what makes the control informative. *Would change it:* nothing; the limitation (it is an adaptation) is stated wherever selectivity is reported.

**D9. Robustness check = our own variable renaming (A,B,C… → Z,Y,X…) of the test set.** (Thu 17:30)
*Alternatives:* the released perturbation files. *Why:* the released files have 2,246 rows that do not align with the 1,162-row test set and differ in label scheme (upstream issue U4), so no paired comparison is possible. Renaming our own test items gives a paired design: same item, same label, different letters. *Would change it:* an aligned release.

**D10. All CIs are cluster bootstraps over premises/stories (2,000 resamples).** (Thu 17:30)
*Alternatives:* item-level bootstrap. *Why:* 1,162 test items come from 21 premises; treating them as independent would understate uncertainty several-fold. *Would change it:* nothing.

**D11. Headline layer chosen on half the premises/stories, reported on the other half; "represents but doesn't say" criterion fixed before seeing results.** (Thu 17:35)
*Alternatives:* report the best layer on the same data. *Why:* picking the maximum of ~27 noisy numbers inflates it. The criterion (in `05_confound.py`): on items the model gets wrong, the gold probe's accuracy CI must exclude 50% and its paired advantage over TF-IDF and over the type prior must exclude 0. We additionally report within each model-answer stratum (model said Yes / said No), where the model's binary answer is constant and so cannot help the probe, and compare with the model's own graded confidence. *Would change it:* nothing; written before any probe was run.

**D12. Gemma's embedding matrix lives on disk; Gemma runs one item at a time and stores answer-token logits only.** (Thu 19:00)
*Alternatives:* drop Gemma; run the full model anyway; a 4-bit quantised Gemma. *Why:* the full float16 model (5.3 GB) thrashes 8 GB of unified memory (0.07–0.2 items/s; ~20 hours for the cache). Gemma's embedding matrix is 256k × 2304 = 1.18 GB and is used for exactly two things: looking up the input tokens' rows, and producing logits. Keeping it as a memory-mapped file and computing only the Yes/No rows per item brings the GPU footprint to 4.4 GB and the speed to ~1.1 items/s. The arithmetic is unchanged (answer logit differences agree with the full model to 0.012 on the check prompts). Dropping Gemma would lose the SAE tier; quantisation would change the model being studied. *Cost:* no full-vocabulary top-1 token in Gemma's cache, and its generated-text check is first-token-only on 100 items per set. *Would change it:* 16 GB of RAM.

**D13. A "calibrated" model answer (median split of the logit difference) is analysed next to the raw answer.** (Thu 19:10, after seeing data)
*Alternatives:* raw answer only. *Why:* Qwen-0.5B answers "No" to essentially everything, so its raw answer is a constant; "model-wrong items" is then just "gold-Yes items" and a model-output probe has nothing to predict. The median split removes the constant bias and asks about the model's *relative* preference. This was added after seeing results, so the pre-registered verdict (D11) stays on the raw answer and the calibrated one is labelled secondary. *Would change it:* n/a.

**D14. Steering: three meaningful directions + random; scale in units of typical activation spread; final position only; KV-cache shortcut.** (Thu 19:35)
*Alternatives:* steer at all positions; scale in units of the residual norm; probe direction only. *Why:* the probes were trained at the final position, so that is where the direction is defined. The unit s = RMS distance of an activation from the layer mean makes "alpha = 1" comparable across models of different width. Difference-of-means is included because probe weights are discriminative (they include an inverse-covariance factor) and are known to steer worse than mean differences; reporting only the probe direction could wrongly suggest "no causal role". Directions are fitted on the selection half and tested on the report half. Because a final-position edit cannot affect earlier positions, the prefix is computed once and only the last token is re-run per vector (tested equal to the full computation). *Would change it:* a need to steer earlier positions.

**D15. Minimal pairs on Corr2Cause only, two kinds (hyp, prem), premises with at most 5 variables; effect = pooled recovery.** (Thu 19:40)
*Alternatives:* CLadder pairs; per-pair normalised recovery; only pairs the model answers correctly. *Why:* Corr2Cause has exact token-aligned pairs for free (same premise, sibling hypotheses), and relabelling variables gives a second kind that changes the causal structure rather than the question. CLadder stories do not align token-for-token. Five-variable premises keep prompts ~350 tokens so the sweep is affordable. The models are near chance, so "pairs answered correctly" would leave almost nothing; pooled recovery (sum of signed effects over sum of |clean−corrupt| differences) uses all pairs and weights them by how much the model distinguishes them, in whichever direction. *Would change it:* a model that solved the pairs (then the standard per-pair normalisation on correct pairs would be reported too).

**D16. Patch token regions, not single tokens; attribution patching for components, validated on top-10 + 10 random.** (Thu 19:40)
*Alternatives:* per-token patching; exhaustive head patching. *Why:* (layers × ~350 tokens) forward passes per pair is out of reach; five regions (differing tokens, rest of premise, rest of hypothesis, chat-template tail, final token) answer "where" at the granularity we can interpret. Exhaustive component patching is 300–450 passes per pair; one backward pass approximates it, and the 20-component validation tells us whether to trust it. *Would change it:* validation correlation below ~0.7 (then only really-patched components would be reported).

**D17. (Revises D7.) Probe regularisation is chosen from a small grid on the selection half; the fixed-C result is kept as a sensitivity analysis.** (Thu 19:57–20:20, after seeing the pre-planned sensitivity check)
*What happened:* D7 fixed C = 0.01 and said a sensitivity check showing the conclusion depends on C would change it. The check showed Corr2Cause probe accuracy rising from 0.76 (C = 0.01) to 0.80 (C = 1), while the TF-IDF baseline had been run at C = 1: the probe was handicapped. *Decision:* run every probe (all targets, controls and the random-init model) at C ∈ {0.01, 0.1, 1} and TF-IDF at C ∈ {0.1, 1, 10}. D11 already splits premises/stories into a selection half and a report half to choose the layer; now the pair (layer, C) is chosen there too, and the TF-IDF C likewise. Every reported number is on the report half, which plays no part in any choice. *Alternatives tried:* true nested cross-validation inside every training fold (statistically equivalent in spirit, but 10× the fits; it ran 20 minutes without finishing one model and was abandoned); keep C = 0.01 (biased against the probe); choose C by test accuracy (tuning on the test set). *Safeguard:* the original setting (probe C = 0.01, TF-IDF C = 1) is reported next to the selected one (`results/confound_*_fixedC.json`), because this change was made after seeing results and it moves towards the more exciting hypothesis. *Would change it:* nothing further; the probe has no other hyperparameters.

**D18. A stronger, parsed-feature text baseline is added as a secondary check; the pre-registered verdict is left as it is.** (Thu 23:00, after the criterion was first met)
*What happened:* for Qwen-1.5B on Corr2Cause the D11 criterion came out positive: on items the model gets wrong, the gold probe beats TF-IDF and the type prior. *Decision:* before believing it, test it against a text baseline that can do what bag-of-words cannot, namely tie premise statements to the two variables in the hypothesis (Corr2Cause), or compare the quoted probabilities and read the question's polarity (CLadder), using gradient boosting on those parsed features. *Alternatives:* accept the TF-IDF comparison as specified (it is what was pre-registered, but TF-IDF is a weak model of "what is in the text" for a task whose label is a function of the text); swap the baseline silently (would hide a post-hoc choice). *Why:* the claim "the model represents the answer beyond what the text gives away" is only as strong as the text baseline. This check can only weaken the claim, so adding it after the fact does not risk a false positive. Both verdicts are reported side by side. *Would change it:* nothing.

**D19. The verdict is re-defined after the independent audit ("audited verdict"); the pre-registered one is kept and labelled.** (Fri 00:05)
*What the audit found:* (1) on Corr2Cause the exact hypothesis sentence predicts the label with ~0.86 accuracy without reading the premise, which neither TF-IDF nor my template prior captured; (2) "accuracy above 50% on items the model gets wrong" is not a test against chance when those items are almost all gold-Yes. *Decision:* the test that licenses "represents more than the text gives away" is now: the gold probe must beat the **strongest text-only baseline** (chosen on the selection half among TF-IDF, template prior, hypothesis-string prior, question-only TF-IDF, and gradient boosting on parsed features plus that prior), on all report-half items (paired accuracy) and inside both calibrated model-answer strata (paired AUROC), each with a 95% CI excluding zero. *Alternatives:* keep the D11 criterion as coded (it was passed by a model for reasons the audit showed to be artefacts); rebuild the probe pool balanced within hypothesis string (cleaner, but needs a new cache for every model: ~4 GPU-hours, not affordable before the Tier 1 gate; listed under "what I'd do next"). *Why:* a criterion should be as hard to pass by shortcut as we can make it; this one can only be stricter than D11. Both verdicts are reported in every table. *Would change it:* a still stronger text-only baseline that someone proposes; the claim is always relative to the baselines tried.

---

### D20. Heavy GPU work moves to Kaggle; the laptop does CPU analysis only (Fri 13 Nov, 09:40)
*Decision:* Gemma-2-2B steering, patching and the SAE tier, and one larger model, run in a Kaggle GPU notebook (`notebooks/kaggle_gpu.ipynb`, `scripts/kaggle_run.py`, `KAGGLE.md`). The laptop runs probes, baselines, figures and the two small Qwen models, one job at a time, threads capped. No further Gemma forward passes locally. The LoRA tier is dropped unless time remains.
*Alternatives:* keep everything local with smaller batches; drop Gemma's Tier 2.
*Why:* Gemma steering stalled overnight (item 0 of 300 after 52 minutes, process swapped out). Gemma's weights alone are about 5.2 GB of the 8 GB shared by CPU and GPU, so any intervention that keeps a key-value cache per steering vector does not fit. A free Kaggle GPU has 16 GB and also makes a model large enough to possibly answer above chance affordable. Decided before any Gemma Tier 2 number existed.
*Would change it:* a 16 GB machine.
*Consequence to state in the write-up:* Gemma's probes come from activations cached on the Mac (float16, MPS, embedding matrix on disk); Gemma's steering, patching and SAE results come from activations recomputed on Kaggle (float16, CUDA). `results/cache_agreement_gemma2b.json` reports how closely the two agree per item.

### D21. A debiased Corr2Cause pool decides "represents but doesn't say" (Fri 13 Nov, 09:45; revised 10:40)
*Decision:* new item set `c2c_debiased` (`data.c2c_debiased_pool`), built from the train split: 2,400 items over 1,200 premises and the 48 hypothesis strings (of 120) that occur with both labels. It is made of 25 blocks. Inside a block every hypothesis string appears exactly once as a valid and once as an invalid hypothesis. Every premise is used once in the whole pool and contributes exactly one valid and one invalid hypothesis. Cross-validation folds and the selection/report split are unions of whole blocks. So in every training set and every test set, each hypothesis string and each premise is exactly half valid, and any classifier that sees only the hypothesis, or only the premise, scores exactly 0.50. The probe-versus-text-baseline comparison on this pool, not on the original pool, decides the headline claim.
*First version, and why it was replaced (10:35, before any model was run on it):* I first balanced labels within each hypothesis string over the whole pool (25 valid + 25 invalid items per string) and kept ordinary premise-held-out folds. The text baselines then showed two problems. (1) A hypothesis-only classifier scored 0.315 accuracy, AUROC 0.247: below chance. With exact balance in the whole pool, a string that happens to be mostly valid in the test fold is mostly invalid in the training folds, so its training-fold prior points the wrong way. Anything that can read hypothesis identity, including a probe, would be pushed below chance by this, which would bias the test against the probe. (2) A premise-only classifier scored 0.722: valid items come disproportionately from premises with many valid hypotheses. The block design removes both: measured on the new pool, hypothesis-only 0.500, premise-only 0.500, full-text TF-IDF 0.495 (`results/benchmark_baselines.json`).
*Alternatives:* (a) subset the existing balanced pool: only 648 of its 2,400 items survive string balancing, too few; (b) keep the original pool and only add the hypothesis-string prior as a baseline (done in D18/D19, but a probe can still score by decoding hypothesis identity); (c) reweight instead of subsample.
*Why:* in the original pool the hypothesis string alone predicts the label with 0.86 accuracy, so a probe that reads "which hypothesis is this" looks like a probe that reads "is this valid". The pool needs new forward passes for every model, which is why it was fixed before the Kaggle notebook was launched.
*What still carries label information in the text:* only the combination of premise and hypothesis, which is the task itself. The parsed-feature baseline (does the premise say the two named variables are correlated, adjacent in the skeleton, how many separating sets, shared neighbours) reaches 0.812 on this pool. That is partial reasoning done by a hand-written parser, and it remains the bar a probe has to clear under the audited criterion.
*Would change it:* if reviewers consider the parsed-feature baseline too strong a bar (it implements the first step of the PC algorithm), report the probe against the hypothesis-only, premise-only and bag-of-words baselines as well, which is done.
*Post-hoc status:* this pool was designed after seeing the Tier 1 results and the audit, and revised after seeing text baselines (not model results) on its first version. It is not pre-registered. The criterion applied to it is the audited one (D19), unchanged.

### D22. Steering layer for models without a full probe analysis (Fri 13 Nov, 09:50)
*Decision:* `07_steering.py` takes the layer and probe regularisation from `05_confound.py` when its output exists (the three local models on the original item sets). Otherwise (the larger model on Kaggle; every model on the debiased pool) it picks them itself with the same rule, best selection-half accuracy over the (layer, C) grid, using a cheaper 3-fold grouped cross-validation inside the selection half (`quick_select`). The choice and its source are stored in the result (`layer_selection`). Two extra runs at fixed relative depths 0.5 and 0.85 are added as a layer-choice sensitivity check.
*Alternatives:* run the full probe pipeline on Kaggle first (hours of CPU on the GPU session); steer at one arbitrary depth.
*Why:* the probe analysis for the larger model runs on the laptop after the cache comes back, but steering needs the model on a GPU now. The report half is never used for selection in either route.
*Would change it:* if the layer later chosen by `05_confound.py` differs by more than a few layers from the quick choice, rerun steering at that layer in a second Kaggle session.

### D23. Only final-token activations leave Kaggle (Fri 13 Nov, 09:55)
*Decision:* the Kaggle export keeps the residual stream at the final prompt token only (float16, all layers), for the three probe sets, trained and random-init. Qwen2.5-3B: 37 layers × 2,048 dims × 2 bytes ≈ 152 KB per item, × 7,800 items ≈ 1.2 GB per archive. Per-item logits and generated answers for the other item sets are exported without activations.
*Alternatives:* all three positions (3.5 GB per archive).
*Why:* download size. *Cost:* the position comparison (final token versus end of hypothesis versus end of premise) is not available for the larger model, nor for Gemma on the debiased pool. `--export-positions final,content_end,premise_end` restores it.

### D24. Steering is summarised by signed slopes and judged against matched controls (Fri 13 Nov, 11:50; after the Tier 2 audit)
*Decision:* the headline steering number is the slope of the Yes−No logit difference per unit of vector added, from both signs, with a cluster-bootstrap interval (`07b_steering_analysis.py`). Each real direction is compared with 20 directions of the same construction fitted to shuffled labels, and with 100 isotropic directions (`07c_steering_controls.py`). Effects are also reported per natural standard deviation along each direction. Flip rates stay in a secondary table; "accuracy towards gold" is renamed controllability; no ablation comparison is claimed.
*Alternatives:* keep flip rates and three random directions (the original design).
*Why:* the audit showed the original summary hid a sign-consistent effect, could not support "no more than random", and used steps of 40 to 300 natural standard deviations along the probe direction.
*Would change it:* nothing foreseeable; more control directions would tighten the percentiles.
*Post-hoc status:* metric and controls were changed after seeing results, at an auditor's request, in the direction of a weaker claim.

### D25. Two kinds of baseline, and three tests of what the probe adds and reads (Fri 13 Nov, 13:58; user request)
*Decision:* on the debiased pool, baselines are reported in two labelled groups: shallow confound controls (bag-of-words, hypothesis prior, premise-only, random-init probe, shuffled labels) and a structure-aware reference (parsed features). Three analyses are added (`05b_probe_content.py`): incremental predictive value of the probe over the parsed features (nested cross-validation); probes for individual parsed properties with shuffled-label and random-init controls; the gold probe's AUROC inside strata of the strongest properties.
*What does not change:* the pre-registered criterion (D11) and the audited criterion (D19) and their verdicts. On the debiased pool the audited criterion is not met; the pre-registered one, as coded, is met for both Qwen models and is reported but not relied on.
*Why:* "below the best text baseline" lumps together two different comparisons. Being above the confound controls says the signal is real; being below the reference says how much of the task it amounts to.
*Would change it:* if the third audit finds a cheap feature between bag-of-words and the full parse that matches the probe, the "beyond shallow features" wording has to be narrowed to name that feature.
*Post-hoc status:* all of it.

### D26. Running the 1.5B model on the laptop under a memory watchdog (Fri 13 Nov, 11:40)
*Decision:* Qwen 1.5B GPU jobs run through `scripts/guarded.sh`: batch size about one prompt, GPU memory capped just above the model size (PyTorch MPS watermarks), two CPU threads, nothing else running, and a watchdog that stops the job when swap use passes 5.2 GB while growing. Cache builds resume from finished chunks.
*Alternatives:* move every 1.5B pass to Kaggle; or run uncapped as on Thursday night.
*Why:* the user's rule is to stop any job that pushes memory into swap. A 1.5B model (3.1 GB of weights) on an 8 GB machine with a desktop session cannot run with zero swap; with the cap, swap stayed flat at 3 to 4 GB and throughput steady. The first uncapped attempt reached 5.5 GB of wired memory and was stopped.
*Would change it:* if the user prefers zero swap use, the remaining 1.5B passes go to Kaggle.

## How to read each figure

All figures are drawn by `scripts/06_figures.py` from `results/*.json`. Shaded bands and whiskers are 95% cluster-bootstrap confidence intervals (premises or stories are resampled). Each model keeps the same colour in every figure.

**fig1_behaviour.png — how well do the models answer?** One bar per model: macro-F1 and accuracy on the Corr2Cause test set, accuracy on CLadder, scored from the answer-token logits. The dashed line is the majority-class baseline (always answer the commoner label) and the dotted line is a coin flip. A bar level with the dashed line means the model is doing no better than answering "No" every time. On Corr2Cause read macro-F1, not accuracy: 84.5% of labels are "No", so accuracy rewards a model that never says "Yes".

**fig2_logit_lens.png — when does the answer take shape?** x: layer. y: the Yes−No logit difference the model would produce if it stopped at that layer (positive = leaning Yes). Blue: items whose gold answer is Yes; orange: gold No. Solid: the model's final answer is right; dashed: wrong. A model that tracked the truth would pull blue above orange. What the panels show instead is all four curves moving together: the lens shows a shared Yes/No drift, with at most a small gap between colours. (An earlier version of this figure plotted a "gold-signed" difference split by correct/incorrect; the audit pointed out that for a model with a near-constant answer that plot is a mirror image by construction and shows bias, not knowledge.) Layer 0 is omitted (the lens on raw embeddings is off-scale noise).

**fig2b_lens_auroc.png — the same lens, threshold-free.** y: AUROC of the per-layer lens logit difference for the gold label (0.5 = no information, 1 = perfect ranking). This ignores any constant Yes/No bias and asks only whether the model's would-be answer ranks true-Yes items above true-No items. x is relative depth so models with different numbers of layers can share an axis.

**fig3_probes_<dataset>.png — what can a linear probe read, and is it more than the controls?** One panel per model. Solid blue: held-out accuracy of the gold-label probe at each layer. Solid orange: probe for the model's own answer (shown only where the model's answers vary). Dashed yellow: control task (random label per question type; high = the probe can memorise). Dashed pink: same probe on a randomly initialised model. Dotted grey: shuffled labels (must sit at 0.5). Horizontal dash-dot lines are text-only baselines, which have no layers: TF-IDF of the whole prompt (black), template/type prior (violet), and the two added after the audit: the hypothesis-string prior (red, Corr2Cause only) and gradient boosting on parsed text features (green). The claim "the activations hold the answer beyond what the text gives away" needs blue clearly above *every* horizontal line. It is not.

**fig4_selectivity_positions_<dataset>.png.** Left: selectivity (gold accuracy minus control-task accuracy) against relative depth, one line per model; near or below zero means the probe's success is within what memorising question types could achieve. Right: the gold probe at three token positions for the largest model: final token, end of hypothesis/question, end of premise. The end of the premise comes before the question is asked, so a probe there cannot know which hypothesis is being tested; it is a built-in sanity check and should sit near the baselines.

**fig5_confound_<dataset>.png — the key test.** Report-half items only. Left: accuracy of each scorer on all report-half items. Middle and right: AUROC for the gold label *within* the items where the model leans Yes, and within those where it leans No ("leans" = its answer logit difference is above/below its own median, which removes a constant Yes/No bias); inside one group the model's answer is fixed, so a probe that merely reads that answer scores 0.5. Bars: gold probe (blue), parsed text features + hypothesis prior (green), hypothesis-string prior alone (red), full-prompt TF-IDF (black), random-init probe (pink), the model's own graded confidence (orange). "Represents more than the text gives away" needs blue above green and red by more than the whiskers, in all three panels.

**fig6_steering_<dataset>.png — what does pushing along each direction do to the answer?** One column per model. *Top:* x is the vector added at the final token, from −8 to +8 units of s (negative = pushed the opposite way; the axis is stretched near zero); y is the average change in the Yes−No logit difference. A direction the model reads gives a line through the origin with a clear tilt; a flat line means no effect. Thick lines are the three real directions, thin grey dashed lines the random ones. *Bottom:* the same effect expressed per *natural* standard deviation along each direction, on a log scale. This is the fair scale: the probe's weight direction is quiet, so a unit of s along it is a huge push. Read the bottom panel for "how much does a realistic movement along this direction matter", the top panel for the sign and shape. The matched random controls are in the table `steering_controls`, not in this figure.

**fig7_patching_<kind>.png — where does a change of variable reach the answer position?** Heat-map per model. x: layer at which the residual stream is overwritten; y: which group of tokens is overwritten (the tokens that differ between the two prompts, the rest of the premise, the hypothesis, the fixed template tail, the final token). Colour: pooled recovery; 1 (blue) means the patched run produces the other prompt's Yes−No logit difference, 0 (neutral) means no effect. Early layers: all the difference sits at the differing tokens. Late layers: it sits at the final token. The layer where the blue moves from one row to the other is where the information is carried across. **This is not "where the answer is decided"**: the models do not prefer the correct member of a pair, so the figure tracks any difference between the two prompts. *Bottom row:* patching in an unrelated prompt of the same length. Few pairs have such a donor and the statistic is not zero under the null, so this row is shown for completeness and is not used as a control.

**fig8_attribution_validation.png — can the cheap method be trusted?** Each point is one attention head or MLP. x: effect predicted by attribution patching (one gradient). y: effect measured by really patching that component. Blue: the ten components ranked highest by attribution; grey: ten random ones. Points on the diagonal mean the approximation is accurate. The agreement is established for the large components; among small ones the two can disagree in sign, and ten random components out of several hundred cannot show that nothing important was missed.

**fig9_cross_model.png — how do the models compare?** (A handful of models from two families; descriptive, not a scaling law.) x: parameters (log scale). Left: gold-probe accuracy on the report half. Middle: probe minus the strongest text-only baseline (paired); at or below zero means the activations add nothing linearly decodable beyond that baseline. Right: relative depth of the selected layer (large marker) and of the first layer within one point of the maximum (small marker). Filled circles: Corr2Cause pool; open squares: CLadder.

*(Tier 3 figures are added below when produced.)*

---

## Questions a professor would ask

Short model answers. Numbers are quoted from `report/tables/*.md` (generated from `results/*.json`); where a number is not given here, the table is named.

<!-- QA-TIER1 -->
### Tier 1 — behaviour, logit lens, probes

**1. Why score with answer-token logits rather than just reading the generated text?**
Logits give a graded score for every item (so AUROC and calibration are available), cost one forward pass, and cannot be "unparseable". The risk is that the first-token logits might not match what the model would actually write, so both were computed on 500 items per set; they disagree on under 1% of items for the Qwen models (`behaviour_generation.md`).

**2. Corr2Cause is 85% "No". How do you stop that from flattering the models?**
Three ways: macro-F1 and AUROC are reported next to accuracy; the majority-class baseline is shown on the same items; and probing uses a pool built to be exactly 50/50, so chance is 50%.

**3. The test set has 1,162 items. Why are your confidence intervals so wide?**
Because those items come from only 21 distinct premises, and hypotheses about one premise share their fate. The bootstrap resamples premises, not items. Treating items as independent would shrink the intervals several-fold and be wrong.

**4. What exactly is a "held-out" item for a probe here?**
One whose whole premise (Corr2Cause) or story (CLadder) was absent from the probe's training fold. `tests/test_data.py::test_grouped_folds_have_no_group_leakage` enforces it.

**5. Your probe gets 80–87% on Corr2Cause. Doesn't that show the model knows the answer?**
No. Corr2Cause labels are a deterministic function of the text, and cheap text statistics predict them well. The exact hypothesis sentence alone (no premise!) gives ~86%, because the dataset orders variables canonically so that some hypotheses are almost always valid and most almost never. Gradient boosting on parsed premise features plus that prior gives ~92%. Both beat the probes (`text_baselines.md`).

**6. Then what would count as evidence that the model "represents but doesn't say"?**
Two criteria are reported. The one fixed before any probe was run (D11): on items the model gets wrong, the gold probe beats 50%, TF-IDF and the template prior. The stricter one adopted after the independent audit (D19): the probe must beat the *strongest* text-only baseline on all report-half items and inside both model-answer strata. Outcomes are in `audited_verdict.md`. No model × dataset cell meets the audited criterion.

**7. Why was the pre-registered criterion not good enough?**
Two reasons, both found by the audit. First, if a model answers "No" to nearly everything, the items it gets wrong are nearly all gold-Yes, and *any* scorer that leans Yes "beats 50%" there (an always-Yes rule scores 90–99.8%). Second, TF-IDF over a 400-token prompt and a six-template prior are weak models of what the text gives away. One cell (Qwen-1.5B, Corr2Cause) passed the pre-registered test for exactly these reasons and fails the audited one.

**8. What does the "within stratum" analysis add?**
It holds the model's own answer fixed (all items where it said No, say) and asks whether the probe can still rank gold-Yes above gold-No. Inside a stratum a probe that only reads the model's yes/no decision is useless (AUROC 0.5), so any skill there is information beyond the binary answer. It is also compared with the model's own graded confidence.

**9. What is the control task and why is your selectivity sometimes negative?**
Each question *type* gets a fixed random label; only memorisation can predict it. On CLadder the probe predicts those random labels better (≈0.79) than the real ones (≈0.56): the residual stream identifies the graph and query type very clearly, while the gold answer is barely decodable. Negative selectivity says "this probe is expressive enough to memorise types, and still cannot find the answer".

**9b. The probe at the end of the premise scores about the same as at the final token on CLadder. Why does that matter?**
At the end of the premise the model has not yet read the question, so nothing there can be "the answer to the question". Equal accuracy (`probe_positions.md`) means the small CLadder signal (0.56–0.60) is a context-only regularity, such as which stories and probability patterns go with Yes. On Corr2Cause the same check comes out as it should: the end-of-premise probe is at chance.

**10. Why include a randomly initialised model?**
To measure what a probe can read from the architecture and the input alone. A random transformer still mixes token embeddings into the final position, so some text statistics survive. The trained model must beat it for the result to be about learned computation.

**11. You changed the regularisation after seeing results. Isn't that p-hacking?**
It is the reason the change is documented so heavily (decision D17). The sensitivity check was planned in advance, with its trigger written down. The new procedure chooses C on the selection half only, is applied to the text baseline and every control as well, and the original fixed-C result is reported next to it (`confound_fixedC_*.md`). The verdicts are shown under both settings.

**12. Why choose the layer on half the data?**
Taking the best of ~25 noisy layer scores and reporting it inflates the number (winner's curse). Selection on one half of the premises/stories and reporting on the other removes that bias at the cost of wider intervals.

**13. What does the logit lens show that probes do not?**
The lens uses the model's *own* readout, so it shows what the model is on course to say at each depth. A probe uses a readout we trained, so it shows what is *available*. A gap between them (probe AUROC far above lens AUROC at the same layer) is exactly "available but not on the path to the output".

**14. Why is few-shot prompting not simply "better"?**
Four demonstrations changed the answer bias more than the skill: see `behaviour_few_shot.md` (for the smallest model the fraction of "Yes" answers on Corr2Cause moved from 0% to 81%, and accuracy fell because the labels are mostly "No"). Macro-F1 and AUROC are the fairer readings.

**15. What does the variable-renaming check tell you?**
Renaming A,B,C to Z,Y,X leaves the logic untouched. A model that reasons should not care. Prediction flips and the change in macro-F1 are in `robustness_renaming.md`; the original paper used the same idea to show that fine-tuned models had learned surface patterns.

**16. The models score 50%. Are they guessing, or biased?**
Both, in different places, and the difference matters. `report/tables/answer_bias.md` gives, for every model and prompt, the Yes-rate, the AUROC of the Yes−No score (which a constant lean cannot move) and the accuracy after moving the threshold to its best position on other items. Qwen2.5-0.5B says "No" to nearly everything, yet on the original Corr2Cause items its score ranks valid above invalid slightly better than chance (AUROC about 0.59): biased, not empty. On CLadder all three models are at 0.50 by AUROC too: empty. And the faint Corr2Cause signal disappears when the variables are renamed or the debiased pool is used, so it was the same surface cue the probes were reading.

**17. How can the wording of the hypothesis predict the answer without the premise?**
Because of how the benchmark is built. Graphs are generated with variables in causal order, so a cause always has an earlier letter than its effect. "A directly causes D" can be valid; "D directly causes A" never is. Of 120 distinct hypothesis strings, 72 are never valid in the whole training split. A classifier that only memorises each string's track record ranks test items with AUROC 0.90. The construction is described in the benchmark paper (Sec. 3.3 and Appendix D); the consequence is not, and no earlier report of a hypothesis-only baseline was found (`notes/literature_check.md`).

**18. Is that a flaw in the benchmark?**
That is not the claim. The benchmark was designed to test models zero-shot, and a model that has never seen the label statistics cannot exploit them. The finding matters for anything *trained* on in-distribution items: fine-tuned models, few-shot prompts, and probes. The original paper itself showed fine-tuned models failing when variables are renamed. What is added here is a measurement of how far a hypothesis-only classifier gets, and the consequence for probing.

**19. Why did your first debiased pool put a hypothesis-only classifier *below* chance?**
Because the pool was balanced as a whole (25 valid, 25 invalid items per hypothesis string) but then cut into folds by premise. If the test fold happens to hold 7 valid and 3 invalid copies of a string, the training folds hold 18 valid and 22 invalid, so the training-fold prior says "probably invalid" exactly where the test fold is mostly valid. Exact balance in the whole turns chance imbalance in one part into the opposite imbalance in the rest. It is the same arithmetic as the leave-one-out leak in the primer. The fix was to make every fold balanced, by building the pool from blocks that are each balanced and using whole blocks as folds.

**20. In the debiased pool the parsed-feature baseline still gets 81%. Isn't the pool still biased?**
No single input is: hypothesis-only and premise-only classifiers are at exactly 50%. What the parsed-feature classifier uses is the *combination*: it looks up, in the premise, what is said about the two variables named in the hypothesis (are they correlated, are they ever made independent, by how many conditioning sets, do they share neighbours). That is the first step of the textbook algorithm for this task. It is deliberately kept as a baseline: if a probe cannot beat it, the activations expose no more about the answer than a hand-written parser that stops after step one. A probe that only ties it would still be interesting, and is reported against the weaker baselines as well.

**21. Gemma's probes come from activations cached on the laptop, its steering from activations recomputed on Kaggle. Is that a problem?**
It would be if the two differed materially. Both use the same weights in half precision; the laptop keeps Gemma's embedding matrix on disk and runs on Apple's GPU backend, Kaggle runs everything on an NVIDIA GPU. `results/cache_agreement_gemma2b.json` (written by the Kaggle run) compares the answer scores item by item with reference values committed from the laptop cache: maximum and mean absolute difference, correlation, and the share of items with the same Yes/No answer. Steering directions are fitted and applied on the Kaggle activations, so each experiment is internally consistent; only the *choice of layer* is carried over from the laptop analysis.

**22. You pre-registered a criterion, it passed, and you say the claim is false. Which is it?**
The criterion passed as coded and the claim is not supported; both statements are in the report. Pre-registration protects against moving the goalposts towards a result. Here the goalposts were found to be in the wrong place, in the direction that made the claim too easy: the baselines named in the criterion were weak. Tightening a test after seeing that it can be passed by a shortcut is legitimate, provided the original analysis is still shown and the change is labelled post hoc, which it is (decisions D18, D19, D21).

**23. On the debiased pool the probe scores 0.75 and you still say the criterion is "not met". Isn't that hiding a positive result?**
No: both are stated, because they answer different questions (primer: confound controls versus reference baselines). Against every shallow control the 1.5B model's probe is clearly real (0.75 against at most 0.58), and that is reported as a finding. The criterion for the headline claim "represents the answer but doesn't say it" asks for more: that the activations hold more about the answer than can be read off the text. A parser gets 0.82, and adding the probe to the parser's features gains nothing measurable. So the supported statement is the modest one: the model carries part of the premise's structure that it does not use.

**24. Why believe the model computes adjacency rather than the probe computing it?**
A linear probe cannot bind "which two variables does the hypothesis name" to "what does the premise say about that pair" by itself; that needs information from different places in the prompt to have been brought together in one vector. The control is the randomly initialised copy of the same network: there, adjacency of the queried pair is decodable at AUROC 0.62, against 0.98 in the trained 1.5B model, while prompt-level counts are decodable perfectly from both. So the binding is something training put there. What the probe does not show is whether anything downstream reads it.

**25. The probe's signal does not drop inside adjacency strata. Doesn't that contradict "it reads adjacency"?**
It shows that adjacency *alone* is not the signal, which is expected: in this pool adjacency by itself predicts validity at 0.54. Validity depends on combinations (which relation is asked, whether the pair is adjacent, whether common neighbours sit inside separating sets). The probe tracks such a combination: inside the joint strata of three properties its AUROC falls from 0.83 to 0.77, and with all parsed features in the model the probe adds nothing.
<!-- /QA-TIER1 -->

<!-- QA-TIER2 -->
### Tier 2 — steering, patching, cross-model

**26. What exactly do you add to the model when you "steer", and how big is it?**
A vector, at one layer, at the last token of the prompt, before the remaining layers run. Its length is a multiple of s, the typical distance of an activation from the average activation at that layer. That sounds like "one typical deviation", and I first described it that way. It is not: s adds up the spread over about a thousand dimensions, so along one particular direction the natural spread is a small fraction of s. Along a probe's weight direction one unit of s is 60 to 300 natural standard deviations. The report therefore gives every effect twice: per unit of s and per natural standard deviation.

**27. Why is the slope better than counting flipped answers?**
A flip happens when the push is larger than the margin by which the model preferred its answer. A model that says "No" with a small margin flips easily; one with a large margin does not. So flip rates mix up the strength of the direction with the model's confidence, and for a model that says "No" to everything only one sign of push can flip anything. The slope (change in the Yes−No logit difference per unit pushed, measured with both signs) describes the direction alone.

**28. Your first conclusion was that the gold-probe direction is "inert". What changed?**
Three things, all from the second audit. In the 1.5B model the direction has a small, perfectly consistent effect of the *wrong* sign, which a table of one-sided flip rates hid. Three random directions gave slopes differing 17-fold, so "no more than random" had no basis. And the natural comparison for a probe's weight vector is not a random direction but a probe's weight vector fitted to shuffled labels. Against twenty of those, the gold direction sits at the 20th to 85th percentile in the four cases tested. So the statement is now "no more effect than a label-free direction of the same kind", which is a statement about that direction, not about the information.

**29. If the probe direction does nothing, isn't the information unused?**
That does not follow. A probe's weights say which combination of activations to *read*; they point into quiet directions where a small difference is reliable. How the activations actually *differ* between valid and invalid items is another vector, the difference of class means, which lies in loud directions. The two are linked by the covariance of the activations and can be nearly perpendicular. Later layers respond to the second, not the first. And the second turned out not to be specific to validity either: differences of means for shuffled labels move the answer about as much.

**30. So what is the evidence that the model does not use what the probe reads?**
The plainest evidence is behavioural: on the debiased pool the probe decodes validity at 0.75 from the 1.5B model while that model's own answers rank valid above invalid items no better than chance. Steering adds that no gold-specific direction was found whose movement changes the answer more than matched controls. What is not shown: that no later computation reads the information in some non-linear way, or at another position.

**31. Why is the model-answer direction called a positive control and not a finding?**
It is the weight vector of a probe trained to predict the sign of the very quantity steering measures. Of course moving along it changes that quantity. It is there to show that the method can detect an effect when one exists, and how large an effect looks.

**32. What does activation patching show here, and what does it not?**
It shows the route by which a change in the prompt reaches the place where the answer is read out: at the changed word in early layers, at the final token from about 60% of the depth. It does not show where the model "decides" the causal question, because the models do not treat the valid and the invalid prompt of a pair differently in the right direction; any difference between the prompts is tracked equally. A control with pairs that differ in a variable but share the gold label is what is missing.

**33. The random-donor row in the patching figure looks like the real one. Is that a problem?**
It means that row is not a control. Patching in activations from an unrelated prompt of the same length should, one would think, do nothing. But few pairs have such a donor (16 of 60, 6 of 40), in half of them the donor has the same word at the changed position, and the recovery statistic is not zero on average even for random patches. The row is kept in the figure and labelled as uninformative.

**34. Can attribution patching be trusted?**
For the components with the largest effects, yes: its estimates match real patching closely. For small components the two can disagree even in sign, and checking ten random components out of several hundred cannot show that none of the unchecked ones matters. So it supports "these are among the largest contributors", not "this is the complete circuit".

**35. Do the results scale with model size?**
There are too few models to say (three local ones from two families, plus one larger one from the remote run). On the debiased pool the 1.5B model's probe is clearly above the 0.5B model's (0.75 against 0.63); whether that continues is what the larger model is for.
<!-- /QA-TIER2 -->

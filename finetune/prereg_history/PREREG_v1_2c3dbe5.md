# Pre-registration: the fine-tuning study (D3)

Written Monday 16 November 2026, before any fine-tuned model exists. The commit that adds this file precedes every
training run; later changes are appended under "Amendments" with date and reason, never edited in place.
Power numbers: `finetune/scripts/power.py` → `finetune/power/power.json`. Dataset counts quoted here were measured
today on the released v2.0 data (`data/raw/corr2cause`, revision in `report/benchmark_text_baselines.md`).

## 0. Question and what would count as an answer

When Qwen2.5-1.5B-Instruct is LoRA fine-tuned on Corr2Cause until it does the task in distribution, has it learned the
causal structure or the shortcuts measured in v1.0 (hypothesis wording, prompt length within hypothesis type, lookup of
the premise statements about the queried pair)?

"Learned the causal structure" is given an operational meaning here, because "above chance on the debiased pool" is
not one: on that pool a lookup of the statements naming the queried pair already scores 0.765, a hand-written parse
0.816 and the tuned text stack 0.887 (v1.0, claim 2). The ladder, weakest reading first:

| Level | Evidence required | Experiment |
|---|---|---|
| L0 shortcut only | in distribution high; on the length-balanced pool the within-string AUROC with length removed is not above 0.5; condition C reproduces most of A in distribution | E1, E2, E3 |
| L1 pairwise lookup | above 0.5 on the length-balanced pool, but not above 0.5 inside cells of hypothesis string × adjacency of the queried pair, and not above the hand parse | E5 |
| L2 beyond lookup | above 0.5 inside those cells, and survives renaming and paraphrase on held-out premises | E4, E5 |
| L3 represented beyond the text | a probe for the gold label beats every audited text reference, including ones trained on the model's own training set | E6 |
| L4 used | steering along the gold direction beats 500 matched controls in a held-out test; patching localises it | E7, E8 |

Each level is claimed only if its test passes as registered below. No level is inferred from a higher or lower one.

## 1. Data (fixed before training; built on the laptop)

**Premises.** Official train has 205,734 items on 2,338 premises; test 1,162 items on 21; dev 1,076 on 22. Premise
overlap: train–test 0, train–dev 2, dev–test 3. Parsing every premise into its set of correlation and independence
statements and canonicalising over all relabellings of the variables gives 2,338 distinct classes for 2,338 train
premises, none shared with test: **every premise is its own graph class**, so held out by premise is held out by
graph. The build recomputes this and fails if it stops being true.

- EVAL premises: the 1,200 premises of the v1.0 debiased pool, plus the 21 official test premises.
- TRAIN premises: train premises that are not EVAL and not in dev: 1,137 (98,964 official items, 12,666 valid among
  the 42,949 items of the 48 hypothesis strings that occur with both labels).
- Dev is not used for anything.
- Proof of separation, written to `finetune/data/overlap_report.json` and enforced by a test: for every (training set,
  evaluation set) pair, the count of shared premise texts, shared premise hashes after variable renaming, and shared
  graph classes, each required to be 0.

**Training conditions.** All three have exactly N items and the same number of optimisation steps. N is not chosen by
me: it is the number of items condition B yields (below), rounded down to a multiple of 100.

- **A, official.** A uniform random sample of N official items of the TRAIN premises, as released (all 120 hypothesis
  strings, about 18% valid). "Official train as is" cannot be taken literally: see Deviations.
- **B, debiased.** Built by the balancing function (`finetune/ftstudy/balance.py`, the owner's implementation; a
  reference implementation is kept in `balance_reference.py`). Properties, by construction and tested: (i) inside
  every hypothesis string, valid = invalid; (ii) inside every premise, valid = invalid; (iii) inside every hypothesis
  type × prompt-length bin, valid = invalid, with bins narrow enough that (iv) holds. (iv) Verification on the built
  set, premise-grouped 5-fold: AUROC for the label of a hypothesis-only classifier, a length-only classifier (token
  count of the full prompt under the Qwen2.5 tokenizer; within type and within string), a premise-only classifier, and
  a classifier on hypothesis string × length jointly. Requirement for each: |AUROC − 0.5| ≤ 0.02 and the premise
  bootstrap interval contains 0.5. If a requirement fails the set is rebuilt with narrower bins; no model is trained
  on a set that fails.
- **C, shortcut control.** The items of A with each premise replaced by the premise of another TRAIN item that has the
  same number of variables and the same prompt-length bin, drawn at random (seeded), never the item's own premise.
  Hypothesis, label and (to bin width) length are those of A, so every hypothesis-only and length-only statistic of C
  equals A's; the premise content is unrelated to the label. C measures how far hypothesis wording plus length carry a
  fine-tuned model. It cannot see the lookup shortcut; that one is measured by text baselines (E3).

Measured today with premise word count as a stand-in for token count: inside a hypothesis type, length alone predicts
the label in the TRAIN items of the 48 two-label strings with AUROC 0.26 to 0.30 (shorter premise, more often valid).
That is a far larger leak than the 0.41 to 0.58 of the v1.0 pool, and it is what condition A is free to learn.

**Evaluation sets.** Every model is scored on exactly the same items (ids and sha256 of each set are committed).

| Set | Items | Premises | Purpose |
|---|---|---|---|
| ID-heldout | 6,000 official items sampled uniformly from the EVAL pool premises (natural label rate, all strings) | 1,200 | "does the task in distribution" on unseen premises |
| official test | 1,162 | 21 | continuity with the paper; descriptive |
| POOL | the v1.0 debiased pool, unchanged | 1,200 | continuity with v1.0 |
| POOL-LB | rebuilt from all official items of the 1,200 pool premises with the same balancing function as B | 1,200 at most | the primary set: wording, premise and length carry nothing |
| renamed | ID-heldout, POOL, POOL-LB with A…F → Z…U (v1.0 `rename_variables`) | same | variable refactorisation |
| paraphrased | ID-heldout, POOL, POOL-LB with the authors' paraphrase templates, taken from their released file | same | paraphrase |
| authors' perturbed | `perturbation_by_refactorization_test.csv`, `perturbation_by_paraphrasing_test.csv` (2,246 items each; valid = `label == 1`) | 21 | replication of Table 5b; descriptive |

The authors' perturbed sets are released (Hugging Face `causal-nlp/corr2cause`) and rest on the 21 test premises.
The refactorisation file contains all 1,162 renamed v2.0 test items with identical labels (checked today). Our
paraphrased sets apply the authors' own four hypothesis templates to our held-out premises; the mapping from relation
to template is read from their file and committed.

## 2. Training (fixed before the first run)

Qwen2.5-1.5B-Instruct; LoRA with pinned peft 0.21.2, rank 16, alpha 32, dropout 0.05, on all attention and MLP
projections; AdamW, learning rate 2e-4, linear warm-up 30 steps then linear decay; batch 16; 3 epochs over N; loss:
cross-entropy on the answer token (Yes/No) at the final prompt position over the full vocabulary; prompt format
identical to v1.0 (`causal_probe.prompts`). No class weighting in any condition. Seeds 0, 1, 2 change LoRA
initialisation and data order; the training items of a condition are the same for every seed. Nothing is tuned on
any evaluation set; there is no early stopping; the last checkpoint is the model.

A time test (one seed of A, 300 steps, plus the evaluation pass on 500 items) runs first. Only seconds per step,
memory and "the loss decreased" are read from it; its adapter is discarded.

**Reference model A-full** (one seed, all 98,964 TRAIN items, one epoch). Not part of any A/B/C comparison. It exists
because N is set by B and may be too small for A to do the task in distribution. It is trained only if A at N fails
the gate of E1, and is then the "model that does the task" for E6 to E8, labelled as such.

## 3. Experiments

Unit of inference everywhere: the premise (cluster bootstrap over premises, 2,000 draws, paired where two models
score the same items). Seeds are a replication criterion, not a sample: with 3 seeds against 3 an exact permutation
test cannot go below p = 0.10, so no seed-level test is run. A difference between conditions is **claimed** only if
(a) the 95% premise-bootstrap interval of the difference of seed-averaged statistics excludes 0, and (b) all nine
seed pairs have the same sign. If (a) holds and (b) fails, the page says "not consistent across seeds".

Score = answer logit difference (Yes − No) at the final position. S(model, set) = mean over hypothesis strings of the
within-string AUROC of the score for the label, **with prompt length removed** (residual of the score on a quadratic
in log token count, fitted within string; the owner's implementation, checked against the reference).

**E1. In distribution (gate, not a test).** A reaches pooled AUROC ≥ 0.90 and valid-class F1 ≥ 0.70 on ID-heldout,
mean of three seeds, with accuracy, macro-F1 and intervals reported. *Prediction:* passes (the paper's Table 5a
reports F1 of about 92 for fine-tuned LLaMA-7B, trained on far more data; N is the risk). *If it fails:* A-full is trained; if A-full
fails too, the page reports that a 1.5B LoRA did not learn the task at this budget and E6 to E8 are not run.
*Power:* not a test.

**E2. A against B where the shortcuts carry nothing (primary).** Statistic: S(B, POOL-LB) − S(A, POOL-LB).
*Hypothesis:* A's in-distribution skill rests largely on wording and length, so on POOL-LB it is near chance, and B,
which could not use them, is higher. *Prediction:* S(A) between 0.50 and 0.60; S(B) ≥ S(A) + 0.10.
*Outcomes:* difference claimed positive → training on debiased data buys validity-tracking that official data does
not. Difference not claimed and both intervals inside 0.5 ± 0.05 → neither learned anything the shortcuts do not
carry (L0). Difference not claimed and both clearly above 0.5 → official data already teaches it; the shortcuts were
not the whole story. Negative → B's smaller effective variety hurt; reported as such.
*Test:* rule (a)+(b) above, two-sided, α = 0.05. *Power* (48 strings; 12 to 25 pairs per string; item noise
correlated 0.5 between conditions): with no seed variance, ≥ 0.99 for a difference of 0.05. With a between-seed SD of
0.03 in AUROC (unknown until the runs exist; assumed): 0.41 to 0.52 for 0.05, 0.96 to 0.98 for 0.10, 1.0 for 0.15.
**Smallest effect this design is registered to detect: 0.10.** A true difference under 0.10 may be missed, and the
page will say so rather than "no difference". Pre-stated rescue: if the observed seed SD exceeds 0.02 and the
observed difference lies between 0.04 and 0.10, two more seeds of A and B are trained from the quota reserved for
the scaling check, and the test is run once on five seeds; this option is used at most once.

**E3. How much of A is shortcut.** (i) S and pooled AUROC of C on ID-heldout against A: shortcut share =
(AUROC_C − 0.5) / (AUROC_A − 0.5), with interval. (ii) Sanity: S(C, POOL-LB) interval contains 0.5; if it does not,
POOL-LB or C is leaking and nothing downstream is read until that is explained. (iii) Text references on the same
items, trained on A's own training set: hypothesis-only, length-only, lookup, hand parse, tuned stack.
*Prediction:* share ≥ 0.7 in distribution (the hypothesis alone reaches 0.895 on the official test split).
*Outcomes:* share near 1 → A's in-distribution score is reproducible without reading the premise. Share well below 1
→ A uses premise content; whether that content is the lookup is E5's question.
*Test:* premise-bootstrap interval of AUROC_A − AUROC_C on ID-heldout (6,000 items, 1,200 premises), claimed by
(a)+(b). *Power:* pooled AUROC on 6,000 items has a standard error under 0.01; a difference of 0.03 is detected with
power above 0.9 even with seed SD 0.02.

**E4. Robustness, extending Table 5b.** Drop = statistic on the original items − statistic on the perturbed version
of the same items, for renaming and for paraphrase, on ID-heldout (pooled AUROC and F1) and POOL-LB (S).
*Hypothesis:* the paper's drops are what a wording and name shortcut predicts; a model trained without those
shortcuts drops less. *Prediction:* drop(A) > drop(B) for renaming on ID-heldout by at least 0.05 AUROC.
*Outcomes:* A drops, B does not → the fragility the paper reported is a property of the training distribution, not of
fine-tuning. Both drop → the fragility survives debiasing (names or templates are still read literally). Neither
drops → the paper's result does not reproduce in this model family; reported with the replication table.
*Test:* drop(A) − drop(B), (a)+(b). *Power:* as E3 on ID-heldout (standard error of a paired drop under 0.01).
**On the authors' own perturbed sets (21 premises) no A-against-B test is run:** a within-model drop of 0.05 has
power 0.99 to 1.0 there, but a difference of 0.03 between two models has power 0.61 to 0.82 before seed
variance, which falls below 0.6 with it. Those sets enter as a descriptive replication table next to Table 5b.

**E5. Beyond the lookup.** On POOL-LB, for A, B (seed-averaged): (i) S restricted to cells of hypothesis string ×
adjacency of the queried pair (premise states the pair correlated / independent), cells with both labels only;
(ii) pooled AUROC minus the hand parse's and minus the lookup's on the same items.
*Hypothesis:* a model that learned the structure tracks validity where the pairwise statements do not decide it.
*Prediction:* B above 0.5 inside cells; A not.
*Outcomes:* above 0.5 inside cells and above the hand parse → L2. At the lookup's level and 0.5 inside cells → L1:
the debiased model learned the lookup, which is the next shortcut in line.
*Test:* (i) permutation of labels within cell, 5,000 permutations, one-sided, α = 0.05/2 (two conditions), on the
seed-averaged score, plus (b) across seeds; (ii) premise-bootstrap interval of the difference.
*Power:* for one model against 0.5 with 12 pairs per string in two-class cells: 0.52 for a true 0.55, 0.99 for 0.60;
with 6 pairs: 0.29 and 0.79. **Registered for effects of 0.60 or more inside cells.** If the build leaves fewer than 6
pairs per string in two-class cells, E5(i) is dropped and only (ii) is reported.

**E6. Representations.** Base, A, B (seed 0 for layer selection and the headline; seeds 1 and 2 as replication):
the v1.0 probe ladder on POOL-LB with every audited text reference (tuned stack, lookup, hand parse, each with
length), plus the same references fitted on the condition's training set. Layer and regularisation chosen on the
selection half of the blocks; numbers from the report half.
*Hypothesis:* in v1.0 no probe beat the tuned text stack in any model. If fine-tuning builds a representation of
validity, the probe now does. *Prediction:* for the model that passes E5, probe − tuned stack ≥ 3 points.
*Outcomes:* probe above all text references in a fine-tuned model and not in base → fine-tuning made the label
decodable beyond the text (L3), at the reported layer. Probe level with the stack → the model's behaviour needs no
representation beyond what a text classifier extracts.
*Test:* premise-bootstrap interval of probe accuracy − best text reference on the report half, α = 0.05/3 (three
models). *Power* (standard error 0.45 points, from v1.0's report-half intervals): 0.43 for 1 point, 0.98 for 2, 1.0
for 3. **Registered for gains of 2 points or more.**

**E7. Steering (held-out, one test per model, at most two models).** Run only on a model with S(POOL-LB) ≥ 0.65 and
lower interval bound above 0.55. Site: last token of the hypothesis (not the answer position, where v1.0 showed a
gold direction reads back the answer), layer chosen on the selection half as the best gold probe layer. Direction:
gold class-mean difference fitted on the selection half. Statistic: slope of the answer logit difference per unit
step on the report half. Controls: 500 class-mean differences of labellings that share both balances of the gold
label (within string and within premise), as in v1.0's held-out test. p = (1 + number of controls ≥ gold) / 501,
one-sided in the positive direction; threshold 0.05 for one model, 0.025 each for two.
*Hypothesis:* in a model that does the task the gold direction is used upstream of the answer.
*Outcomes:* p under threshold → L4 for that site. p over → "not shown at this site", with the detectable size stated.
*Power:* depends on r = gold slope in control standard deviations, which cannot be known before the model exists.
From v1.0's control spread: r = 2.0 gives 0.86 at α = 0.05 and 0.51 at 0.025; r = 2.5 gives 1.0 and 0.93. v1.0
observed r = 1.3 in models that could not do the task (power under 0.3). **Rule:** before the run, r is estimated on
the selection half only, with 100 controls, from activations already exported by the training notebooks (no new GPU
time). If the estimate is under 2.0 (one model) or 2.3 (two models), the test is dropped and reported as "not run:
power below 0.6", and E8 alone is run.

**E8. Patching (localisation; descriptive with one test).** On the same model: activation patching across layers
and the three positions on label-changing against label-preserving pairs from POOL-LB (same premise, hypotheses of
one type), as v1.0 `08_patching.py`. *Test:* at the layer × position cell chosen on half the pairs, the
label-changing effect minus the label-preserving effect on the other half; interval excludes 0. *Power:* computed
from v1.0's all-pairs run (`patching_*_n240`) before launch and appended here; dropped if under 0.6 for an effect of
one third of the model's mean |logit difference|.

**Scaling check (only if quota remains after E1 to E8).** Qwen2.5-0.5B and 3B, conditions A and B, one seed each:
E1 and E2 statistics, descriptive only. One seed per size supports no test and none is run.

## 4. Order, stopping, audits

Order: E1 → E2, E3, E4, E5 (one behaviour audit) → E6 (audit) → E7, E8 (audit). Each audit is done by a fresh
reviewer given code and result files, not conclusions. Only audited results enter `RESULTS_FT.md`. If Sunday 22
November 18:00 arrives first, the page says "in progress" and reports the audited part.

## 5. Deviations from the brief, decided before any run

1. **A is not the official train split "as is".** The evaluation pool is cut from train, so A is sampled from the
   1,137 train premises outside it; and "same item count" makes A's size N, set by B. A-full covers the risk.
2. **B is also balanced within premise** (not asked for). Without it the premise alone predicted the label with
   AUROC 0.72 in v1.0's first design (decision D21).
3. **Length is balanced in bins, not exactly.** Premise lengths range from 60 to 1,358 words; exact matching within
   string would keep 9,394 of 42,949 candidate items and 184 of the pool's 2,400. The verification in (iv) is the
   criterion, not the bin width.
4. **No test on 21 premises**, and no seed-level test (see E4 and Section 3).
5. **The analysis of "what it learned" has a rung for the lookup** (E5), which the brief lists as a shortcut but does
   not test separately.

## Amendments

(none)

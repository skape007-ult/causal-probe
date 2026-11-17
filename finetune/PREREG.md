# Pre-registration: the fine-tuning study (D3) — version 2

Version 2, 17 November 2026 (UTC), before any fine-tuned model exists. Version 1 (commit `2c3dbe5`, kept unchanged
in `finetune/prereg_history/PREREG_v1_2c3dbe5.md`) was reviewed independently before any run
(`finetune/notes/audit_prereg_raw.md`): eight blocking findings. It was replaced, not patched; Section 6 answers
every finding. A second independent review (`finetune/notes/audit_prereg2_raw.md`) passed version 2 on five
conditions, all edits to this text; they are made here and listed in Section 7. From the commit that carries them
onward this file is changed only by dated entries under "Amendments".

Every count below is read from `finetune/itemsets/build_report.json` (built by `finetune/scripts/build_data.py`
with the reference balancing function) and `finetune/itemsets/references.json` (`finetune/scripts/references.py`);
every size and power figure from `finetune/power/power.json`
(`finetune/scripts/power.py`, which simulates the function the analysis will call, `finetune/ftstudy/seedtest.py`).

## 0. Question, and what would count as an answer

When Qwen2.5-1.5B-Instruct is LoRA fine-tuned on Corr2Cause, what has it learned: the causal structure, or what can
be had without it? "Without it" has four parts here, each with its own reference:

| What a model could be doing | Reference that does only that | Its score |
|---|---|---|
| reading the hypothesis wording | P(valid | hypothesis string) from the training set | pooled AUROC 0.885 on ID (fitted on A); with prompt length added, 0.922 |
| reading prompt length | token count within string | raw within-string AUROC 0.758 on ID48; removed from S by construction |
| reading surface counts of the premise | a per-string classifier on four premise counts (Section 1) | S on POOL-LB fitted on B 0.588 (logistic) and 0.666 (boosting); on A48 0.523 and 0.584; on A 0.520 and 0.566 |
| copying from similar training premises | the memorisation reference: the best of a fixed family of fourteen copy rules (Section 1) | S on POOL-LB 0.898 from B's training set, 0.757 from A48's, 0.703 from A's; on ID48 0.763 from A's |
| looking up the statements about the queried pair | v1.0 lookup, hand parse, tuned text stack, fitted on the condition's training set | computed with the behaviour analysis, before model scores are opened |

The ladder. Every rung is defined by registered tests of Section 4, run per model family (A and B separately), and
claimed on the Holm-adjusted p of Section 3. "Not shown" is reported with its interval and never read as "absent".

| Rung | Definition | Tests |
|---|---|---|
| L0 | E5a not shown: inside cells of hypothesis string × adjacency, S is not shown above 0.5 | E5a |
| L1 | E5a claimed above 0.5, E5b not: validity is tracked beyond adjacency, not shown beyond what copying from training premises gives | E5a, E5b |
| L2 | E5a and E5b both claimed (S above the memorisation reference of the model's own training set) | E5a, E5b |
| L3 | L2, and E5c claimed (S above the best fitted text reference) | E5c |
| L4 | an intervention shows the label direction is used | E7, E8: not registered yet |

E5b claimed with E5a not shown has no rung: it is reported as found and no rung above L0 is claimed.

## 1. Data (built and fixed; `finetune/itemsets/`)

**Premises.** Official train: 205,734 items on 2,338 premises; test 1,162 on 21; dev 1,076 on 22 (2 of them also
in train, 3 in test; dev is not used; one of the two shared premises is a POOL premise and stays one). The 2,358
premises of TRAIN, ID, POOL and test are 2,358 distinct graph
classes (statement set canonicalised over every relabelling of the variables). That is necessary for "held out by
graph" and **not sufficient**: a held-out premise differs from its nearest training premise by a median of 4
statements (quartiles 3 and 8) out of about 33. Hence the memorisation reference.

Three groups of premises:

- **POOL premises (1,200):** those of the v1.0 debiased pool. They were *selected on the label* by the v1.0 builder
  (premises holding a valid item of a two-label string): 23.8% of their official items are valid, 54.8% inside the
  48 two-label strings. They are not a random sample of train.
- Of the remaining 1,137 premises, a seeded random 187 are the **ID premises** (in-distribution evaluation) and
  the other **950 are TRAIN**. Valid rate of official items: TRAIN 12.5%, ID 14.5% (28.7% and 33.5% inside the 48
  strings). "In distribution" in this study means the ID premises. The POOL premises are a shifted population
  (about twice the valid rate), and every statement about them says so.

**Training conditions**, N = 7,184 items each (the size B yields; fixed):

| | Items | Seeds | Role |
|---|---|---|---|
| A | uniform sample of the 82,458 official TRAIN items: 120 strings, 12.8% valid; 3,126 items in the 48 two-label strings, 919 of them valid | 5 | "official data" |
| A48 | uniform sample of the TRAIN items of the 48 two-label strings, natural labels (28.8% valid), no balancing | 3 | exposure control for B: same strings, same count, only the balancing differs |
| B | balanced (below): 48 strings, 707 premises, exactly half valid | 5 | "debiased data" |
| C | A's items, each on the premise of another TRAIN premise | 3 | how far everything except the item's own premise carries A |
| Bswap | B's items, each on the premise of another TRAIN premise | 3 | negative control for B and POOL-LB: must score 0.5 |

- **B** is built by the balancing function (`ftstudy/balance.py`; the core runs use the reference implementation
  `balance_reference.py`; the owner's independent implementation must reproduce the stored sets exactly, and the
  page will report it that way) with 4 length bins and at most 8 pairs per premise. By construction, valid =
  invalid inside every hypothesis string, every premise, every type × length bin, and every string × length bin.
- **Premise swap (C, Bswap).** TRAIN premises are stratified by number of variables, then sorted by premise token
  count and cut into consecutive strata of at least 10 premises (93 strata of 10 to 19). The donor is drawn
  uniformly among the *premises* of the item's stratum other than its own. This is independent of B's bins. A
  swapped item keeps its hypothesis and label. In C the kept label is still the true label of (new premise,
  hypothesis) for 85% of the items that exist on the new premise, because most strings have one label and a
  stratum fixes a premise's size: **C is a mostly correct, noisy Corr2Cause in which the item's own premise is not
  available.** What C measures is how far hypothesis wording, length and premise size carry, not "shortcuts" in
  general. In Bswap the kept label is correct for 49.9%: nothing but chance is available.
- **Seeds** change LoRA initialisation and data order. The items of a condition are one draw, the same for every
  seed: "consistent across seeds" does not cover "consistent across draws of the training set" (limitation).

**Shortcut checks on B and POOL-LB** (passed; in the build report). Exact balance holds in-sample. Classifiers
fitted on one set and tested on the other (disjoint premises), AUROC with premise-bootstrap interval, requirement
|AUROC − 0.5| ≤ 0.02 and interval containing 0.5: hypothesis string (label rate per string): 0.500 both ways;
length within string (per-string logistic regression on centred log token count and its square): 0.519 [0.499,
0.538] fitted on B, 0.507 [0.491, 0.521] the other way; string and length jointly: 0.515 and 0.509. A classifier on
the premise alone (logistic regression on its surface counts) scores exactly 0.500, but that is true by
construction of any set that is half valid inside every premise and is not evidence of anything. The same
classifiers fitted on A and tested on ID: 0.885, 0.726 and 0.922 (premise alone 0.676). (Folds inside one exactly
balanced set give below-chance values, 0.37 to 0.43, by the mirror effect of v1.0 decision D21; that method was
mine in version 1 and is not used.)

**What these checks do not cover, measured.** A per-string classifier on four surface counts of the premise
(number of correlation statements, of marginal and of conditional independence statements, log word count; nothing
about the queried pair), fitted on a training set and scored by S on POOL-LB: fitted on B 0.588 (logistic
regression) and 0.666 (gradient boosting); on A48 0.523 and 0.584; on A 0.520 and 0.566; on Bswap 0.525 and 0.501.
So B and POOL-LB share a regularity of hypothesis string × premise counts that A48 has less of, worth 0.06 to 0.08
in S, which is the size of effect E2 is about. It is content of the premise, not wording or length, but it is not
causal structure either. **Bswap controls only what survives the swap: hypothesis string, number of variables and
premise length.** It cannot see this regularity, because the swap destroys the item's premise. The premise-count
classifier is therefore a reference in E5c, and E2's outcome table says what E2 cannot show.

**Evaluation sets** (ids and sha256 committed; every model is scored on every item):

| Set | Items | Premises | Use |
|---|---|---|---|
| ID | 2,000 official items of the ID premises, natural labels (15.3% valid) | 187 | pooled AUROC, F1; E1 |
| ID48 | 2,000 items of the ID premises from the 48 two-label strings, natural labels (34.8% valid; 47 strings with at least 3 items per class) | 187 | S in distribution; E1, E3, E4 |
| POOL-LB | 4,706, built from the POOL premises by the balancing function (4 bins, 2 pairs per premise; at most 4 items per premise; 47 strings with at least 3 items per class, median 29.5 pairs per string; 67% collider and confounder hypotheses, where B is 62% direct-cause: S weights strings equally) | 1,193 | S where wording, the premise alone and length carry nothing; E2, E5 |
| POOL | the v1.0 pool, 2,400 | 1,200 | continuity with v1.0; descriptive |
| official test | 1,162 | 21 | continuity with the paper; descriptive |
| ID renamed, ID paraphrased | ID with the authors' two perturbations | 187 | Table 5b on held-out premises; descriptive |
| ID48 renamed, ID48 renamed keeping order | A…F → Z…U (the authors'; reverses alphabetical order) and A…F → U…Z (order kept) | 187 | E4 |
| POOL-LB renamed | A…F → Z…U | 1,193 | descriptive |
| authors' perturbed sets | distinct inputs of their two files: 1,884 and 1,522 (the files have 2,246 rows each) | 21 | Table 5b; descriptive |

No training set shares a premise text, a premise text with renaming undone, or a graph class with any evaluation
set (60 pairs checked, `overlap_report.json`, enforced by the build). Our renaming and paraphrase reproduce the
authors' files on all 1,162 official test items with identical labels.

Two properties of the authors' perturbations, stated so that no drop is over-read: (i) in this benchmark the
letters carry causal order: a directional hypothesis whose first-named variable is the later letter is never valid
in train. Reversing the alphabet therefore inverts a real regularity of the data as well as changing names.
(ii) The paraphrase gives "parent" and "child" one wording and the two indirect relations one wording (four
templates for six relations), and the paraphrase file is labelled three-way (valid = entailment).

**The memorisation reference** (`datasets.memorisation_family`; values in `references.json`). For a test item, the
candidates are the premises of the given training set that have the same number of variables and carry the item's
hypothesis string. Distance between two premises: the size of the symmetric difference of their statement sets,
counted over all statements or only over statements naming one of the two queried variables. A copy rule scores
the item by the mean label of the string in the k nearest candidates (k = 1, 3, 5, 9) or by the mean label over all
candidates weighted by exp(−distance/τ) (τ = 1, 2, 4): fourteen rules, 0.5 where there is no candidate. The
memorisation reference of a condition on a set is the rule with the largest S there, chosen once on the full set
(a choice in the reference's favour). Computed per condition on that condition's own training set, no model
involved. S on POOL-LB: 0.898 from B, 0.757 from A48, 0.703 from A (the nearest single premise over all
statements, the weakest rule: 0.821, 0.674, 0.631). On ID48: 0.763 from A, 0.807 from A48, 0.708 from B. From
Bswap and C at most 0.52 and 0.55.

## 2. Training (fixed)

Qwen2.5-1.5B-Instruct; LoRA (peft 0.21.2), rank 16, alpha 32, dropout 0.05, all attention and MLP projections;
AdamW, learning rate 2e-4, 30 warm-up steps then linear decay; batch 16; 3 epochs over the 7,184 items (1,347
steps); cross-entropy on the answer token at the final prompt position; prompt format of v1.0. No class weighting,
no early stopping, nothing tuned on any evaluation set; the last checkpoint is the model. 19 fine-tuned models
(5 + 3 + 5 + 3 + 3) and the base model.

A time test comes first: one seed of A for 150 steps, then the evaluation pass on 300 items. Read from it:
seconds per step, memory, "the loss decreased". Its evaluation scores are not opened and its adapter is discarded.
There is no A-full model (version 1 had one; it changed items, epochs and steps at once).

## 3. Statistics

**S(model, set).** The items are first restricted to hypothesis strings with at least 3 valid and 3 invalid items
in that set. Then `within_string_auroc_length_removed` (the owner's part (b); reference in
`within_string_reference.py`; one test fixture for both): inside each string the score is regressed on centred
log token count and its square, without the labels; the AUROC of the residual for the label, ties one half; the
unweighted mean over strings. Score = answer logit difference (Yes − No) at the final position. Token count = the
full templated prompt of the item as evaluated (re-counted for renamed items). S is always computed per model and
then combined; scores are never averaged across seeds.

**The one registered test** (`ftstudy/seedtest.py`), used for every comparison below, two-sided:

    D   = mean over seeds of the statistic in condition X − mean over seeds in condition Y
          (or − the statistic of a fixed reference score; or − a constant; or the mean of within-seed differences)
    SE² = s_X²/n_X + s_Y²/n_Y + v_premise
    t   = D / SE, Student's t, Welch–Satterthwaite degrees of freedom of the seed terms

s² is the variance of the per-seed statistics; v_premise is the variance of D over 2,000 premise resamples with
the trained models fixed (`ftstudy/stat.py`, `boot_D`: every model and every reference is rescored on the same
resample; which strings and cells are eligible, and the length residual, are recomputed on each resample). There is
no second criterion.

**Size of exactly this test** (200,000 replications per setting). For normally distributed per-seed values, seed
SD 0.005 to 0.04 equal in both conditions, premise SE 0 to 0.03: at most 0.051, for 5 against 5, 5 against 3, and
one condition with 5 or 3 seeds; conservative when the premise term is large. Exceptions, measured (the second
review's simulation and mine): with 5 against 3 seeds, unequal seed SDs (0.01 in the five, 0.04 in the three) and
no premise term: 0.065. With skewed or bimodal per-seed values and a dominant seed term (one seed that fails to
train): 0.12 to 0.32. Hence the **rule for an outlying seed**, fixed now: every per-seed value is reported; a seed
whose statistic lies more than 0.10 from the median of its condition is outlying (`seedtest.outlying_seeds`); a
test with an outlying seed is reported with and without that seed, and is claimed only if both versions are.

**Multiplicity.** The family is eleven tests: E2a, E2b, E3, E4a, E4b, and E5a, E5b, E5c each for A and for B. Every
test is reported with its unadjusted p and its Holm-adjusted p over the eleven (`seedtest.holm`); "claimed" means
Holm-adjusted p < 0.05. E2c is a control and E1 a gate; neither is in the family.

**Power** is therefore computed at the strictest level a member of the family can face, α = 0.05/11, for seed SD
0.01, 0.02 and 0.03 (unknown until the models exist). The premise SE comes from the built sets (scores simulated on
the real items, strings, cells and premise clusters, two models sharing half of their item noise): 0.011 for a
difference of two models' S on POOL-LB, 0.013 on ID48. With independent item noise, or a real copy score as the
reference, the premise SE is larger and power lower (the second review's figures at unadjusted α: 0.71 → 0.57 for
E2a at 0.05). **Rule: a test is registered for the smallest effect on the grid 0.05, 0.07, 0.10, 0.15, 0.20 at
which power at α = 0.05/11 and seed SD 0.02 is at least 0.6.** If the observed seed SD exceeds 0.03 the page says
the test was underpowered for its registered effect, not "no difference".

| Test | Registered effect | Power there at seed SD 0.01 / 0.02 / 0.03 |
|---|---|---|
| E2a, S(B) − S(A), 5 against 5 | 0.10 | 1.00 / 0.92 / 0.63 |
| E2b, S(B) − S(A48), 5 against 3 | 0.15 | 0.93 / 0.79 / 0.62 |
| E3, S(A) − S(C) on ID48, 5 against 3 | 0.15 | 0.89 / 0.76 / 0.59 |
| E4a, E4b, within-seed difference, 5 seeds | 0.10 | 0.90 / 0.68 / 0.42 |
| E5a, S inside cells − 0.5, 5 seeds | 0.10 | 0.99 / 0.87 / 0.58 |
| E5b, E5c, S(model) − S(reference), 5 seeds | 0.10 | 0.99 / 0.85 / 0.55 |

At the unadjusted level the same tests reach 0.6 at 0.05 to 0.10 (`power.json`, `power`); those figures are not the
registered ones.

## 4. Experiments

**E1. Does A do the task in distribution? (gate, no test)** Passed if, for A (5 seeds): (i) mean S on ID48 ≥ 0.80,
and (ii) mean pooled AUROC on ID ≥ 0.922, the score of the string prior with prompt length. Reported beside it:
valid-class F1 on ID at the threshold "logit difference > 0" and at the threshold that maximises F1 on the training
set, accuracy, macro-F1, and A's memorisation reference on ID48 (S = 0.763).
*Why these numbers:* length alone gives a raw within-string AUROC of 0.758 on ID48, which is why S removes it; a
per-string classifier on premise counts scores S = 0.59 there. The string prior alone gives pooled AUROC 0.885 on
ID and F1 0.56 at its best threshold; on version 1's evaluation population it gave 0.923 and 0.716 (first review),
so version 1's gate could be passed without the premise. Part (i) is what carries the gate: no reference that
ignores the binding of hypothesis to premise comes near 0.80. But 0.80 is only 0.04 above A's memorisation
reference: **passing E1 means "answers correctly in distribution", not "does so otherwise than by copying"**, which
is E5b's question. Length removal also costs a correct model something on ID48 (a score equal to the label plus
noise has S = 0.93 after removal). 0.80 is a convention stated before the fact; the value is reported whichever
side of it A falls.
*Outcomes:* passes → A is "a model that does the task in distribution" for the rest. Fails → the page says that a
1.5B LoRA at N = 7,184 did not reach it, E2 to E5 are still reported (they do not depend on the gate), and nothing
is said about "a model that does the task".

**E2. Balanced against official training data, where wording, premise and length carry nothing.** Statistic S on
POOL-LB.
- **E2a (primary):** S(B) − S(A), 5 against 5.
- **E2b (exposure control):** S(B) − S(A48), 5 against 3. B has 2.3 times as many training items of the tested
  strings as A and 3.9 times as many valid ones; A48 has the same number as B.
- **E2c (negative control, read first):** S(Bswap) − 0.5, 3 seeds. If claimed different from 0.5, B and POOL-LB
  share an artefact of the balancing function and E2a, E2b and E5 are not interpreted until it is explained.
*Prediction:* S(A) between 0.50 and 0.65; S(B) ≥ S(A) + 0.10; S(B) − S(A48) positive but smaller.
*What each outcome licenses:*

| E2a | E2b | Statement |
|---|---|---|
| claimed > 0 | claimed > 0 | on this pool, training on the balanced set gives higher S than the same number of unbalanced items of the same strings |
| claimed > 0 | not shown | B is above A; the difference is not shown to be more than exposure to the tested strings |
| not shown | any | not shown that balancing changes S; intervals given |
| claimed < 0 | any | reported as found |

**No outcome of E2 says what B learned.** Two things other than structure predict "B above A48" on this pool, and
both are measured: the memorisation reference is 0.898 from B's training set against 0.757 from A48's (B's set
carries the pool's strings with both labels), and the per-string premise-count classifier is 0.07 to 0.08 higher
from B than from A48. E5 is where a model is compared with those.
*E2c, how it is read:* S(Bswap) is reported with its 95% interval from the registered test. The control passes if
the interval lies inside 0.5 ± 0.05. With 3 seeds the test itself has almost no power against a leak of 0.05
(0.17 to 0.45) and 0.71 to 1.0 against 0.10, so a pass rules out only what survives a premise swap, at about that
size; it does not clear B and POOL-LB of shared artefacts (Section 1).
*Power:* Section 3. **Registered for a difference of 0.10 (E2a) and 0.15 (E2b).**

**E3. How far does A get without the item's own premise?** S(A) − S(C) on ID48, 5 against 3; beside it pooled
AUROC and F1 of A and C on ID, the share (AUROC_C − 0.5)/(AUROC_A − 0.5) with a bootstrap interval (descriptive),
and the text references fitted on A's training set.
*Prediction:* C's pooled AUROC on ID is at least 0.85 (the string prior alone is 0.885); S(C) on ID48 is near 0.5
after length is removed; S(A) − S(C) ≥ 0.15.
*Outcomes:* claimed > 0 → A uses the content of the item's own premise in distribution, by that much. Not shown →
not shown that it does. Either way the pooled AUROC of C says how much of A's headline in-distribution score is
available without the premise; the write-up says "available without", not "A is shortcut".
*Power:* Section 3. **Registered for 0.15.** Length removal lowers S of a correct model on ID48 (to about 0.93), for A
and C alike.

**E4. The authors' renaming: names, or the order the letters encode?** For A, on ID48, within seed:
- **E4a:** S(original) − S(renamed, A…F → Z…U). Replicates Table 5b's direction on held-out premises.
- **E4b:** S(renamed keeping order, A…F → U…Z) − S(renamed reversing).
*Prediction:* both positive; E4b at least half of E4a (most of the drop is the reversed order cue).
*Outcomes:* E4a claimed, E4b claimed → part of the published fragility under "variable refactorisation" is due to
the reversal of letter order (which includes the direction regularity and the order in which the premise lists
variables and statements), not to the new names. E4a claimed, E4b not shown → not shown
to be more than the names. E4a not shown → the drop does not reproduce for S in this model; the descriptive table
(pooled F1 on ID, ID renamed, ID paraphrased, the authors' sets, every condition) is still given beside Table 5b.
No A-against-B test of drops is run (different baselines; floor effects). Paraphrase is descriptive only: it
merges strings, so S is computed on the original string identity and reported as such.
*Power:* Section 3. **Registered for 0.10.** The premise SE used is that of two different models and overstates
the SE of a within-model difference; the power is a lower bound.

**E5. What is S made of? (on POOL-LB; tested for A and for B, 5 seeds each; reported for A48 with intervals, no test)**
- **E5a:** S inside cells of hypothesis string × adjacency, minus 0.5 (`stat.S_cells`). Adjacency is v1.0's
  `probes.structural_features(items)[:, 2]` (the pair is stated correlated and no statement makes it
  conditionally independent). Cells with at least 3 items per class (70 of 95 cells, 4,139 items, 43 strings);
  AUROC of the length-removed score (residual fitted within string, as in S) inside each cell, averaged with
  weights valid × invalid pairs, as v1.0 `15_answer_bias.py`. Adjacency alone scores S = 0.498 on POOL-LB.
- **E5b:** S(model) − S(memorisation reference of the model's own training set) (0.703 for A, 0.898 for B), the
  reference's scores resampled with the model's.
- **E5c:** S(model) − S(best text reference), the best being the largest S on POOL-LB among: lookup, hand parse and
  tuned stack of v1.0, and the per-string premise-count classifier (logistic and boosting), each fitted on the
  model's own training set, the stack's settings chosen by premise-grouped cross-validation inside that training
  set, none on POOL-LB. The best is chosen once on the full set, in the reference's favour.
*Prediction:* B passes E5a; B does not exceed its memorisation reference (E5b not shown, or negative).
*Outcomes:* per model, the rung of Section 0. A pure lookup of all statements naming the pair can pass E5a, which
is why L2 needs E5b and L3 needs E5c.
*Descriptive, beside the tests:* S of each model on the half of POOL-LB whose premises are nearer than the median
distance (4 statements) to a training premise and on the farther half, with the memorisation reference on each.
*Power:* Section 3. **Registered for 0.10 in each of E5a, E5b, E5c.** A model 0.05 above its memorisation
reference will most likely be filed "not shown", and the page will say that this is what the design can resolve.

**E6. Representations (descriptive in this version).** Base, A, B: the v1.0 probe ladder on POOL-LB (selection
and report halves by premise), at the final position and at the last token of the hypothesis, with the text
references of E5c and the model's own answer as reference rows. No rung rests on it: in a model that answers
correctly a probe for the gold label after the hypothesis decodes the model's own decision, so "probe above text"
would repeat E5c through a linear readout. What is reported: the layer at which the gold label becomes decodable,
and whether the probe exceeds the model's own answers (information present and not used).

**E7, E8. Steering and patching: not registered in this version.** They need: a site upstream of where the answer
is decodable; the model's own-answer direction and the text prediction's direction as controls, with the test on
the gold direction after projecting them out; a power figure from a small GPU pilot (a steering slope cannot be
computed from stored activations, as version 1 proposed). They will be registered in an amendment, reviewed
independently before any intervention runs, after E1 to E5 are audited. If that does not happen by the deadline,
the page says the causal tests were not run.

## 5. Order, quota, stopping

1. Time test. 2. A and B (5 seeds each). 3. A48, Bswap, C (3 seeds each). 4. Behaviour analysis E1 to E5 in the
order E2c, E1, E2, E3, E4, E5; one independent audit (code and result files, not conclusions). 5. E6 and its
audit. 6. Amendment for E7 and E8.
Three seeds of Bswap are the owner's allocation; two more would raise E2c's power against a leak of 0.05 from
about 0.25 to about 0.8 and are the first use of any quota left over.
Quota: 26 GPU hours per week through the queue runner; Kaggle reported 15 hours left on 16 November and resets on
Saturday 21 November 00:00 UTC. If the time test shows the 19 runs do not fit, runs are launched in the order
above and the rest wait for the reset; no condition is shortened.
Sunday 22 November 18:00: the page reports only audited results; everything else is "in progress". If the page
must go out before step 3 is complete, E2a alone may be reported (it needs only A and B), with E2c stated as not
yet run and no interpretation beyond the first column of E2's table.

## 6. Answers to the first review (`finetune/notes/audit_prereg_raw.md`)

| Finding (blocking) | Answer |
|---|---|
| 1. The claim rule (interval and all nine seed pairs) had a false-positive rate of 0.10 to 0.12 | Replaced by one test pooling seeds with both variance terms; 5 seeds for A and B; simulated false-positive rate at most 0.051 (Section 3) |
| 2. `power.py` simulated a different test | Rewritten: it calls the registered function; no clipping; 200,000 replications; premise SE from the built sets |
| 3. E3 and E4 quoted power that was never computed | Computed for the registered test: E3 is registered for 0.10 and E4 (a within-model difference) for 0.05 |
| 4. The in-distribution gate could be passed without the premise | E1 now requires S on ID48 (length removed) ≥ 0.80 and pooled AUROC above the string prior by test |
| 5. The split was selected on the label | Stated; a random 187 of the remaining premises are the in-distribution set; the POOL premises are named a shifted population |
| 6. A against B was confounded with exposure | A48 added (E2b); the outcome table says what each pattern licenses |
| 7. A nearest-training-premise copy scores 0.78 to 0.81 | Added as a reference per condition (E1, E5b); L2 requires beating it |
| 8. The adjacency cell was defined two ways | Pinned to v1.0's `structural_features[:, 2]`, with weighting, minimum cell size and residualisation stated (E5a) |

Should-fix items: the rescue with extra seeds is deleted; POOL-LB's size, bins and items per premise are fixed;
the balancing code exists and its output is stored; Bswap added; C's strata, sampling unit and meaning restated;
E4 reduced to two within-model tests plus a descriptive table, with an order-preserving renaming; the paraphrase's
merged templates and the files' duplicate rows stated; E5's references are S against S, fitted off the evaluation
set; E6 made descriptive; E7 and E8 withdrawn until registered properly; the 21-premise sets carry no test;
multiplicity handled by Holm over the family of tests (Section 3); one training-set draw per condition stated as a limitation;
the deadline's drop order stated (Section 5).

## 7. Answers to the second review (`finetune/notes/audit_prereg2_raw.md`): pass with five conditions

| Condition | Where it is met |
|---|---|
| 1. Multiplicity and power must agree | Section 3: the family is listed (eleven tests), claims are on Holm-adjusted p, power is computed at α = 0.05/11, and the registered effects are raised accordingly (0.10 and 0.15) |
| 2. One definition of each rung | Section 0: L0 to L3 defined by E5a, E5b, E5c only; E5c is in the family and tested for A and B; the unassigned pattern is assigned |
| 3. The memorisation reference | Section 1: values are the registered S; the score is described as coded; the reference is the best of a fixed family of fourteen copy rules per training set (0.898 / 0.757 / 0.703 on POOL-LB; 0.763 from A on ID48); E1's margin corrected to 0.04 |
| 4. What the checks and Bswap show | Section 1: the premise-alone check is stated to be constant by construction; the per-string premise-count classifier is reported (0.588 / 0.666 from B) and is a reference in E5c; Bswap's reach is stated; POOL-LB's use reworded; E2c no longer implies that a pass clears shared artefacts |
| 5. The size statement | Section 3: "at most 0.051" limited to normal per-seed values with equal SDs; the exceptions quoted; a rule for an outlying seed fixed |

Should-fix items taken: E2c read as an interval against a bound, with its low power stated; the premise-SE
assumption and its sensitivity stated; the three-per-class rule lives in `ftstudy/stat.py` with a test, and the
function that builds the resamples of D exists before any score does; E1 (ii) compares with string-and-length;
S by distance to the nearest training premise is reported with E5; E4b's outcome says "letter order"; the cost of
length removal on ID48 is stated; POOL-LB's and B's type mix stated. Not taken: five seeds for Bswap (the owner's
allocation is three; noted in Section 5 as the first use of spare quota). My values for the memorisation family
differ from the reviewer's in the third decimal for some rules (tie-breaking); the registered ones are those of
`references.json`.

## Amendments

(none)

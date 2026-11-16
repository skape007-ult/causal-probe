# Audit of finetune/PREREG.md before any run (raw reviewer report)

Reviewer: independent methods review (statistics, interpretability), 16 November 2026. Given: PREREG.md, power.py,
power.json, RESULTS.md, notes/handoff.md (b), src/causal_probe/data.py, scripts/15_answer_bias.py, raw data.
Nothing in the repository was edited. No model was loaded. My scripts are outside the repository
(scratchpad `audit/a1.py`, `a2.py`, `a3.py`, `sim.py`); every number marked "measured" below comes from them, on
`data/raw/corr2cause` through `causal_probe.data.load_c2c` and `data/derived/c2c_debiased_seed0_300x8_blocks25.json`.

Severity: **blocking** (do not train before it is fixed), **should fix**, **minor**.

## Measured facts used below

| Quantity | Value |
|---|---|
| Premises train / dev / test; overlaps | 2,338 / 22 / 21; train–test 0, train–dev 2, dev–test 3 (as stated) |
| TRAIN premises; items; items of the 48 strings; valid among those | 1,137; 98,964; 42,949; 12,666 (as stated) |
| Graph classes over TRAIN + pool + test (2,358 premises) | all distinct: 2,354 distinct relabelling-invariant signatures, the 4 colliding pairs checked over all 720 relabellings, none isomorphic (claim confirmed) |
| Valid rate of official items, TRAIN premises | **12.8%** (PREREG says "about 18%") |
| Valid rate of official items, the 1,200 pool premises | **23.8%** |
| Valid rate inside the 48 strings: TRAIN / pool premises | **29.5% / 54.8%** |
| TRAIN premises with both labels among the 48 strings | 900 of 1,137 |
| Upper bound on N: balanced within string / within premise | 24,436 / 18,762 (before length bins) |
| Pool premises: official items; in the 48 strings; valid | 106,746; 46,291; 25,378 |
| POOL-LB upper bound (string-balanced); pairs per string | 32,054 items; 80 to 581 pairs per string; about 13 to 19 pairs per premise |
| Hypothesis-string prior (fitted on TRAIN) scored on all official items of pool premises | AUROC **0.923**; valid-class F1 0.22 at threshold 0.5, **0.716** at the best threshold |
| Premise word count predicting the label within type, 48 strings: TRAIN / pool premises | 0.26 to 0.30 (as stated) / 0.32 to 0.41 |
| Nearest TRAIN premise (same variable count, same letters) to each pool premise | median symmetric difference 4 statements of a median 33; 43% of pool premises differ from a TRAIN premise by at most 10% of their statements |
| Copying the label of the same hypothesis in that nearest TRAIN premise | v1.0 pool: accuracy 0.774, mean within-string AUROC **0.777**; all 48-string items of pool premises: 0.830, **0.808** |
| `parent`, ancestor, descendant items in TRAIN: valid rate when the first-named letter is later in the alphabet | 0.000 (n = 16,494 + 8,305 + 8,189); `child` is never valid |
| "Queried pair marginally correlated" vs v1.0's "adjacent" (correlated and never conditionally independent), v1.0 pool | disagree on 32.6% of items; two-class cells 54 vs 78 |
| Condition C, random other TRAIN premise of the same variable count and length bin: assigned label equals the true label of (new premise, hypothesis) | 86% of items (70% inside the 48 strings), for any bin width from 10 to 100 words |
| C strata at 10 / 25 / 50-word bins: premises alone in their stratum | 56 / 23 / 6 |
| Authors' paraphrase file | 2,246 rows, 1,522 distinct inputs, three-way labels (342 / 712 / 1,192), **four templates for six relations**: parent and child share one, ancestor and descendant share one |
| Authors' refactorisation file | 2,246 rows, 1,884 distinct inputs |

## 1. Per experiment: falsifiable, determined, threshold fixed, forking paths

**Section 3, the claim rule.** "A difference between conditions is **claimed** only if (a) the 95% premise-bootstrap
interval of the difference of seed-averaged statistics excludes 0, and (b) all nine seed pairs have the same sign."
and "with 3 seeds against 3 an exact permutation test cannot go below p = 0.10, so no seed-level test is run".
- **Blocking.** Rule (b) *is* that seed-level permutation test, at its smallest attainable p. Under no condition
  effect the six seeds are exchangeable and "all nine pairs same sign" has probability 2/C(6,3) = 0.10. Rule (a)
  does not protect: the premise bootstrap holds the six fitted models fixed, so its interval contains no seed
  variance, and whenever the seeds separate the mean difference is several bootstrap standard errors wide. Simulated
  false-positive rate of (a)+(b) with seed SD 0.01 to 0.03 and bootstrap SE 0.004 to 0.012: **0.10 to 0.12**
  (sim.py). Every "α = 0.05" attached to an A/B/C comparison (E2, E3, E4) is therefore wrong by a factor of two.
  The statement in Deviations 4 ("no seed-level test") is also wrong.
- Should fix. (b) is undefined for the five-seed rescue (25 pairs? sign of the mean?), and for one-model tests
  (E5: "plus (b) across seeds" for a test against 0.5).
- Should fix. Is S computed per seed and then averaged (Section 3: "seed-averaged statistics") or on the averaged
  score (E5: "on the seed-averaged score")? The two differ; E2 does not say.

**S(model, set).** "mean over hypothesis strings of the within-string AUROC of the score for the label, with prompt
length removed (residual of the score on a quadratic in log token count, fitted within string ...)".
- Determined for the v1.0 pool (it matches `15_answer_bias.py`, `quadratic_in_log_token_count`, log centred within
  string, labels not used). Not determined for: the minimum number of items per string; ties; whether the token
  count is of the templated prompt (v1.0 `meta["n_tokens"]`) or the user text; on renamed and paraphrased sets,
  whether length is re-counted on the perturbed prompt; and on paraphrased sets what a "string" is (see E4). Minor
  each, should fix together: point to one function and one test fixture.

**E1.** "A reaches pooled AUROC ≥ 0.90 and valid-class F1 ≥ 0.70 on ID-heldout".
- **Blocking (validity of the gate).** The gate does not show that the model does the task. A table of
  P(valid | hypothesis string) fitted on TRAIN scores AUROC 0.923 on the ID-heldout population and F1 0.716 at its
  best threshold. A model that never reads the premise passes the AUROC half outright and the F1 half if its
  threshold happens to sit well. E6 to E8 are then run on "the model that does the task".
- Should fix. F1 is not determined: threshold (logit difference > 0?) not stated; it matters because A is trained at
  12.8% valid and scored at 23.8% (see Q3, split).
- Should fix. "mean of three seeds" of pooled AUROC, no interval in the decision: fixed, fine. But A-full differs
  from A in three things at once (items, epochs 1 vs 3, steps), so "N is the risk" cannot be read from A-full passing.

**E2.** Falsifiable in direction; statistic determined once S is; threshold: see claim rule.
- Should fix. Outcome map has holes and an undefined word: "both clearly above 0.5" ("clearly"?); one interval
  inside 0.5 ± 0.05 and the other not, with the difference not claimed, is unassigned.
- Should fix. "difference claimed positive → training on debiased data buys validity-tracking that official data
  does not." This sentence does not follow from the statistic; see Q3 (exposure confound; lookup; nearest premise).
- Should fix (forking path). The rescue: "if the observed seed SD exceeds 0.02 and the observed difference lies
  between 0.04 and 0.10, two more seeds ... the test is run once on five seeds". Seed SD of which condition (A, B,
  pooled)? Estimated from three values it has a relative error of about 50%. And what is rule (b) on five seeds:
  with all 25 pairs, power for a true 0.05 at seed SD 0.03 **falls** from 0.50 (3 seeds) to 0.25 (5 seeds)
  (sim.py); the rescue as written makes the test weaker in exactly the case it is for.
- Should fix. POOL-LB's size is not fixed and the power statement assumes "12 to 25 pairs per string"; the data
  allow 80 to 581 pairs per string and about 25 items per premise. Either number changes the test (Q2).

**E3.** "shortcut share = (AUROC_C − 0.5) / (AUROC_A − 0.5), with interval"; "share near 1"; "well below 1".
- Should fix. No threshold: "near 1" and "well below 1" are not defined, so the outcome sentences are not
  falsifiable. The prediction (≥ 0.7) is; the outcome map should use the same number.
- Should fix. The registered *test* is on AUROC_A − AUROC_C, the reported quantity is a ratio; a ratio's interval
  method is not given (bootstrap of the ratio of seed means?).
- Should fix. "(ii) Sanity: S(C, POOL-LB) interval contains 0.5": interval of the seed mean? one seed failing?
- Should fix. "(iii) Text references ... trained on A's own training set: hypothesis-only, length-only, lookup,
  hand parse, tuned stack": hyper-parameters and the tuning data of the "tuned stack" are not stated (v1.0 tuned by
  cross-validation on the pool, which here is evaluation data).

**E4.** "Drop = statistic on the original items − statistic on the perturbed version", "on ID-heldout (pooled AUROC
and F1) and POOL-LB (S)", "Test: drop(A) − drop(B), (a)+(b)."
- Should fix (multiplicity, forking). Two perturbations × three statistics × two sets = up to six A-against-B
  tests, one prediction (renaming, ID-heldout, AUROC, ≥ 0.05). Which are tests? None is corrected.
- Should fix. Absolute drops are compared between models with different baselines: B never saw 72 of the 120
  strings nor a natural label rate, so its pooled AUROC on ID-heldout starts lower and has less to lose.
  "drop(A) > drop(B)" can be a floor effect. The outcome "A drops, B does not → the fragility ... is a property of
  the training distribution" does not follow.
- Should fix (determinacy). The authors' paraphrase uses four templates for six relations: ancestor and descendant
  strings of one pair become the same sentence. On our pools the 48 strings collapse, some premises get two
  identical prompts, and "within string" must be defined (original string id, or paraphrased string?). Not stated.
- Should fix (interpretation). Renaming A…F → Z…U reverses alphabetical order. In this benchmark the letters carry
  the causal order: a directional hypothesis whose first letter is the later one is never valid (0 of 32,988 TRAIN
  items). A model that uses "earlier letter can cause later letter" (a real regularity of the data, not a name
  shortcut) is turned upside down by this renaming. A drop therefore does not separate "names read literally" from
  "order cue reversed". The authors' file does the same (A→Z, B→Y), so replication is fine; the interpretation is not.
- Minor. The authors' perturbed files have 2,246 rows but 1,884 and 1,522 distinct inputs; a "replication table"
  on rows double-counts. The paraphrase file's label is three-way; "valid = label == 1" is right (342 entailment)
  but should be said for that file specifically.

**E5.** "(i) S restricted to cells of hypothesis string × adjacency of the queried pair (premise states the pair
correlated / independent), cells with both labels only".
- **Blocking (determinacy).** Two analysts will compute different numbers. The parenthesis defines the cell by the
  *marginal* statement; v1.0 (`probes.structural_features[:, 2]`, used by `15_answer_bias.py`) defines adjacency as
  "correlated and never conditionally independent". On the v1.0 pool the two disagree on 32.6% of items and give 54
  against 78 two-class cells. Also open: mean over cells unweighted (as S) or weighted by valid × invalid pairs (as
  `15_answer_bias.py`); length residual fitted within string or within cell; minimum cell size (cells with one
  item of a class exist: minimum 1 in both definitions).
- Should fix. "If the build leaves fewer than 6 pairs per string in two-class cells, E5(i) is dropped": mean,
  median or minimum over strings? Under v1.0's definition the pool has a median of 7.5 minority items per cell and
  3 strings with no two-class cell.
- Should fix. "(ii) pooled AUROC minus the hand parse's and minus the lookup's on the same items": pooled AUROC of
  a language model's logit difference mixes between-string offsets (the reason S exists, v1.0 audit 11) with
  within-string signal; the text references are fitted classifiers. Use S for both. Where the references are
  fitted (condition's training set, or cross-validated on POOL-LB) is not stated here.
- Should fix. The outcome map is not exhaustive: above 0.5 inside cells but not above the hand parse; above the
  lookup but below the hand parse. And the logic is loose: v1.0's lookup uses all statements naming the pair
  (including the conditioning sets), so a pure lookup can be above 0.5 inside string × adjacency cells. Passing (i)
  is not "beyond lookup".
- Should fix. "permutation of labels within cell": POOL-LB will have many items per premise; a within-cell
  permutation treats them as exchangeable. v1.0 permuted within premise.

**E6.** Statistic determined by v1.0 code if POOL-LB has blocks; "selection half of the blocks" presupposes a block
structure that the new balancing function is not said to produce. Should fix.
- Should fix (what L3 means). In a fine-tuned model that answers correctly, a probe for the gold label at any
  position after the answer is formed decodes the model's own decision. "Probe beats every text reference" is then
  E5(ii) measured again through a linear readout, not evidence of a representation "beyond the text". Register the
  positions and say which outcome would *not* be implied by E5.
- Minor. "for the model that passes E5" in the prediction, but the test is run on Base, A, B at α = 0.05/3. Fixed, fine.

**E7.** Threshold and statistic fixed (v1.0 `07d`). Forking paths:
- Should fix. "at most two models": which two of A, B, A-full if all pass the gate? Not stated.
- Should fix. "r is estimated on the selection half only, with 100 controls, from activations already exported by
  the training notebooks (no new GPU time)". A steering slope is the effect of an intervention on the forward
  pass; it cannot be computed from stored activations. Either this is a different quantity (then define it and
  show it predicts r) or it needs GPU time. As written it is not executable, and whoever executes it will choose.
- Minor. The gate uses S on all of POOL-LB, including the report half the test is run on. Different statistic; say so.

**E8.** "Power: computed ... before launch and appended here" is a promise, not a registration; acceptable only if
the amendment is committed before any fine-tuned activation is patched. The number of same-premise, same-type pairs
of both kinds in POOL-LB is unknown. Minor.

## 2. Power

**Re-derivations (independent of power.py).**

| Quoted | power.json | Mine | Verdict |
|---|---|---|---|
| E2, seed SD 0.03, difference 0.05: "0.41 to 0.52" | 0.38 to 0.52 | analytic z-test, SE = sqrt(0.012² + 2·0.03²/3) = 0.0273: 0.45 | matches (for the test power.py simulates) |
| E2, seed SD 0.03, difference 0.10: "0.96 to 0.98" | 0.96 to 0.98 | analytic 0.956 | matches (same caveat) |
| E5, 12 pairs, 0.55 / 0.60: "0.52", "0.99" | 0.5175, 0.9875 | Hanley–McNeil null SE 0.0174 per model, ×√2 as coded: 0.53, 0.98 | matches as coded |
| E5, 6 pairs: "0.29 and 0.79" | 0.285, 0.785 | 0.29, 0.81 | matches as coded |
| E7, r = 2.0: "0.86 at α = 0.05 and 0.51 at 0.025" | 0.857, 0.511 | normal approximation 0.88, 0.55 (ignores sampling of the control quantile) | matches |
| E6, 2 points: "0.98" | 0.980 | 0.980 | matches |

The arithmetic is right. The problems are in what is simulated.

1. **Blocking. The simulated test is not the registered test.** `power_within_string` rejects when
   |difference| / SD(difference over replications) > 1.96, where that SD includes the seed variance. The registered
   test is (a) a premise bootstrap that cannot see seed variance and (b) all nine seed pairs. Under the registered
   rule (sim.py, seed SD 0.03): power 0.50 for 0.05 and **0.92 to 0.94** for 0.10 (quoted 0.96 to 0.98), at a real
   false-positive rate of about 0.10. So the headline "registered to detect 0.10" survives, but with twice the
   stated α. If the rule is repaired to hold α = 0.05 with six models, power for 0.10 must be recomputed; with 3
   against 3 no rank-based seed test can reach 0.05, so the repair needs more seeds or a parametric seed-level test
   (simulated, Welch t on the seed values: 3 vs 3 gives 0.76 for 0.10 at seed SD 0.03 and 0.96 at 0.02, 0.28 and
   0.52 for 0.05; 5 vs 5 gives 0.99 for 0.10 and 0.61 / 0.89 for 0.05).
2. **Blocking. E3 claims power it does not have.** "a difference of 0.03 is detected with power above 0.9 even
   with seed SD 0.02." There is no such calculation in power.py. With seed SD 0.02 and item SE 0.01 the registered
   rule has power **0.43** and a correct test 0.35 (sim.py). It reaches 0.83 only if the seed SD is 0.005. E4's
   "Power: as E3" inherits this and is worse: a difference of two drops. Both are tests claimed above 0.6 that are
   not, on the document's own assumption.
3. Should fix. `noise = rho * e + sqrt(1 - rho**2) * ...` with `e` shared: the correlation between two models'
   noise is rho² = 0.25, not the 0.5 the document quotes ("item noise correlated 0.5 between conditions").
   Measured 0.247. The error is conservative for the no-seed-variance rows and irrelevant once seed variance
   dominates, but the sentence in PREREG is false.
4. Should fix. `np.clip(auc + seed_sd * z, 0.5, 0.995)`: for the "A = 0.5" rows half of A's seeds are clipped up,
   which raises A's mean by about 0.012 and shrinks the true difference from 0.05 to about 0.038. Those rows
   (0.38 to 0.43) are for a smaller effect than labelled. A fine-tuned model can be below 0.5 within string (v1.0
   saw 0.47; the length leak is 0.26 to 0.30); do not clip.
5. Should fix. The design in the simulation is not the design in the data. It assumes 48 strings × 12 to 25 pairs,
   items independent. POOL-LB "rebuilt from all official items of the 1,200 pool premises" can hold up to about
   32,000 items, 80 to 581 pairs per string, about 25 items per premise: unequal strings and clustered items. Then
   the bootstrap SE is a few thousandths, rule (a) is satisfied by any visible difference, and the whole decision
   is rule (b). No premise effect is simulated in A at all.
6. Should fix. `A_one_model_vs_half` calls the two-model function with a null model at 0.5, so its SE is √2 too
   large: a one-sample test at 12 pairs has power 0.82 for 0.55, not 0.52. Conservative, but it is not the
   registered test either (permutation within cell, α = 0.025 one-sided, on a seed-averaged score, plus (b)), has
   no seed variance, assumes one two-class cell per string with 6 or 12 pairs, and equal cells. The real cells are
   very unequal (minimum 1).
7. Should fix. 400 replications (200 for the 21-premise case): Monte-Carlo SE up to 0.035. The table is not
   monotone in sample size (0.515 at 12 pairs, 0.41 at 25) for that reason. "0.61" is not distinguishable from 0.6.
8. Minor. `power_few_premises` gives both score sets the same premise effect, so the premise-level variance of the
   *difference* is zero by construction; it answers "is a paired difference detectable when premises do not differ
   in the difference". The claim made from it ("0.61 to 0.82 ... falls below 0.6 with seed variance") is used only
   to *not* run a test, so the error is harmless; the within-model figure "0.99 to 1.0" should not be quoted as power
   of anything real. A percentile bootstrap over 21 clusters is also anti-conservative; its size was not simulated.
9. Should fix. E6: the SE of 0.45 points is one v1.0 interval (Gemma-2-9B, 1,200 report-half items, a probe highly
   correlated with the text stack). POOL-LB has a different size and a fine-tuned model's probe need not be
   correlated with the text stack to the same degree. The "registered for 2 points" depends on a transfer that is
   not argued.
10. Should fix. E7: `gold_noise = 0.3` is one pair of fits of one model. The decision rule conditions on an
    estimated r ≥ 2.0; power given "estimate ≥ 2.0" is not power at r = 2.0 (regression to the mean; the estimate has
    noise of at least 0.3 plus a control SD from 100 draws). Not shown to be ≥ 0.6.
11. Minor. The seed model treats seed variance as a shift of the true AUROC and assumes 0.03 without evidence; fine
    as an assumption, but the training sample of A and the build of B are one draw each, shared by all seeds, so
    "consistent across seeds" does not cover "consistent across draws of the training set".

**Tests claimed ≥ 0.6 that are not:** E3 (0.43 at the stated seed SD), E4 on ID-heldout (inherits E3). E2 at 0.10
holds (0.92 to 0.94) at the cost of α ≈ 0.10. E5, E6, E7 are not shown either way for the registered tests.

## 3. Design validity

1. **Blocking. The split is not random, so "in distribution" is not.** EVAL premises are "the 1,200 premises of the
   v1.0 debiased pool", and `c2c_debiased_pool` picks premises *because* they hold a valid item of a two-label
   string (`pos_prems`, scarce strings first). What is left for training is the complement. Measured: valid rate
   12.8% in TRAIN against 23.8% in the pool premises; inside the 48 strings 29.5% against 54.8%; 237 TRAIN premises
   have no valid item of the 48 strings at all; the length leak differs (0.26 to 0.30 against 0.32 to 0.41). The
   sentence 'A, official ... (all 120 hypothesis strings, about 18% valid)' is wrong for this TRAIN (12.8%).
   Consequences: E1 measures transfer to premises selected on the label, not in-distribution skill; F1 at a fixed
   threshold is under a prior shift of a factor two; the hypothesis-string priors that C and A learn are the wrong
   ones for the evaluation premises; B and POOL-LB are balanced, so B is the only condition evaluated at its own
   training prior.
2. **Blocking. E2 is confounded with exposure.** A draws N items uniformly from all 98,964: 43% fall in the 48
   strings and 29.5% of those are valid. B has all N items in the 48 strings, half valid. For any N, B sees 2.3
   times as many training items of the strings on which both are tested and 3.9 times as many valid ones, at the
   test set's label rate and length bins. "S(B) − S(A) > 0" is predicted by "more examples of the tested strings"
   without any reference to shortcuts. The outcome sentence ("training on debiased data buys validity-tracking
   that official data does not") is not supported by this contrast.
3. **Blocking for the L1/L2 reading. A nearest-premise copy scores S ≈ 0.78 to 0.81 on held-out graphs.** Copying
   the label of the same hypothesis string from the most similar TRAIN premise (same letters, median 4 statements
   different) gives mean within-string AUROC 0.777 on the v1.0 pool and 0.808 on all 48-string items of the pool
   premises: level with v1.0's lookup (0.765 accuracy) and hand parse (0.816). So "above 0.5 on the length-balanced
   pool", "above 0.5 inside cells" and probably "above the lookup" are all reachable by memorising training
   premises and interpolating. This reference is missing from E3(iii), E5 and E6. (It also needs the label of a
   string in the neighbour premise, which B's and A's samples contain only partly: compute it on each condition's
   actual training set.)
4. Should fix. **Condition C.** "the premise content is unrelated to the label". Not so:
   - The hypothesis variables are always present (same number of variables means the same letters), so there is no
     detectable contradiction of that kind. Fine.
   - But the assigned label equals the true label of (new premise, hypothesis) for 86% of items (70% in the 48
     strings), because most strings are single-label and the stratum fixes density. C is trained on a mostly
     *correct* but noisy Corr2Cause. That is what a shortcut control should be, but then "C measures how far
     hypothesis wording plus length carry" should read "hypothesis wording plus everything about a premise that its
     variable count and length bin determine" (number of statements, density).
   - The strata are thin at the top. With 10-word bins 56 premises are alone in their stratum ("never the item's
     own premise" is then impossible: rule undefined) and 141 are in strata of at most three, where the "swap" is a
     near-deterministic relabelling of premises and C keeps premise-level information. C's bin width is tied to B's,
     which is tuned ("rebuilt with narrower bins"), so C's meaning moves with a choice made for another purpose.
   - "the premise of another TRAIN item": drawn among items (weights premises by item count) or among premises?
     Among all TRAIN items or A's N? Not stated.
   - C cannot be scored on its own claim. The share (AUROC_C − 0.5)/(AUROC_A − 0.5) says shortcuts *suffice* for
     that much; "share near 1 → A's in-distribution score is reproducible without reading the premise" is the right
     wording and should not become "A is shortcut" in the write-up. C does read the premise's length and size.
5. Should fix. **B, and POOL-LB built by the same function.**
   - Balancing selects items given their labels. It removes the four audited statistics (iv) and can leave or
     create others: string × adjacency (the lookup, acknowledged), string × number of variables, string ×
     number of independence statements, residual length inside a bin (same sign in B and POOL-LB, because the
     same function meets the same leak), and anything specific to how the function breaks ties. Whatever it
     leaves is identical in B and POOL-LB and absent from A. E2 is unfair to A in that respect too.
   - The only negative control (E3 ii) is S(C, POOL-LB), and C is built from A. There is no control for B's own
     artefacts. Add **B-swap**: B's items with the premise swapped inside variable count × length bin, one seed.
     S(B-swap, POOL-LB) must be 0.5; if it is not, B and POOL-LB share an artefact and E2 is void.
   - (iv) is required of B. It is not said to be required of POOL-LB, nor whether POOL-LB uses B's bin width or its
     own, nor POOL-LB's size, items per premise, or whether it has blocks (E6 needs them).
   - (iv) allows |AUROC − 0.5| ≤ 0.02 per classifier, against a registered effect of 0.10: adequate. The classifiers
     in (iv) are not specified (model class, features); a weak classifier passes trivially. Name them.
   - `finetune/ftstudy/balance.py` and `balance_reference.py` do not exist in the tree. N, the bins, POOL-LB and
     all power inputs depend on code not yet written. The document must gain an amendment with the built numbers,
     and the power must be rerun on them, before the first training run.
6. Should fix. **The E5 cell** is defined two ways (Q1). With either, "L1 = not above 0.5 inside cells" is a
   failure to reject at a registered minimum effect of 0.60: a model at 0.55 inside cells would be filed as L1.
   The ladder should say "not shown above" and give the interval.
7. Should fix. **Seed rule (b):** too lax as a guard (false-positive rate 0.10 on its own, and (a) adds almost
   nothing once seed variance exceeds the bootstrap SE); too strict as a detector (power 0.50 for a true 0.05 at
   seed SD 0.03, 0.75 at 0.02; with five seeds and all 25 pairs, 0.25 and 0.57).
8. Should fix. **E7 cannot separate "used" from "read back" in a model that does the task.** The prompt ends with
   the hypothesis (`prompts.user_parts`), so the "last token of the hypothesis" has the whole input behind it and
   only template tokens after it. In a model with S ≥ 0.65 the gold class-mean difference there is, to the extent
   the model is right, the model's own decision variable. Steering along it and seeing the answer move shows that
   the decision is linearly represented before the answer token, which any model that answers must satisfy
   somewhere. The 500 controls (random labellings with the same balances) are unrelated to the model's output, so
   they do not control for this. Needed: (i) a direction for the model's own answer (class means of the sign of
   its logit difference) and for the text stack's prediction, fitted the same way; (ii) the gold direction with
   those projected out; the claim "validity is used" rests on (ii), and on a site upstream of where the answer is
   decodable (premise-end or earlier hypothesis tokens). Otherwise L4 is true by construction for any model that
   passes the gate.
9. Should fix. E4 renaming reverses the alphabetical order that carries causal order (Q1). Add an order-preserving
   renaming (A…F → U…Z) to separate the two.
10. Minor. Seeds share one draw of A's sample and one build of B.
11. Minor. E6's Base model is evaluated zero-shot; A and B were trained on the answer token. A difference between
    Base and fine-tuned probes is expected from training on the label and is not by itself a finding.

## 4. Leakage

- "every premise is its own graph class, so held out by premise is held out by graph": the premise claim is
  **true** (confirmed, including the 21 test premises). It is **not sufficient**. Held-out graphs are near-copies of
  training graphs (median 4 of 33 statements different under the same letters; 43% within 10%), and a label copy
  from the nearest training premise reaches within-string AUROC 0.78 to 0.81 (Q3.3). Sub-structures are shared
  wholesale: almost all premises have six variables (1,036 of 1,137 in TRAIN, 1,160 of 1,200 in the pool), and the
  label of a pair mostly depends on local statements. **Should fix** (blocking for any L2 claim): report the
  nearest-premise reference and, for the strongest claims, a subset of POOL-LB whose premises are at least a stated
  distance from every training premise.
- The overlap test checks "shared premise hashes after variable renaming" with one fixed map; it should use the
  canonical form (it says "shared graph classes" as well; fine if that is the canonical one).
- Dev is unused and its two shared premises are excluded: fine. Test premises are disjoint from train: fine.
- Tuned on evaluation data: (a) the choice between A and A-full as "the model that does the task" is made on
  ID-heldout (mild; it is a gate); (b) E7's gate uses S on all of POOL-LB (mild); (c) POOL-LB's bin width, if
  adapted on POOL-LB, uses labels but no model (fine if fixed before training); (d) the "tuned stack" reference: if
  tuned by cross-validation on POOL-LB as in v1.0, the reference is tuned on evaluation data while the model is
  not, which is conservative for claims that the model beats it; say which. (e) Training hyper-parameters are
  fixed: fine. (f) The 300-step time test trains on A's items and scores 500 evaluation items: state that those
  scores are not looked at beyond "ran".
- ID-heldout (6,000 of 106,746 items, 5 per premise) and POOL-LB come from the same premises: fine, both are
  evaluation.

## 5. Deviations

- Deviation 1 is incomplete. It does not say that the evaluation premises were selected on the label by the v1.0
  pool builder and that TRAIN is therefore not a random sample of official train (Q3.1). This is the largest
  deviation from "official train as is" and is missing. "A-full covers the risk" overstates: A-full changes items,
  epochs and steps together.
- Deviation 2 (balance within premise): justified by D21.
- Deviation 3 (bins, not exact): justified; but "The verification in (iv) is the criterion, not the bin width"
  leaves an open loop with no stopping rule other than success, and ties C's definition to it.
- Deviation 4: "no seed-level test" is not true of rule (b).
- Deviation 5: fine.
- Missing: (i) B is restricted to 48 of 120 strings and A is not (the exposure confound); (ii) N is unknown at
  registration (bounded above by 18,762, likely far less after length bins; A at N may see about 5,000 items of
  the tested strings); (iii) POOL-LB's size and structure are unknown at registration; (iv) the balancing code does
  not exist yet; (v) the paraphrase is the authors' four-template one, which merges relations; (vi) the renaming
  reverses letter order; (vii) one training-set draw per condition; (viii) the power of E3, E4 and E8 is asserted,
  not computed; (ix) the stopping date (22 November) is a deviation from "run what is registered" and should say
  which experiments are dropped first.

## 6. The three changes I would require before the first training run

1. **Repair the inference rule and recompute every power figure for the rule actually registered.** Replace
   (a)+(b) by one test whose size is known: either five seeds per condition for A and B from the start (exact
   permutation on seed-level S, two-sided p ≥ 2/252; or Welch t on seed values), or keep three seeds and state
   that the false-positive rate of an A/B claim is 0.10. Add the seed variance to the interval (bootstrap premises
   *and* resample seeds). Then rerun power.py with that test, on the built POOL-LB (real strings, real cell sizes,
   premise clusters), no clipping, correlation as documented, at least 2,000 replications, and include E3 and E4,
   which currently quote "above 0.9" where the document's own assumptions give 0.43. Drop the tests that fall
   under 0.6. Define rule (b) for the rescue or delete the rescue.
2. **Make the A-against-B contrast interpretable.** (a) Add condition **A48**: N items drawn uniformly from the
   42,949 TRAIN items of the 48 strings, natural labels, no balancing; E2's primary contrast becomes B against A48
   (same strings, same count, only the balancing differs), with A kept as the "official" row. (b) Add the control
   **B-swap** (one seed) with the requirement S(B-swap, POOL-LB) = 0.5 before E2 is read. (c) Fix, in an amendment
   committed with the built data: N, B's and POOL-LB's bin widths, POOL-LB's item count, items per premise and
   block structure, (iv) for POOL-LB as well as B with the classifiers named, C's sampling unit and minimum stratum
   size (merge strata under, say, 10 premises). (d) Reword E2's first outcome so that it claims only what the
   contrast shows.
3. **Fix what "does the task" and "beyond lookup" are measured against.** (a) E1's gate: require within-string
   AUROC on ID-heldout (two-label strings) above a stated value, or pooled AUROC above the hypothesis-string prior's
   0.923 by a stated margin; fix the F1 threshold; and either accept and state the label-prior shift between TRAIN
   and the pool premises or carve a random 150 to 200 of the 1,137 TRAIN premises out as the true in-distribution
   set. (b) Add the nearest-training-premise reference (0.78 to 0.81 here) to E3(iii), E5 and E6, computed on each
   condition's own training set; L2 requires beating it. (c) Pin E5's cell to `structural_features[:, 2]` (or to
   the marginal statement, but one of them), the weighting, the minimum cell size, the permutation unit (premise),
   and compare S with S, not pooled AUROC with accuracy. (d) For E7, register the own-answer and text-prediction
   directions as controls and the projected-out gold direction as the test, at a site where the answer is not yet
   decodable; replace the "r from exported activations" rule by something that can be computed.

## What is fine (one line each)

- Premise counts, overlaps, TRAIN = 1,137 premises / 98,964 items / 42,949 / 12,666: reproduce exactly.
- Graph-class claim: true.
- Length leak of 0.26 to 0.30 within type in TRAIN: reproduces.
- Training recipe is fully fixed; no early stopping; no tuning on evaluation data in training.
- No test on 21 premises: right decision.
- Arithmetic of power.json: reproduces analytically for every number I checked.
- E2's honesty about 0.05 being undetectable at seed SD 0.03: correct, and survives the rule change.
- Deviation 2 and the decision to treat the lookup as its own rung.

# Second audit of finetune/PREREG.md (version 2) before any training run (raw reviewer report)

Reviewer: second, independent methods review (statistics, interpretability), 16 November 2026. Given: PREREG.md v2,
the first review, v1, `seedtest.py`, `power.py`, `power.json`, `build_data.py`, `datasets.py`, the two reference
implementations, the stored sets and reports, v1.0 `data.py`, `probes.py`, `15_answer_bias.py`, raw data.
Nothing in the repository was edited except this file. No model was loaded (the Qwen tokenizer was, to count tokens).
My scripts and their outputs are outside the repository, in
`/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bd8db98d-2abd-448b-abdd-fc03c722bc29/scratchpad/audit2/`:
`d1.py` (raw data, counts, the alphabet claim), `d2.py` (stored sets, balance, references, bootstrap SEs),
`d3.py` (premise-surface cue, E4 sets, Holm), `sim.py` (the test, own implementation; does not import `seedtest`
or `power.py`), each with a `.json` of results. "Measured" below means from these.

Severity: **blocking** (must be edited before the first training run), **should fix**, **minor**.

## Verdict

**PASS WITH CONDITIONS.** The data are built as described (every balance property, cap, stratum, count and hash I
checked on the stored sets holds), there is no premise or graph-class leak, the registered test has the stated size
for normal seed values, and `power.py` simulates the function the analysis will call. The five conditions in Task 7
are all edits to the text of PREREG.md (definitions of references, of the multiplicity family, and of what two
controls can show). None changes a training set or the training recipe. They must go in before the first run
because after it the file is amendment-only.

---

## Task 1. The first review's findings, one by one

### Blocking findings

| # | First review | Status in v2 | Evidence |
|---|---|---|---|
| 1 | Claim rule (interval + nine seed pairs) had false-positive rate 0.10 to 0.12 | **Resolved**, with a caveat on wording | "SE² = s_X²/n_X + s_Y²/n_Y + v_premise … Student's t, Welch–Satterthwaite degrees of freedom of the seed terms"; `seedtest.t_and_df`. My simulation, normal seeds, equal SD, 80 settings × 200,000: maximum 0.0513. Caveat: "at most 0.051 in every setting" is true only of the settings simulated (Task 2). |
| 2 | `power.py` simulated a different test | **Resolved** for the test; **partly** for its inputs | `simulate()` calls `seedtest.t_and_df` and `seedtest.p_two_sided`; no clipping; 200,000 replications. The premise SE is from simulated scores on the real items under one assumption (two models share half of their item-noise variance); and no power is computed for the Holm-adjusted rule on which rungs are claimed (Task 3). |
| 3 | E3 and E4 quoted power never computed | **Resolved** | power.json has both; I reproduce E3 0.10 → 0.952 and E4 0.05 → 0.643 at seed SD 0.02 with my own code. |
| 4 | The in-distribution gate could be passed without the premise | **Mostly resolved** | "(i) mean S on ID48 ≥ 0.80, and (ii) pooled AUROC on ID exceeds the hypothesis-string prior's (0.885…)". (i) does the work: a per-string classifier on premise counts only scores S = 0.586 on ID48. (ii) does not: string prior + prompt length scores 0.922 (the document's own number), string prior + four premise counts + length 0.929 (logistic) and 0.937 (boosting), measured; the registered margin is 0.885 + 0.05 = 0.935. And 0.80 is not "0.12 above" memorisation (Task 5.2). |
| 5 | The split was selected on the label | **Resolved** | "a seeded random 187 are the ID premises … The POOL premises are a shifted population". I regenerate the 187 from `default_rng(11)` and they are the stored ones; valid rates 12.45% / 14.53% / 23.77%, 28.69% / 33.49% / 54.82% inside the 48 strings, as quoted. |
| 6 | A against B confounded with exposure | **Resolved as to exposure; a second confound is not addressed** | A48 exists (7,184 items, 48 strings, 28.8% valid). But B is drawn by the same selection function as POOL-LB and A48 is not; a no-binding classifier already predicts B − A48 = 0.06 to 0.10 from that alone (Task 5.3). |
| 7 | A nearest-training-premise copy scores 0.78 to 0.81 | **Partly resolved** | The reference exists and is computed per condition. But (a) the quoted values are not the registered S, (b) the score is not "the label of the string there", (c) k = 1 is the weakest member of its family by 0.05 to 0.11 (Task 5.2). |
| 8 | Adjacency cell defined two ways | **Resolved** | "Adjacency is v1.0's `probes.structural_features(items)[:, 2]` … Cells with at least 3 items per class (70 of 95 cells, 4,139 items, 43 strings) … weights valid × invalid pairs". Measured: 95 / 70 / 4,139 / 43; adjacency alone S = 0.498. |

### Should-fix items

| First review | Status |
|---|---|
| Rule (b) undefined for 5 seeds and one-model tests | Resolved (rule deleted). |
| S per seed then averaged, or on the averaged score | Resolved: "S is always computed per model and then combined; scores are never averaged across seeds." |
| S not determined (minimum items, ties, token count, renamed sets, paraphrase) | Partly. All five are stated in Section 3 and E4. But "one function and one test fixture": the restriction to strings with at least 3 per class is in neither `within_string_reference.py` nor its tests, and `build_data.py` computes every quoted reference S without it (Task 4). Whether a string's eligibility is fixed on the full set or re-decided in each premise resample is not stated (`power.py` re-decides). |
| E1 F1 threshold; A-full | Resolved (two thresholds named; A-full removed). |
| E2 outcome map; "buys validity-tracking" | Resolved (table; "No outcome of E2 says what B learned"). |
| The rescue with extra seeds | Resolved (deleted). |
| POOL-LB size not fixed | Resolved (4,706 items, 1,193 premises, stored with hash). |
| E3 thresholds, ratio interval, sanity check, tuned stack | Mostly. Outcome is the test's; stack tuned "by premise-grouped cross-validation inside that training set". The ratio's "bootstrap interval" is still unspecified (ratio of seed means? resampling premises, seeds or both?): minor, it is descriptive. |
| E4 multiplicity, floor effects, paraphrase templates, alphabetical order, duplicate rows | Resolved (two within-model tests; order-preserving renaming; stated). One wording issue, Task 5.5. |
| E5 references S against S; outcome map; permutation unit | Mostly. One pattern is still unassigned (E5a not shown, E5b claimed). |
| E6 | Resolved (descriptive). |
| E7, E8 | Resolved (withdrawn). |
| power.py correlation, clipping, design, one-sample, replications | Resolved. `premise_se` uses `sqrt(0.5)·shared + sqrt(0.5)·own`: correlation 0.5 as documented. |
| Condition C: meaning, strata, sampling unit | Resolved. Measured: 93 strata of 10 to 19 premises; every donor of C and Bswap is in the item's stratum and is another TRAIN premise; C's kept label correct on 85.3% of the 83.8% of items that exist on the new premise (68.4% inside the 48 strings). |
| B and POOL-LB share whatever the function leaves; add B-swap; name the classifiers | Partly. Bswap added; classifiers named. But the "premise surface statistics" check is vacuous by construction and Bswap cannot see the cue I measured (Task 5.3). |
| "L1 = not above" is a failure to reject | Resolved ("not shown … never read as absent"). |
| Leakage: canonical form; a subset of POOL-LB at a stated distance from training premises | Canonical form: resolved. Distance subset: **not adopted** and not answered in Section 6. |
| Time test scores | Resolved ("not opened … adapter discarded"). |
| Deviations list, drop order | Resolved in substance (Section 5). |

---

## Task 2. The registered test

### 2.1 Size, independently simulated (`sim.py`, 200,000 replications per setting unless stated; Monte-Carlo SE 0.0005)

| Setting | 5 v 5 | 5 v 3 | one sample, 5 | one sample, 3 |
|---|---|---|---|---|
| Normal seeds, equal SD 0.005 to 0.04, premise SE 0 to 0.06: maximum | 0.0504 | 0.0504 | 0.0513 | 0.0506 |
| Same, at the realistic point (seed SD 0.02, premise SE 0.011) | 0.024 | 0.019 | 0.009 | 0.0004 |
| Unequal seed SD, premise SE 0: maximum | 0.053 | **0.066** (SD 0.01 in the 5, 0.04 in the 3) | – | – |
| Unequal seed SD, premise SE ≥ 0.005 | ≤ 0.05 | ≤ 0.05 | – | – |
| Bootstrap estimate of the premise SE noisy (relative SD 10% or 30%), unbiased: maximum | 0.050 | 0.050 | 0.050 | 0.050 |
| Bootstrap estimate 20% too small, noisy 30%, premise term dominant | 0.094 | 0.063 | 0.062 | < 0.05 |
| Bootstrap estimate 40% too small, premise term dominant | 0.148 | 0.093 | 0.094 | < 0.05 |
| Skewed seeds (exponential tail of bad seeds), premise SE 0 | ≤ 0.05 same shape both sides; **0.118** when one side is tight and the other skewed | 0.139 | **0.117** | **0.115** |
| Log-normal seeds, premise SE 0 | 0.173 (one side) | 0.192 | 0.180 | 0.162 |
| 15% of seeds "fail" (mixture), premise SE 0 | 0.295 (one side) | 0.245 | **0.318** | 0.198 |
| AUROC near 1 (Beta, mean 0.97, SD 0.03), premise SE 0 | 0.117 (one side) | 0.138 | 0.117 | 0.116 |
| Any of the non-normal shapes, seed SD ≤ 0.03, premise SE 0.011 | ≤ 0.037 | ≤ 0.037 | ≤ 0.037 | ≤ 0.037 |
| Same, premise SE 0.005, seed SD 0.03, one sample of 5 | | | 0.061 to 0.099 | |
| Full hierarchical simulation with a real premise bootstrap (seed effect + shared premise effect + seed × premise interaction; 187 premises, 200 resamples, 4,000 replications): maximum | 0.045 | 0.046 | 0.007 | 0.0005 |

Reading:

- **"At most 0.051 in every setting" is true of the settings in power.json and nothing else.** It holds for normal
  per-seed values. It fails (a) with unequal SDs when the premise term is negligible (0.066: power.json tested
  unequal SDs only at premise SE 0.01), (b) for skewed or bimodal per-seed values whenever the seed term dominates
  the premise term (0.12 to 0.32 for one-sample tests, 0.12 to 0.29 for two-sample tests with one tight and one
  skewed condition), (c) if the bootstrap underestimates the premise SE. **Should fix (condition 5, wording).**
- At the premise SEs the built sets give (0.011 to 0.02) and seed SD up to 0.03, I found no setting above 0.04, of
  any shape. The danger case is concrete, though: one LoRA seed of five that fails to train makes the seed SD many
  times the premise SE, which is the top-right corner of the table. The document has no rule for it.
- The test is **strongly conservative** where it will be used: 0.024, 0.019, 0.009 and 0.0004 at seed SD 0.02.
  This is the price of giving the whole SE the degrees of freedom of the seed terms. It is honest for the claims
  (power.json includes it) but it is the wrong direction for a **negative control**: see 2.4.

### 2.2 Is a shared, additive premise shift defensible?

Yes, and the error is on the safe side. Per-seed statistics computed on one premise sample contain the seed ×
premise interaction, so s² already carries it; the bootstrap variance of the seed-mean carries it again (divided by
n). SE² is therefore too large by (interaction variance)/n. My hierarchical simulation with an interaction and no
additive structure gives 0.001 to 0.046. Non-additivity costs power, not size.

### 2.3 "Fixed reference" (E1 ii, E5b, E5c) and within-seed differences (E4)

Coherent, provided `boot_D` is what the docstring says: D recomputed on each premise resample with the reference
rescored on the same resample (E1 ii, E5b, E5c), or the mean within-seed difference on each resample (E4, with the
five differences passed as `per_seed_x`). Simulated "against a fixed reference on the same premises": 0.005 (5
seeds), 0.000 (3). Three remarks:

- **No code computes `boot_D` yet.** There is no analysis script; `seedtest.test` only requires that 200 or more
  numbers are passed. `N_BOOT = 2000` is defined and never used. Whether a string's eligibility (≥ 3 per class) and
  the length residual are recomputed inside each resample is unstated. Should fix: one function, one fixture.
- E5c takes "the best of" three references. If the best is chosen on POOL-LB the comparison is conservative for the
  model; say so and say that the choice is made once on the full set, not per resample.
- The premise SE that power.json uses for E5b (0.0111) is that of two simulated models sharing half their noise.
  Measured with the real copy score: SE of S(copy from B) 0.0097, of S(copy from A) 0.0116; a simulated model at
  0.85 minus the copy from B 0.0120; a model at 0.70 minus the copy from A **0.0158**. E5b for A then has power
  0.48 at its registered effect, not 0.76 (Task 3).

### 2.4 E2c as a use of this test

S(Bswap) − 0.5 with 3 seeds has size 0.0004 to 0.001 and power 0.25 for a leak of 0.05 (the size of E2a's
registered effect), 0.93 for 0.10. A control whose pass is the uninformative outcome should not be run with the
most conservative configuration of the test. The document says "coarse control", which is honest, but the decision
that hangs on it ("E2a, E2b and E5 are not interpreted until it is explained") is taken by the test. With 5 seeds
the same leak is caught with probability 0.79 (measured). **Should fix:** 5 seeds for Bswap, or read E2c by its
interval against a stated bound (pass only if the 95% interval lies inside 0.5 ± 0.05), which 3 seeds cannot meet
and 5 can.

### 2.5 Holm over "seven tests"

"Seven tests are registered (E2a, E2b, E3, E4a, E4b, E5a, E5b …)". The tests actually listed:

- E5 is run "for A, A48, B separately"; A48 has "no test". That leaves E5a(A), E5a(B), E5b(A), E5b(B): four tests,
  counted as two. The family is **nine**, not seven.
- E5c is "tested once" and outside the family, yet L3 rests on it and "a rung of the ladder is claimed on the
  Holm-adjusted value". E5c has no Holm-adjusted value. For which models is it tested?
- L0 and L1 in Section 0 rest on "S on POOL-LB above 0.5" with experiment "E2". E2 registers no such test (E2a and
  E2b are differences, E2c is Bswap). Section 4 E5 defines L1 differently ("E5a claimed and E5b not → … (L1)").
  The two definitions of L1 disagree and one of them has no registered test.
- E1 (ii) and E2c use the test and are outside the family (a gate and a control): acceptable, stated.

**Blocking (condition 1 and 2):** the family and the ladder must name the same tests.

---

## Task 3. Power

**Does `power.py` simulate the registered test on the built sets?** The test: yes (it calls the registered
function). The built sets: the premise SE is a bootstrap over the real premises, strings and cells of POOL-LB, ID48
and ID with simulated scores. The scores have no length term (the residualisation is not exercised) and two models
are assumed to share half their item noise.

**Re-derivations (own code, 200,000 replications, premise SE as in power.json, estimated with 10% noise).**

| Quoted in PREREG | power.json | Mine |
|---|---|---|
| E2a 0.05: 0.93 / 0.71 / 0.48 | 0.931 / 0.713 / 0.475 | 0.930 / 0.714 / 0.471 |
| E2a 0.07: 1.00 / 0.95 / 0.78 | 0.999 / 0.953 / 0.777 | 0.999 / 0.953 / 0.777 |
| E2b 0.07: 0.97 / 0.79 / 0.55 | 0.973 / 0.788 / 0.549 | 0.974 / 0.789 / 0.548 |
| E2c 0.10: 1.00 / 0.93 / 0.71; 0.05: 0.45 / 0.25 / 0.17 | 0.999 / 0.926 / 0.715; 0.451 / 0.253 / 0.167 | 0.999 / 0.927 / 0.716; 0.458 / 0.256 / 0.168 |
| E3 0.10: 1.00 / 0.95 / 0.80 | 0.999 / 0.951 / 0.804 | 0.999 / 0.952 / 0.806 |
| E4 0.05: 0.79 / 0.64 / 0.48 | 0.789 / 0.641 / 0.478 | 0.791 / 0.643 / 0.479 |
| E5a 0.05: 0.94 / 0.79 / 0.57 | 0.937 / 0.785 / 0.574 | 0.936 / 0.785 / 0.577 |
| E5b 0.05: 0.91 / 0.76 / 0.56 | 0.912 / 0.759 / 0.557 | 0.912 / 0.760 / 0.556 |
| E1 (ii) 0.05: 0.94 / 0.79 / 0.58 | 0.940 / 0.792 / 0.578 | 0.939 / 0.790 / 0.577 |

All reproduce. Every registered effect meets the document's rule (power ≥ 0.6 at seed SD 0.02) **for the unadjusted
test**: 0.71, 0.79, 0.95, 0.64, 0.79, 0.76.

**But rungs are claimed on the Holm-adjusted p, and no power is given for that. Blocking (condition 1).** Power at
seed SD 0.02 of each test at its registered effect, at the level it may face (measured):

| Test, registered effect | α = 0.05 | 0.05/3 | 0.05/7 | 0.05/9 |
|---|---|---|---|---|
| E2a, 0.05 | 0.72 | 0.44 | 0.24 | 0.19 |
| E2b, 0.07 | 0.79 | 0.50 | 0.30 | 0.24 |
| E3, 0.10 | 0.95 | 0.75 | 0.56 | 0.50 |
| E4, 0.05 | 0.65 | 0.24 | 0.05 | 0.02 |
| E5a, 0.05 | 0.79 | 0.40 | 0.12 | 0.07 |
| E5b, 0.05 | 0.76 | 0.36 | 0.10 | 0.06 |

Joint simulation of Holm over the seven listed tests with every effect true at its registered size (independent
draws, seed SD 0.02): probability of being claimed E2a 0.30, E2b 0.35, E3 0.60, E4a 0.11, E4b 0.11, E5a 0.20, E5b
0.18; **E5a and E5b both (what L2 needs for one model) 0.07**. With every effect doubled: all ≥ 0.998. So the
registered effect sizes are the right ones for a test nobody will use for a claim, and about half the size the
claim rule can detect. The project's own standing order ("If power is below 0.6, redesign or drop the test")
applies to the rule that decides the claim. Smallest grid effects with power ≥ 0.6 at α = 0.05/9, seed SD 0.02
(measured): E2a 0.10 (0.94; 0.07 gives 0.56), E2b 0.15 (0.82; 0.10 gives 0.57), E3 0.15 (0.79), E4 0.10 (0.79),
E5a 0.10 (0.92), E5b 0.10 (0.90). These agree with the document's own predictions (B ≥ A + 0.10; A − C ≥ 0.15).

**Sensitivity to the premise SE (should fix).** With item noise independent between two models (instead of half
shared) my bootstrap on the real sets gives SE 0.0148 for a difference on POOL-LB (power.json 0.0111) and 0.019 to
0.021 on ID48 (power.json 0.0131). At seed SD 0.02, unadjusted: E2a 0.05 → **0.57**; E2b 0.07 → 0.68; E3 0.10 →
0.85; E5b for A (measured SE 0.0158 against the real copy) → **0.48**. Two registered tests are below 0.6 at their
registered effect under an assumption no less plausible than the one used. Moot if condition 1 is met by raising
the registered effects.

**Tests below 0.6 at the registered effect, as registered:** none unadjusted under power.json's assumptions; all
but E3 under the Holm rule; E2c at 0.05 (0.25), disclosed.

---

## Task 4. Numbers checked against result files and by recomputation

Recomputed from raw data or the stored sets unless marked (file).

| PREREG | Measured | |
|---|---|---|
| Train 205,734 items on 2,338 premises; test 1,162 on 21; dev 1,076 on 22; dev shares 2 with train, 3 with test | same; train ∩ test = 0 | ok |
| "2,358 premises of train-minus-dev plus test are 2,358 distinct graph classes" | 950 + 187 + 1,200 + 21 = 2,358 (file: 2,358 classes). But one of the two premises shared with dev **is a POOL premise** and is kept; only one is removed | minor wording |
| POOL 1,200; ID 187; TRAIN 950 premises; 82,458 TRAIN items | same; the 187 regenerate from the seed | ok |
| Valid rates 23.8% / 54.8% (POOL), 12.5% / 28.7% (TRAIN), 14.5% / 33.5% (ID) | 23.77 / 54.82, 12.45 / 28.69, 14.53 / 33.49 | ok |
| N = 7,184 for all five conditions | same | ok |
| A: 120 strings, 12.8% valid, 3,126 items in the 48 strings, 919 valid | 120, 12.79%, 3,126, 919 | ok |
| A48: 28.8% valid; B: 48 strings, 707 premises, half valid | 28.79%; 48, 707, 3,592 / 3,592 | ok |
| "2.3 times as many training items of the tested strings as A and 3.9 times as many valid ones" | 7,184 / 3,126 = 2.30; 3,592 / 919 = 3.91 | ok |
| B: valid = invalid in every string, premise, type × bin, string × bin; at most 8 pairs per premise | 0 unbalanced of 48 strings, 707 premises, 20 type × bin cells, 190 string × bin cells; max 16 items per premise; bins recomputed from the MY_PARTS spec on the 82,458 candidates | ok |
| POOL-LB: 4,706 items, 1,193 premises, at most 4 per premise, same balance | 4,706; 1,193 (1,160 premises with 4 items, 33 with 2); 0 unbalanced of 48 / 1,193 / 20 / 180 | ok |
| POOL-LB: 47 strings with ≥ 3 per class, "median 29 pairs per string" | 47 (one string has a single pair); median 29.5 (the report truncates) | minor |
| ID 2,000 items, 15.3% valid; ID48 2,000, 34.8% valid, 47 strings with ≥ 3 per class; 187 premises each | 15.35%; 34.85%; 47; 187 and 187 | ok |
| 93 swap strata of 10 to 19 | 93, 10 to 19; all donors in stratum | ok |
| C label correct 85%; Bswap 49.9% | 85.26%; 49.86% (within-string correlation of kept and true label: C 0.19, Bswap −0.004) | ok |
| Hypothesis-string prior fitted on A: pooled AUROC 0.885 on ID, F1 0.56 | 0.8854; 0.556 | ok |
| Length alone, raw within-string AUROC 0.758 on ID48 | 0.758 over all two-class strings; 0.756 over the 47 with ≥ 3 per class | ok |
| Shortcut checks 0.500 / 0.500 / 0.519 [0.499, 0.538] / 0.507 [0.491, 0.521] / 0.515 / 0.509; on A → ID 0.885, 0.676, 0.726, 0.922 | (file) same. The two 0.500s are "constant score": see Task 5.3 | ok as numbers |
| Nearest premise: median 4 statements (quartiles 3 and 8) of about 33 | 4 (3, 8); median 33 statements in POOL-LB premises | ok |
| **Nearest-premise copy: S on POOL-LB 0.805 from B, 0.660 from A48, 0.641 from A; S on ID48 0.678 from A** | With S as Section 3 registers it (strings with ≥ 3 per class): **0.822, 0.674, 0.633; 0.673**. The quoted values are over all 48 strings, including the one-pair string, where the AUROC is 0 or 1: `build_data.py` filters on `x in two`, not on class counts | **mismatch** |
| "score = the label of the string there" | `nearest_premise_scores` returns label ∓ 1e-4 × distance (a graded score: a closer valid neighbour ranks higher). With the label alone: 0.821, 0.674, 0.632; 0.672. Small here, but not what the text says | mismatch (wording) |
| E5 cells: 70 of 95, 4,139 items, 43 strings; adjacency alone S = 0.498 | 70 of 95; 4,139; 43; 0.498 (0.514 without length removal) | ok |
| Later-letter-first directional hypothesis never valid in train | 0 valid of 68,582 (child 34,291; ancestor 17,157; descendant 17,134); also 0 in ID, pool and test premises | ok |
| Authors' files 1,884 and 1,522 distinct inputs; our transforms reproduce 1,162 / 1,162 | (file) same; sets materialise with matching hashes | ok |
| 60 train × eval pairs, zero overlap | (file) 60 pairs, largest overlap 0 | ok |
| 1,347 steps; 19 models | 7,184 × 3 / 16 = 1,347; 5 + 3 + 5 + 3 + 3 = 19 | ok |
| Premise SE 0.011 (POOL-LB difference), 0.013 (ID48), 0.011 (E1) | (file) 0.0111, 0.0131, 0.0105 | ok as file; see Task 3 |
| False-positive rate "at most 0.051" | (file) 0.05092 at n = 5 one-sample, premise SE 0 | ok as file; see Task 2 |
| Every power figure in Section 4 | (file) all match to rounding | ok |

Mismatches: the four nearest-premise S values; the description of that score; "median 29"; "train-minus-dev".

---

## Task 5. Design: what can climb the ladder unaccounted for

### 5.1 Leaks between TRAIN and ID or POOL premises

None that the overlap check misses in the sense of identical premises: premise text, text with renaming undone and
canonical graph class are all disjoint for all 60 pairs, and every item of A, A48, B and every donor and own
premise of C and Bswap is a TRAIN premise (measured). What remains is near-duplication, which the document states:
POOL-LB premises are a median of 4 statements from the nearest TRAIN premise, 24% are within 2 statements; ID
premises a median of 5, 23% within 2; TRAIN premises among themselves a median of 5 (so ID is as close to TRAIN as
TRAIN is to itself: a fair "in distribution"). The first review asked for S on a subset at a stated distance from
training premises; v2 neither adopts nor answers this. **Should fix:** report S by tercile of distance to the
nearest premise of the model's training set (descriptive).

### 5.2 The nearest-premise reference: as described? strong enough? **Blocking (condition 3)**

Implemented as described except for the graded score (Task 4). It is the weakest member of its family. S with the
registered definition, measured, no model, each computed from the condition's own 7,184 training items:

| Reference | B → POOL-LB | A48 → POOL-LB | A → POOL-LB | A → ID48 | A48 → ID48 |
|---|---|---|---|---|---|
| k = 1 (registered) | 0.822 | 0.674 | 0.633 | 0.673 | 0.678 |
| mean label of k = 3 | 0.867 | 0.729 | 0.661 | 0.705 | 0.790 |
| k = 5 | 0.869 | 0.738 | 0.669 | 0.734 | 0.786 |
| k = 9 | 0.872 | 0.723 | 0.680 | 0.728 | 0.793 |
| all neighbours, weight exp(−distance/4) | 0.891 | 0.744 | 0.696 | 0.761 | 0.807 |
| k = 1, distance counted only on statements naming a queried variable | 0.846 | 0.738 | 0.675 | 0.661 | 0.712 |
| same, weight exp(−distance/2) | **0.905** | **0.804** | **0.745** | **0.772** | **0.814** |

Consequences:

- **L2 as written can be reached by memorising.** E5b is registered for 0.05 above the k = 1 copy. A three-neighbour
  vote is 0.045 above it for B, 0.055 for A48; a distance-weighted vote 0.07 to 0.13 above. A model that
  interpolates among training premises passes E5b, and (the document says so itself) a lookup passes E5a. The rung
  "above the nearest-premise copy" then does not mean "beyond copying neighbours".
- E1: "0.80 is 0.12 above the nearest-premise copy". From A's own training set a weighted copy scores 0.76 to 0.77
  on ID48 and a per-string logistic regression on v1.0's 16 structural features scores 0.773. The gate is about
  0.03 above the strongest non-causal reference I built in an hour, with a premise SE of 0.016 for one model.
- The same family explains E2 without any reading of structure: weighted copy B 0.905, A48 0.804, A 0.745. The
  document says this for k = 1; it is larger than stated.

None of these variants is tuned on anything that a pre-registration could not fix now: register the family and take
its maximum on the evaluation set (conservative for the model).

### 5.3 What Bswap and the shortcut checks can and cannot detect. **Blocking (condition 4)**

Section 1: "premise surface statistics (logistic regression on variable count, numbers of correlation, marginal and
conditional independence statements, log length): 0.500 both ways" and "a balancing function can leave or create
others, shared by B and POOL-LB and absent from A. Bswap is the control for that."

- The 0.500 is not a finding. Both sets are exactly half valid inside every premise, so a classifier of premise
  features alone has zero gradient and returns a constant (the report says "constant score"). The check cannot fail.
- The same four counts **inside a hypothesis string** are not uninformative. Per-string classifier on the four
  counts (no binding of the hypothesis to any statement), S on POOL-LB with length removed, measured:

  | Fitted on | logistic | gradient boosting |
  |---|---|---|
  | B | **0.587** [0.561, 0.610] | **0.675** [0.650, 0.691] |
  | A48 | 0.528 [0.508, 0.547] | 0.576 [0.555, 0.595] |
  | A | 0.513 | 0.563 |
  | all 35,789 TRAIN items of the 48 strings | 0.536 | – |
  | Bswap | 0.527 [0.502, 0.551] | 0.501 [0.476, 0.525] |
  | POOL-LB, tested on B | 0.577 | – |

  B minus A48: 0.059 [0.028, 0.085] (logistic), **0.099 [0.067, 0.123]** (boosting). The regularity is specific to
  sets drawn by the balancing function: the classifier fitted on B scores 0.556 on natural items of the same pool
  premises, where classifiers fitted on natural data score 0.668. So B and POOL-LB share a string × premise-count
  relation that A and A48 do not have, worth 0.06 to 0.10 of S: the size of E2a's and E2b's registered effects.
- **Bswap cannot see it.** The swap replaces the premise, so every relation between premise content and label is
  destroyed (structural-feature classifier fitted on Bswap: 0.495). Bswap controls what survives a swap inside a
  stratum: the hypothesis string, the number of variables, and premise length at the resolution of 93 strata. Those
  are already balanced by construction. E2c passing says almost nothing about artefacts of the balancing function
  that live in the premise; the sentence "Bswap is the control for that" is wrong, and the label of POOL-LB's use
  ("S where wording, premise and length carry nothing") is wrong for premise counts within string.
- E2's own outcome sentences are descriptive and survive ("No outcome of E2 says what B learned"). What does not
  survive is the reading of E2c and the absence of this reference from the ladder: it belongs with the text
  references of E5c (v1.0's structural features include the counts, so a per-string or interaction-capable stack
  would contain it; the registration should say so explicitly and give the number).

### 5.4 S: the ≥ 3 restriction and the length residual across build_data.py, power.py and the text

Not consistent. Text: restricted to strings with ≥ 3 per class. `power.py`: restricted (`min_class=3`), re-decided
in every resample, no length term in the simulated scores. `build_data.py` and `within_string_reference.py`: not
restricted (any string with both classes). Effect on quoted numbers: up to 0.017 (Task 4). The residual itself is
applied identically wherever S is computed. One property worth stating in E1 and E3: on ID48 length is truly
related to validity (raw 0.758), so removing it also removes signal from a model that tracks validity. S of the
gold label itself is 0.981, and of the gold label plus noise 0.852 where the raw within-string AUROC is 0.924. The
gate at 0.80 is harder than a raw 0.80. Conservative; say it.

### 5.5 E4b: does the order-preserving renaming isolate order from names?

Largely yes. Checked: the alphabetical regularity holds without exception (Task 4); all 1,171 directional items of
ID48 (58.6% of the set) have the earlier letter first, none in the reversed renaming, all in the order-keeping
one; both renamings use the same six letters and give identical token counts item by item, so the length residual
is unchanged. One confound remains: the reversal also inverts the premise's own conventions (statements are
written earlier-letter-first and enumerated alphabetically in every training premise; reversed they read "Z
correlates with X"). E4b therefore measures "letter order" (the direction regularity or the premise's ordering
convention), not specifically "the alphabetical-order regularity" of the outcome sentence. **Should fix
(wording).** Under Holm E4 has power 0.02 to 0.24 at its registered 0.05 (Task 3).

### 5.6 Anything tuned on evaluation data?

- The number of length bins (4) and pairs per premise were fixed after builds in which the B ↔ POOL-LB transfer
  checks were the criterion (MY_PARTS.md still mentions a 16-bin build). Labels of evaluation items, no model: fine.
- E5c "best of" three references: chosen on POOL-LB unless stated otherwise; conservative for the model.
- F1 threshold: training set. Stack settings: cross-validation inside the training set. Training recipe: fixed.
- Nothing else found.

### 5.7 Is the E1 gate meaningful now?

Part (i), yes: no reference that ignores the binding of hypothesis to premise comes near 0.80 (premise counts per
string 0.586). Its margin over memorisation and lookup is about 0.03, not 0.12. Part (ii) adds nothing: it is
passable by string prior + length or premise counts (0.922 to 0.937 against 0.885). **Should fix:** score (ii)
against the string-and-length classifier (0.922) or drop it, and correct the "0.12 above". The gate decides only a
label, so this does not block training.

---

## Task 6. Tests, and the balancing reference on the stored sets

`pytest finetune/tests -q`: **41 passed, 18 skipped** (the skips are the owner's unimplemented parts).

Coverage:

- Registered test: arithmetic against `scipy` (Welch and one-sample), the premise term, the required resamples,
  and size at four normal settings with the premise variance known exactly. Not covered: unequal SDs, non-normal
  values, the construction of `boot_D`, Holm.
- Balancing: properties P1 to P5, determinism, edge cases, and the length leak, all on synthetic data. Nothing
  runs on the stored sets.
- S: perfect, inverted, length removal, ties, one-class strings. Not covered: the ≥ 3 restriction.
- No test for `nearest_premise_scores`, `swap_strata`, `graph_class` or `shortcut_transfer`.

Reference balancing function on the stored sets (measured, bins recomputed independently from the spec):

- B and POOL-LB: valid = invalid in every string (48), every premise (707; 1,193), every type × bin (20), every
  string × bin (190; 180), every premise × type; no premise beyond its cap (16; 4); no duplicate items; no string
  twice in a premise; all strings two-label among the candidates. **No bug found; the sets are as described.**
- Not a bug, worth a line in the text: the cap is shared over types and types are processed alphabetically, so with
  2 pairs per premise POOL-LB is 67% collider and confounder items and 17% "parent", while B (8 pairs) is 62%
  "parent". S is an unweighted mean over strings, so this does not move S, but pooled numbers on the two sets are
  not comparable. One string of POOL-LB has a single pair (hence 47).

---

## Task 7. Verdict and findings by severity

**PASS WITH CONDITIONS.** Exact edits required before the first training run:

1. **Multiplicity and power agree (blocking).** In Section 3 list the family as it is run: E2a, E2b, E3, E4a, E4b,
   E5a(A), E5a(B), E5b(A), E5b(B) (nine), plus whatever condition 2 adds. Then either (a) register each test for
   the smallest grid effect with power ≥ 0.6 at α = 0.05/9 and seed SD 0.02, computed by `power.py` with an alpha
   argument (my figures: E2a 0.10, E2b 0.15, E3 0.15, E4 0.10, E5a 0.10, E5b 0.10), or (b) replace Holm by a fixed
   order of testing stated in the text. Not both; no other change.
2. **One definition of each rung (blocking).** Make Section 0 and E5's outcome paragraph say the same thing, and
   give every rung a registered test: either add "S(model) − 0.5 on POOL-LB, 5 seeds" for A and B to the family
   (L0/L1) or define L1 by E5a and delete "S on POOL-LB above 0.5 | E2". State for which models E5c is tested and
   whether L3 is claimed on an adjusted or unadjusted p.
3. **The memorisation reference (blocking).** (a) Replace the four quoted values by the registered S: 0.822 (B),
   0.674 (A48), 0.633 (A) on POOL-LB; 0.673 (A) on ID48; or say they are over all 48 strings. (b) Describe the
   score as coded (label, ties ordered by distance). (c) In E5b and E1 define the reference as the largest S among
   a fixed family computed from the model's own training set: k ∈ {1, 3, 5, 9}; all neighbours weighted
   exp(−d/τ), τ ∈ {1, 2, 4}; distance over all statements and over statements naming a queried variable. Quote its
   values (0.905 / 0.804 / 0.745 on POOL-LB; 0.772 from A on ID48) and correct "0.12 above" in E1.
4. **What the checks and Bswap show (blocking).** In Section 1: say that the premise-only check is constant by
   construction; add the within-string premise-count classifier with its values (fitted on B 0.587 logistic, 0.675
   boosting; A48 0.528, 0.576; Bswap 0.527, 0.501); replace "Bswap is the control for that" by "Bswap controls only
   what survives the swap: hypothesis string, number of variables, premise length"; change POOL-LB's use to "S
   where wording, the premise alone and length carry nothing". In E5c name the per-string premise-count classifier
   as one of the references. In E2c delete the implication that a pass clears B and POOL-LB of shared artefacts.
5. **Size statement (blocking, one sentence).** Section 3: "at most 0.051" holds for normally distributed per-seed
   values in the settings of power.json; add the measured exceptions (0.066 for 5 against 3 with unequal SDs and no
   premise term; 0.12 to 0.32 for skewed or bimodal seed values when the seed term dominates) and a rule fixed now
   for a seed that fails to train (for example: every per-seed value is reported; if one seed lies more than a
   stated distance from the others the test is reported with and without it and nothing is claimed unless both
   agree).

Should fix (not required before training):

- E2c: 5 seeds for Bswap or an interval-against-bound reading; at 3 seeds the test has size 0.001 and power 0.25
  for a leak the size of E2a's effect.
- State the premise-SE assumption (half-shared item noise) and the sensitivity: E2a 0.05 → 0.57, E5b(A) → 0.48
  with independent noise or the real copy score.
- Put the ≥ 3-per-class restriction into the S function or a wrapper with a fixture; state whether eligibility and
  the residual are recomputed per resample; write the function that builds `boot_D` before scores are opened.
- E1 (ii): compare with string-and-length (0.922) or drop it.
- S by distance to the nearest training premise (the first review's request, unanswered).
- E4b outcome sentence: "letter order", not specifically the direction regularity.
- Say in E1 and E3 that length removal on ID48 lowers S of a true validity-tracker (0.924 raw → 0.852).
- A test that runs the balance properties and the overlap check on the stored sets.

Minor:

- "median 29 pairs" is 29.5; "train-minus-dev": one premise shared with dev is a POOL premise and is kept.
- E5 outcome "E5a not shown, E5b claimed" is unassigned; the E3 ratio's interval method is unspecified.
- `N_BOOT` is unused; `seedtest.test` accepts 200 resamples where the text says 2,000.
- POOL-LB's type mix (67% collider and confounder) differs from B's (62% parent).

## What is fine (one line each)

- Every count, rate, cell number and power figure quoted, except those in the mismatch list: reproduces.
- The random ID split regenerates from its seed; ID is as close to TRAIN as TRAIN is to itself.
- B and POOL-LB satisfy every stated balance property and cap on the stored sets.
- C and Bswap are built as described; Bswap's labels are unrelated to the truth on the new premise (−0.004).
- The alphabetical-order claim is exact; the two renamings differ only in order and have identical token counts.
- The registered test is valid (conservative) for normal seed values at every premise SE the sets can give.
- `power.py` calls the registered function; its arithmetic reproduces to the third decimal.
- No training hyper-parameter, set or threshold is tuned on evaluation data.

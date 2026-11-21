# Independent audit: behaviour analysis E1 to E5 (raw)

Auditor: an independent reviewer, given code and result files only. Date 21 Nov 2026. Branch `exp/finetune`, HEAD `fef2993`. No
existing file was edited. Scratch scripts are in the session scratchpad (`audit/recompute.py`, `compare.py`,
`tau_sensitivity.py`, `variant_tests.py`, `textref_repro.py`, `ties.py`, `diag_boot.py`).

**VERDICT: PASS WITH CONDITIONS.** Every registered number in `behaviour_registered.json` reproduces from my own code:
S per model, the eleven D's, the premise SEs, the Welch df, p, Holm p, and the claims. I found no bug that changes a
number or a claim. The conditions concern what the results page says. The three that matter most: (a) A's "L2" must
carry the Amendment-1 qualifier in the same sentence, and it is weaker than that qualifier suggests (finding 9). (b)
Four registered items are missing from the result files and must be added or disclosed as deviations (findings 5, 6,
7). (c) Several fields in the JSON invite misreporting (finding 11).

No finding is BLOCKING.

---

## What I ran

1. `recompute.py`: my own code. It does not import `ftstudy.behaviour` or `ftstudy.stat`. It rebuilds the sets with
   `datasets.materialise` (hash-checked). It parses premises with the v1.0 `probes.parse_premise`, not
   `datasets.parse_premise`. It computes:
   - S: `np.polyfit` quadratic in centred log tokens per string, then `sklearn.roc_auc_score`, on strings with ≥3 per
     class;
   - S in cells, with my own adjacency;
   - my own memorisation family (21 rules: all / pair-or-conditioning-set / pair-only distance), A, B, A48 on POOL-LB;
   - my own premise resamples (2,000, rng seed 0), with eligibility, the length residual and cells recomputed on each
     resample, for all models and references on the same resample.
2. `compare.py`: my own assembly of the test (D, SE² = seed terms + v_premise, Welch df, two-sided t, Holm over the
   eleven, the outlier rule), compared with the JSON.
3. Sensitivity scripts (not registered). The copy family at τ = 8, 16, 32, ∞. A against the pair-only rules at τ = 2
   and 8 with the full test. Text reference reproduction and C = 1000. Tie diagnostics.

---

## Findings

### 1. NOTE: Independent recomputation matches to machine precision (items 1 and 2 of the brief)

- **S per model.** max |S_json − S_mine| over all 21 models × {POOL-LB, ID48} = **2.2e-16**.
- **E1.** Mean S(A) on ID48 0.84556 and mean pooled AUROC on ID 0.98032, both identical to the JSON.

Registered tests. D was identical to 5 decimals in every row, so it is listed once.

| Test | D | SE (mine / JSON) | Premise SE (mine / JSON) | df | p (mine / JSON) | Holm (mine / JSON) |
|---|---|---|---|---|---|---|
| E2c | −0.00416 | 0.00701 / 0.00704 | 0.0066 / 0.0066 | 2.00 | 0.6136 / 0.6148 | n/a |
| E2a | +0.13785 | 0.00983 / 0.00983 | 0.0071 / 0.0071 | 6.62 | 3.6e-6 / 3.6e-6 | 0.0000 / 0.0000 |
| E2b | +0.07666 | 0.01415 / 0.01415 | 0.0064 / 0.0064 | 2.35 | 0.02255 / 0.02254 | 0.0676 / 0.0676 |
| E3 | +0.32827 | 0.02381 / 0.02381 | 0.0227 / 0.0227 | 6.00 | 9e-6 / 9e-6 | 0.0001 / 0.0001 |
| E4a | +0.07366 | 0.00995 / 0.00995 | 0.0083 / 0.0083 | 4 | 0.00178 / 0.00178 | 0.0124 / 0.0124 |
| E4b | +0.00064 | 0.00606 / 0.00606 | 0.0054 / 0.0054 | 4 | 0.9214 / 0.9214 | 0.9214 / 0.9214 |
| E5a_A | +0.17893 | 0.01507 / 0.01507 | 0.0090 / 0.0090 | 4 | 0.00029 / 0.00029 | 0.0023 / 0.0023 |
| E5a_B | +0.36618 | 0.01166 / 0.01166 | 0.0079 / 0.0079 | 4 | 1e-5 / 1e-5 | 0.0001 / 0.0001 |
| E5b_A | +0.07465 | 0.01221 / 0.01221 | 0.0107 / 0.0107 | 4 | 0.00362 / 0.00362 | 0.0176 / 0.0177 |
| E5b_B | +0.01832 | 0.00880 / 0.00875 | 0.0081 / 0.0080 | 4 | 0.1057 / 0.1044 | 0.2115 / 0.2088 |
| E5c_A | −0.06776 | 0.00970 / 0.00970 | 0.0078 / 0.0078 | 4 | 0.00221 / 0.00221 | 0.0133 / 0.0132 |
| E5c_B | −0.03981 | 0.00646 / 0.00647 | 0.0054 / 0.0054 | 4 | 0.00352 / 0.00353 | 0.0176 / 0.0177 |

**Claims.** Every claim and direction is the same: above for E2a, E3, E4a, E5a_A, E5a_B and E5b_A; below for E5c_A
and E5c_B; not shown for E2b, E4b and E5b_B. E2c is "pass" (interval [−0.034, +0.026] lies inside ±0.05). The rungs
are A L2 and B L1.

**Where the small SE differences come from.** My resample indices are identical to `stat.boot_indices(groups, 2000,
0)` (checked). The residual differences come from near-ties. Bswap's POOL-LB scores take only 55 distinct float32
values, so the closed-form and lstsq residuals can order two near-equal items differently. This happened on 1 of the
first 300 resamples for Bswap_s0, by 0.0104, and on 0 of 300 for B_s0. Rounding residuals to 1e-9 leaves every
point S unchanged (`ties.py`). This is immaterial.

**The test is implemented as registered.**
- D is the mean of the per-seed statistic, minus the other mean, a constant or the reference's S.
- SE² = s_x²/n_x + s_y²/n_y + var(D over 2,000 resamples, ddof = 1).
- df is the Welch–Satterthwaite value of the seed terms only (n − 1 for one-sample and paired tests).
- p is two-sided t.
- Holm runs over exactly the eleven named tests. A missing test counts as p = 1. E2c and E1 are excluded.
- The outlier rule (|x − median| > 0.10, reported with and without, claimed only if both survive Holm) is
  implemented. It is triggered by no test: I recomputed the outliers myself and there are none.
- Every model and reference is rescored on the same resample, and eligibility, cells and the length residual are
  rebuilt on each resample (`Engine.precompute` builds a new `Plan` per resample). This is confirmed independently,
  because my own implementation gives the same premise SEs.

### 2. NOTE: Seeds, sets, references and directions (item 3)

**Seeds.**
- Registered seeds: A s0–4, B s0–4, A48 s0–2, C s0–2, Bswap s0–2.
- Bswap_s3 and Bswap_s4 are listed in `extra_seeds_not_in_registered_tests` and do not enter E2c.
- With them (not registered), D = +0.0006 [−0.019, +0.021]: the same reading.
- `run.json` of every model gives the right condition, seed and training-set sha256, steps 1347/1347, the same
  versions, and a T4. The seeds 3–4 of A and B were run at commit 923aa39 and seeds 0–2 at 6cca06f, a platform-detection
  fix only; their S values do not stand apart.

**Sets and directions.**
- E2a and E2b: B − A and B − A48 on POOL-LB.
- E3: A − C on ID48.
- E4a: original − reversing renaming. E4b: keep-order − reversing. Both within seed, A, ID48, original string
  identity.
- E5a: cells − 0.5. E5b: S − memorisation reference. E5c: S − best text reference. All on POOL-LB.

**E5a.**
- Adjacency recomputed by me from `probes.parse_premise` (pair correlated and never independent) equals v1.0
  `structural_features[:, 2]` on all 4,706 items.
- Cells: **70 of 95, 4,139 items, 43 strings**, exactly as registered.
- Weighting is n1·n0 with ≥3 per class, and the residual is fitted within the whole string, as in v1.0
  `15_answer_bias.py`.
- Adjacency alone gives S = 0.4975 (registered 0.498).

**E5b.**
- The reference is the registered member `queried|exp|4.0` of the model's own training set.
- My independent family gives exactly 0.70335 (A), 0.89753 (B) and 0.75675 (A48), equal to `references.json`.
- The other 13 members also agree with the stored values to 4 decimals.

**E5c.**
- The reference is `tuned_stack` for A and B, the largest of the five stored references. My S of all five stored
  vectors equals `text_references.json`.

### 3. NOTE: E1 gate (item 4)

- (i) Mean S(A) on ID48 = 0.846 ≥ 0.80. Per seed: 0.831, 0.855, 0.831, 0.856, 0.854.
- (ii) Mean pooled AUROC on ID = 0.980 ≥ 0.922.
- Gate passed, recomputed.

Two small points:
- The code's constant 0.922 is the rounded value. `references.json` has 0.92151 (string and length, fitted on A).
  This is immaterial.
- "Passing E1 means answers correctly in distribution, not otherwise than by copying" (PREREG) still applies. See
  finding 6 for how close a text reference comes on ID48.

### 4. NOTE: E2c reading and the rung logic (item 4)

**E2c.** `e2c_reading` is correct.
- It uses the unadjusted p, because E2c is not in the family.
- "pass" requires the D interval to lie inside ±0.05.
- "inconclusive" is used otherwise, unless the interval excludes 0.

The amendment expected "inconclusive"; the observed result is "pass", because the seed SD is only 0.004.

**`rung`** follows Section 0:
- L0 unless E5a is claimed above.
- L1 if E5b is not claimed above.
- L3 if E5c is claimed above, otherwise L2.
- E5b-without-E5a is assigned, and "claimed below" never counts as a rung.

B = L1 is correct. A = L2 is correct as registered, with the Amendment-1 qualifier attached (see finding 9).

### 5. SHOULD-FIX (deviation, disclose): E1's "F1 at the threshold maximising training-set F1" is not reported

The JSON says "not computed: training-set scores were not stored by ft_run.py". This is a registered "reported
beside it" item, and it cannot be recovered without re-scoring the training sets with the adapters. The page must
list it as a deviation from PREREG Section 4 (E1). It does not affect the gate.

### 6. SHOULD-FIX (registered item missing): E3's "text references fitted on A's training set" are not in the result files

PREREG E3 asks for the text references fitted on A's training set to be reported beside E3. The values exist in
`itemsets/text_references.json` (A → ID48: lookup 0.628, hand parse 0.777, tuned stack **0.831**, premise counts
0.590 / 0.584). They are not carried into `behaviour_registered.json` or the tables.

They matter for the reading. In distribution, A's S on ID48 (0.846) is only **0.015** above a tuned text stack
fitted on A's own training set (0.831). Without that row, "A uses the content of the item's own premise" (E3) reads
stronger than it is. The page should print these four values beside E3. No interval is needed; they are descriptive.

### 7. SHOULD-FIX (registered item missing): E5 for A48 omits E5c

PREREG E5: "tested for A and for B … reported for A48 with intervals, no test". `analyse` reports A48's S in cells
and S − memorisation reference, but not S − best text reference. `refs["text_best"]["A48"]` is loaded and unused.

From the stored values: mean S(A48) on POOL-LB = 0.839, against A48's tuned stack 0.885, so D ≈ −0.046 (point only,
my computation). Add it with its interval.

Also note that A48's seed SD for S in cells is **0.036**, above the 0.03 that Section 3 calls underpowered. It is
descriptive, but it should be stated beside the A48 interval.

### 8. NOTE (deviation, minor): E5 near/far halves

PREREG: "the half of POOL-LB whose premises are nearer than the median distance (4 statements) to a training
premise". The code uses each condition's own median over POOL-LB premises (A 4, **B 5**, A48 4) and a strict "<".
The result is "halves" of 2,014 / 2,692 items (A) and 2,266 / 2,440 (B), not halves.

This is defensible, but it is not literally the registered split. The page should state the medians and the item
counts (the tables do give the counts). It is descriptive only.

### 9. SHOULD-FIX (wording): A's L2 against the pair-only variant, and the τ grid

**The pair-only variant: which numbers the definition gives (brief item 4).**
- I implemented the reviewer's variant from Amendment 1 independently. A statement counts for the "queried" distance
  only when a queried variable is in its **pair**; the "all statements" rules are unchanged.
- The seven pair-only members on POOL-LB (S):

| Rule | A | B | A48 |
|---|---|---|---|
| k = 1 | 0.6756 | 0.8471 | 0.7415 |
| k = 3 | 0.7082 | 0.8853 | 0.7776 |
| k = 5 | 0.7042 | 0.8870 | 0.7862 |
| k = 9 | 0.7216 | 0.8735 | 0.7760 |
| exp, τ = 1 | 0.7224 | 0.8929 | 0.7919 |
| exp, τ = 2 | **0.7453** | **0.9052** | **0.8043** |
| exp, τ = 4 | **0.7693** | **0.9126** | **0.8156** |

- **0.745 / 0.905 / 0.804 is exactly the single member pair|exp|τ=2**, the one row the reviewer's table lists. The
  reviewer's k = 1 pair-only row (0.675 / 0.846 / 0.738) also matches mine.
- **0.769 / 0.913 / 0.816 is the maximum of the family with the pair-only distance**. That maximum is pair|exp|τ=4
  in all three conditions, and the same holds for 21 rules, since the registered pair-or-conditioning rules are
  lower.
- **Why they differ:** member choice, nothing else. There is no tie-breaking or candidate difference.
- **Which one the definition gives:** the registered "memorisation reference" is "the rule with the largest S" of
  the family. Applying the same rule to the variant gives **0.769 / 0.913 / 0.816, which is what the code does**.
  The amendment's figures are one member that the reviewer happened to tabulate.

**Why the choice matters.**
- Against the code's variant (0.769): A, D = +0.009 [−0.023, +0.041], p 0.49, not above.
- Against τ = 2 alone (0.745), my full test: A, D = +0.033 [+0.000, +0.065], premise SE 0.0102, **p = 0.049**.
- So under the amendment's literal numbers the "not above the variant" sentence would not be triggered at unadjusted
  α. The code's choice is the conservative and better-justified one. The page should name the member (pair-only,
  exp, τ = 4) and say that the amendment's 0.745 / 0.905 / 0.804 are its τ = 2 member.

**The registered τ grid is at its edge (sensitivity, not registered).**
- In every condition on POOL-LB the best copy rule is the largest τ in the grid (4), for both the registered and the
  pair-only distance.
- Extending τ:

| Rule | A | B | A48 |
|---|---|---|---|
| registered distance, τ = 8 | 0.723 | 0.901 | 0.767 |
| registered distance, τ = 16 | 0.727 | – | – |
| pair-only distance, τ = 8 | **0.778** | 0.908 | 0.813 |

- A's mean S on POOL-LB is **0.778**. Against pair-only τ = 8: D = −0.0003 [−0.033, +0.032], p 0.98.
- So a distance-weighted copy of the pair-relevant statements of A's own training premises scores **as high as A**.
- This does not change the registered result (E5b_A is claimed, and the grid was fixed before the scores). But "A is
  at L2" must not be written as "A goes beyond copying from training premises". The honest sentence is: "A is
  claimed above the registered memorisation reference (+0.075), but not above the pair-only variant (+0.009, p
  0.49); a copy rule just outside the registered grid matches A exactly."
- For B the extension changes nothing: B is L1, and the best rule is 0.913 at τ = 4.

### 10. NOTE: E5c text references (item 5)

**They match the PREREG description.**
- Lookup: v1.0's per-template `HistGradientBoosting` on `structural_features[:, 0:6]`, same parameters as v1.0
  `05b_probe_content.by_key_gbm`.
- Hand parse: a GBM on all structural features.
- Tuned stack: hand parse plus a per-hypothesis-string TF-IDF 1–4-gram logistic model. The n-gram C ∈ {1, 10, 100} is
  chosen by the out-of-fold accuracy of the stack over premise-grouped 5-fold `StratifiedGroupKFold` inside the
  training set.
- Premise-count classifiers: logistic and boosting, per string.

**Nothing is fitted on or tuned with POOL-LB or ID48.**
- The TF-IDF vocabulary, GBMs, logistic models, stacker and C are all fitted on training items only.
- POOL-LB is used only for the registered "best of five, chosen on the full set", in the reference's favour.
- The file was committed 20 Nov (abf1a9b), before A_s4 and B_s4 were ingested (fef2993, 21 Nov). No analysis code
  changed in fef2993 (checked with `git show`).

**Reproduction.** Rerunning the committed code for A → POOL-LB reproduces the stored tuned-stack vector exactly (max
|diff| = 0), with C = 100 and CV accuracies 0.904 / 0.917 / 0.926.

**C = 100 is the top of the grid in all three conditions.**
- The CV accuracy rises monotonically in C, so the grid may cap the stack. I tried C = 1000
  (outside the grid): CV accuracy 0.9273 (vs 0.9264) and S on POOL-LB 0.8471 (vs 0.8458), +0.001.
- A stronger reference would only make E5c_A and E5c_B more negative, so the "below" claims are not threatened.
- State it as a property of the registered reference.

### 11. SHOULD-FIX: JSON and table fields a write-up could misreport (item 7)

- **E2b.** `p_two_sided` 0.0225 and `below_alpha: true`, but Holm p 0.068 means **not shown**. The registered
  reading (E2 table, row 2) is: "B is above A; the difference is not shown to be more than exposure to the tested
  strings". Do not quote p = 0.02 or "significant".
- **E5_A48** (descriptive). S − memorisation has `below_alpha: true` (p 0.033) and S in cells p 0.007. There is no
  test and no family. Report intervals only.
- **E5b_A.** "above" in the test table. The qualifier lives only in `rung_A` and in a separate line ("against the
  reviewer's pair-only variant"). Per Amendment 1 it must be in the same sentence, and see finding 9.
- **E4b.**
  - D = +0.001 [−0.016, +0.017]. Both renamings change the names; only one reverses order.
  - So the E4a drop (0.074) is attributable to the new letters, not to the letter order.
  - The registered prediction "E4b at least half of E4a" is refuted: the interval excludes 0.037.
  - The outcome sentence is the registered "not shown to be more than the names".
  - Do not write "order plays no role" as a proven absence. The interval bounds it at about ±0.017.
- **Failed predictions must be stated.**
  - E2: S(A) was predicted 0.50 to 0.65; observed mean **0.778**.
  - E4: E4b ≥ E4a / 2 was predicted; refuted.
  - E3 (C's AUROC ≥ 0.85; S(C) near 0.5; S(A) − S(C) ≥ 0.15) and E5 (B passes E5a; B not above its memorisation
    reference) came out as predicted.
- **"S per model" table.** It lists Bswap_s3 and Bswap_s4 beside the registered seeds. A reader averaging five Bswap
  rows gets a non-registered number. Mark them "extra, not in E2c".
- **E5c "below".** It means the model is below a text classifier fitted on its own training set. For B the stack is
  0.956, against B's mean S of 0.916. The rung consequence is that L3 is not reached for either model; do not read
  it as "the model is worse than text" in any further sense.

### 12. NOTE: Data integrity (item 6)

- For all 21 models (and base) × {POOL-LB, ID48, ID, ID48 renamed, ID48 renamed keeping order}:
  - stored uids equal the set spec's uids (plus `-ren` / `-rko`), in order;
  - labels equal the materialised labels;
  - token counts are identical across models;
  - all scores are finite.
- The renamed sets have the same item order, the same premise groups and the same labels as ID48. The query differs
  on all 2,000 items, and the code uses ID48's original string identity for S (`build_sets`). They share ID48's
  resample indices (asserted in code and confirmed).
- Token counts of both renamed sets equal ID48's on all 2,000 items. `ft_run.score` re-encodes each set's own
  (renamed) items, so they are re-counted by construction; Z…U and U…Z tokenise like A…F. Not a problem, but worth one
  line, because "re-counted" could be misread as "different".
- The score is v1.0 `answer_logit_diff`: logsumexp over the Yes variants minus logsumexp over the No variants, which
  is slightly more than "Yes − No". It is the v1.0 convention and the same for every model.
- `ingest.py`'s docstring says it checks "the stored labels", but `check_model` does not compare labels. The analysis
  loader does (`load_scores`), and I confirmed labels independently, so there is no consequence.

### 13. NOTE: Other observations

- The fast closed form (`Plan`) is checked against the registered function only on the point estimates and the first
  3 resamples (`check_first=3`). My fully independent resampling (2,000 resamples) gives the same premise SEs, so the
  closed form is validated on all of them.
- The tolerance `TOL = 1e-4` is looser than the module docstring's "1e-9". The recorded max discrepancy is 4.4e-16.
- The `cached_refs` dependency list omits `ftstudy/stat.py` and `impl.py`, so a change there would not invalidate the
  cache. My independent rebuild of every reference agrees, so this is irrelevant for this run.
- E3's share (AUROC_C − 0.5) / (AUROC_A − 0.5) = 0.889 [0.862, 0.914]. I recomputed the point value from per-seed
  AUROCs: 0.4269 / 0.4803 = 0.889. C's pooled AUROC on ID is 0.925–0.930, so almost all of A's headline ID AUROC is
  "available without the premise", as E3's wording requires.
- Blinding: `.refs_cache.pkl`, `text_references.*` and `behaviour_base.json` all predate the ingestion of A_s4 and
  B_s4. Analysis code is unchanged between abf1a9b and the registered run.

---

## Conditions for using these numbers on the results page

1. A's rung: one sentence with the registered reference (+0.075, Holm 0.018) and the pair-only variant (+0.009,
   p 0.49), naming the member (pair-only, τ = 4) and noting that the amendment's 0.745 is its τ = 2 member. Do not
   describe A as beyond copying (finding 9).
2. Add or disclose:
   - the E1 training-threshold F1 (deviation; finding 5);
   - E3's text references on ID48 (finding 6);
   - A48's E5c interval and its seed-SD note (finding 7);
   - the near/far medians and counts (finding 8).
3. Report E2b, E5_A48 and E4b as in finding 11, and state the failed predictions (S(A) on POOL-LB; E4b).

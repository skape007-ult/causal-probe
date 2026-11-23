# Third review of Amendment 3 (E7 steering, E8 patching): text in PREREG.md, pilot code, timing, pilot diagnostics

Reviewer: fresh, no earlier involvement. Machine date 22 Nov 2026 (commit times below are -0500).

Read: `finetune/PREREG.md` lines 426-588 (Amendment 3 as appended, commit 3a51700); `finetune/notes/audit_amendment3_raw.md`;
`finetune/notes/audit_amendment3_followup_raw.md`; `finetune/ftstudy/e7e8.py`; `finetune/tests/test_e7e8.py`;
`finetune/scripts/e7e8_prepare.py`, `e7e8_gpu.py`, `e7e8_pilot_analysis.py`, `make_e7e8_notebook.py`;
`finetune/notebooks/ft_e7e8_pilot.ipynb` (pinned commit only); `finetune/ftstudy/seedtest.py`; v1.0 `tests/test_steer.py`,
`src/causal_probe/steer.py` (`AddAtLast`), `src/causal_probe/cache.py` (`ResidCollector`); `scripts/notebook_cells.py`
(`FINAL_CHECK`); `finetune/e7e8/pilot_prepare.json`; `finetune/results/e6_descriptive.json` (only `S_selection_oof` at
content_end, to recheck the site layers); `finetune/results/e7e8_pilot_synthetic.json`.

Run: `pytest -q finetune/tests/test_e7e8.py` -> **8 passed** (2 harmless RuntimeWarnings from `nanmean` on an all-NaN
column, which is the case the test checks). I also recomputed the donor skip rate for the registered item sets with
`ftstudy.e7e8.donors` (Section 5 below).

---

## 1. Conditions A-F against the text in PREREG.md

**A. Timing and PREREG consistency: met.**
- "**Superseded sentences [A].** Section 4 ends with "If that does not happen by the deadline, the page says the causal
  tests were not run". The deadline (22 Nov 18:00) has passed, and that sentence is replaced by this amendment: the page
  says E7/E8 are registered and not yet run until they are."
- "Section 5's order (amendment after E6 **and its audit**) is changed: Amendment 3 is registered while E6 is still being
  extracted, before any E6 number has been seen."
- Timing check: see question 4 (met).
- Small blemishes, not blocking: the heading says "03:15 UTC" but the commit is 03:02:14 UTC; the "Earlier status"
  paragraph left in the registered text still says "It goes into `finetune/PREREG.md` after a follow-up review confirms
  the conditions are met", which is now stale history inside the registration; the follow-up review asked that the
  22 Nov page "reported E7/E8 as not run" be stated as fact; the text says the page "says E7/E8 are registered and not
  yet run", which covers it going forward.

**B. Degenerate controls: met in the text, one clause missing.**
- "Cells with a single gold class are left unchanged by the shuffle [B]."
- "Reported per model [B]: the share of selection-half items in cells with both gold classes, the share of gold-label
  variance that lies within cells, and the median |cos(g, control)|."
- "A condition's "not shown" is labelled **uninformative**, not "not shown", if, in any of its models: r < 0.10; or the
  report-half separation of class means along g (in σ) is not above that of the median control; or the median
  |cos(g, control)| > 0.9."
- Missing: B's last sentence, "The pilot reports these numbers and the loading correlations", is not in the text. The
  pilot code does report the B numbers (`pilot_prepare.json`), but not all the loading correlations (mismatch M3 below).

**C. Power rule specification: met.**
- "s × sqrt(2 / q), where q = 0.446 is the lower 20% point of χ²₂, so a factor of 2.117. The floor of 0.03 applies to
  that bound: max(bound, 0.03)."
- "**Premise SE [C]:** from 2,000 premise-cluster resamples of the pilot items, scaled by sqrt(pilot premises / premises
  in N_items). It is computed per condition (A and B each with its own premise variance), with the seed SD pooled."
- "per test (E7(A), E7(B), E8(A), E8(B)). N_items is the largest N needed by any test not dropped [C]."
- "the power script and the E7/E8 code ... are committed and tested on synthetic input."
- Gap left open (M9 below): whether a test made descriptive by the 30% donor rule counts as "not dropped" for N_items.

**D. Half-swap specification: met in the text.**
- "Directions, d_text (5-fold out-of-fold stack inside the report half), the shuffle cells, controls and σ are all
  rebuilt inside the report half by the same rules. ... with donors from the selection half, at the same L_c. **The
  replication sentence states that L_c was chosen on the half the swap tests on**".
- "including the L_c rule, as a unit test on synthetic E6 output that must return the planted layer, with NaN layers
  present" -- the test exists and passes.
- The code part of D is only partly met: no half-swap code and no registered-test code exist (M10).

**E. GPU checks: met in the text.**
- "(a) [E] A `finetune/` copy of the logic of all three tests in v1.0 `tests/test_steer.py` (KV-cache steering = full
  forward at both positions, and the hooked-layer test), run on the merged model, tolerance 0.02 in LD (float16)."
- "(c) [E] ... |(h_stored − h_live)·v| ≤ 0.05 σ_v for g and for every control direction v, and ‖h_stored − h_live‖ /
  ‖h_stored‖ ≤ 0.5%."
- "A failed check halts the run. Any change to a check or a tolerance is a new, dated amendment."
- No tolerance is written for the hooked-layer test itself (it compares residuals, not LD); the code picks one (M6).

**F. One-line fixes: all met.**
- Folds: "`data.grouped_folds(sel_items, 5, seed=0)` [F]".
- Donor pool: "E8 donors come from all registered report-half items (pilot premises excluded), not only the N_items set,
  and an item can be a donor [F]". Either-missing: "If either donor is missing, the item is skipped and counted [F]."
- Truncation: "truncated at the premise boundary at or below N_items [F]".
- s_LD: "s_LD is the SD of the model's zero-vector LD over the items used (E8: over the items not skipped) [F]."
- Upstream: "If the site value is ≥ the final-position value − 0.02 [F]".
- Outliers: "claimed only if both versions have Holm-adjusted p < 0.05, each within the family of four [F]."
- Ratio: "the pilot reports this ratio as a number too [F]."
- NaN: "layers where m(l) is NaN are ignored [F]."
- Still loose (not one of F's items): "the first item j in the seeded order" for donors does not say which seeded order
  (M8).

## 2. Do A-F close the first review's partly met conditions?

| First review | Closed? | Reason |
|---|---|---|
| 1 (commit before `e6_descriptive.json`) | **Yes** | A; timing verified (question 4). |
| 3 (stratified controls; loading reported, "the pilot output contains these correlations") | **Text yes, code no** | Cell definition and single-class rule are in the text. The pilot output (`pilot_prepare.json` `loading_g`) has answer, log tokens and adjacency, but **not** the tuned-stack score, and **nothing for the controls** (median and range), which both the text ("The same is reported for the controls (median and range)") and 3(b)'s check require. |
| 4 (degenerate g rule) | **Yes** | r < 0.10, separation clause restored, cos > 0.9 added; code computes all three quantities. |
| 5 (power rule free of X; χ² bound) | **Yes** | Text has the formula; in code the per-model X values enter only `seed_sd_bound` (`e7e8_pilot_analysis.py` lines 69, 78); the bound and floor are as written (`e7e8.py` lines 171-179); unit test checks 2.117. |
| 9 (undefined pieces a-e) | **Yes in the text; (e) yes in code** | (a) folds named and used (`e7e8_prepare.py` line 43); (b), (c) as before; (d) pool named, and the code uses it; (e) tolerances fixed and coded (`TOL_C_PROJ`, `TOL_C_NORM`). The donor order is still ambiguous (M8). |

## 3. Code against Amendment 3 Sections 1-5

Unchanged since 0873fed (22:09, before any E6 data existed): `git log` shows no later commit touching `e7e8.py`, the
tests, or the three scripts. Notebook pinned to 0d44366 (on `origin/exp/finetune`), which contains the real
`pilot_inputs.npz`.

What matches (checked line by line): site rule (first layer with e ≥ 0.9 max e, None if max e ≤ 0.02; recomputed by hand
from `e6_descriptive.json`: A gives 14, B gives 16, as in `pilot_prepare.json`); Gram-Schmidt order d_ans then d_text;
cells string × own sign × text sign, seeds 1-20, same projection and normalisation; single-class cells unchanged by
construction (tested); r, mixed-cell share, within-cell variance share, median |cos|; report-half separation in σ for g
and the median control; σ from selection-half projections; steps ±1, ±2 σ with 0 the zero vector; OOF d_text with
`grouped_folds(.., 5, seed=0)`; own-answer probe by `probes.fit_predict` on the selection half, S with own sign as label
on the report half, at (content_end, L_c) and (final, last layer); take-premises truncation (tested; real pilot gives 198
items, 50 premises); donor rule (other premise, same string, same adjacency, ±10% tokens, both donors required);
slopes; e_i sign; s_LD over the items used and over the non-skipped items for E8; X7/β(d_ans) reported; premise SE by
2,000 premise-cluster resamples; seed SD pooled over A and B, 2 df, factor sqrt(2/0.446) = 2.1176, floor 0.03 on the
bound; power simulated with the registered `seedtest.t_and_df` one-sample form, item slopes fully correlated (one common
premise error), α = 0.05/4, threshold 0.6; grid {300, 600, 1000, all left}; per-test N, N_items = max over tests not
dropped; tolerances 0.02 LD, 0.05 LD, 0.05 σ_v, 0.5% exactly as `TOL_A, TOL_B, TOL_C_PROJ, TOL_C_NORM`; check (c) on g and
all 20 controls for the first 20 items; layer indexing consistent between `ResidCollector` (E6), `AddAtLast` and check (c),
and the hooked-layer test ties them together.

### Mismatches between text and code

- **M1. Halt on failure (Section 5, "A failed check halts the run").** `e7e8_gpu.py` writes FAILED for a model and
  **continues with the next model** (docstring: "the run continues with the next model"). The notebook does not halt
  either; it marks the kernel's STATUS FAILED at the end (and the laptop ingest refuses such a run unless overridden,
  which happened for E6). Worse, `e7e8_pilot_analysis.py` lines 51-53 **skip a failed model and compute the power rule
  from the rest**: with one A model failed, `seed_sd_bound` falls to df = 1 and silently uses `chi2.ppf(0.20, 1)`
  instead of the registered 2.117 factor, and the premise SE comes from one seed. Nothing in the text allows a power
  rule from three pilot models.
- **M2. Check (b) timing (Section 5, "Before anything else ... (b) A zero vector gives an LD within 0.05 of the model's
  stored POOL-LB score").** The code runs (b) **after** all interventions. Also it compares with E6's
  `answer_logit_diff` (asserted within 0.05 of the stored score in `e7e8_prepare.py` line 99), not with the stored
  POOL-LB score itself; the E6 log says the two are bit-identical, so this second point is a wording mismatch only.
- **M3. Loading (Section 2, "correlation of h·g with the model's answer logit difference, with the selection-half tuned
  stack's score, with log token count, and with adjacency. The same is reported for the controls (median and range)").**
  `e7e8_prepare.py` lines 108-111: no tuned-stack score correlation, no controls. First review 3(b) unmet in the code.
- **M4. Steered-residual norm (Section 2, "The norm of each steered residual relative to the item's own norm is
  reported").** Not computed anywhere (computable on the laptop from E6's h and the stored vectors).
- **M5. Merge precision (Section 5, "merged into the float16 base (`merge_and_unload`)").** Code merges into a
  **float32** base and runs under float16 autocast (`e7e8_gpu.py` line 105 and docstring). Probably the better choice
  (it matches E6's arithmetic, and check (c) tests it), but it is not what is registered.
- **M6. Check (a) is not "the logic of all three tests".** Present: KV-cache = full forward at the final position and at
  content_end, at layers 0, L_c and the last, tolerance 0.02 LD; hooked-layer test at L_c only. Missing from the v1.0
  tests: zero vector = plain forward (`final_logits`); steering changes the answer (> 1e-3) where it should, and does not
  at the last layer from content_end; the hooked-layer test at several layers and its propagation assert (index L+1
  changes). The hooked-layer tolerance (`atol = 0.05‖v‖/√d + 1e-2`, about 0.11 per element for d = 1536) is the code's
  own choice; the text fixes none. Check (a) runs on one item.
- **M7. Site rule, NaN handling (Section 1, "average over the five seeds, giving m(l); layers where m(l) is NaN are
  ignored").** Code uses `np.nanmean` over seeds, so a layer where one seed is NaN keeps a four-seed mean instead of
  being ignored. **Moot for the real data**: no NaN occurs in any of the ten runs' `S_selection_oof` at content_end
  (recomputed; plain mean gives the same layers 14 and 16).
- **M8. Donor order (Section 3, "the first item j in the seeded order").** Section 1's "seeded order" is a premise order;
  the code uses a seed-0 permutation of the individual items of the pool (`e7e8.py` line 132). The text does not say
  which; name it. Also the pilot's own donor pool (code: report half minus pilot premises) is not specified in Section 4.
- **M9. Descriptive-only E8 and N_items.** `e7e8_pilot_analysis.py` lines 90-96 compute N_items over all four tests,
  including an E8 flagged descriptive-only by the 30% rule. The synthetic run shows the effect: E7(A), E7(B) dropped,
  both E8 flagged descriptive, and N_items = 300 set by E8. The text ("the largest N needed by any test not dropped") does
  not say whether a descriptive test counts. With the real data this decides N_items (Section 5 below).
- **M10. Registered-run code absent.** Section 4 says "the power script and the E7/E8 code ... are committed and tested
  on synthetic input" before the pilot. There is no code for the registered E7(c)/E8(c) test (2,000 premise resamples
  with every direction's slopes and s_LD recomputed), Holm over m = 4 with p = 1 entries, the outlying-seed rule, the
  uninformative / upstream / not-run labels, or the half swap (report-half OOF stack, cells, controls, selection-half
  donors). Only the pilot side exists.
- **M11. Pilot "Measured" list (Section 4).** "the item-level SD of b_i(g)/s_LD and of the controls' mean, and of e_i,
  together with their correlation": the code reports the SD of the per-item difference and of e_i, not the two SDs
  separately and not the correlation. Not used by the rule; minor.
- **M12. Notebook docstring.** "A_s0 and B_s0 on one T4": the code puts A_s0, A_s1 on GPU 0 and B_s0, B_s1 on GPU 1.
  Harmless.

Non-mismatches worth recording: the pilot's X values enter nothing but the seed SD (verified); `e7e8.power` reproduces the
registered test's form; steering and patch vectors are stored as float16 in the npz (relative error ~1e-3, harmless).

## 4. Was the amendment committed before the E6 numbers existed?

- 3a51700 (Amendment 3 appended): **2026-11-22 22:02:14 -0500 = 23 Nov 03:02 UTC.**
- First commit containing `finetune/results/e6_descriptive.json`: 9d7f465, **2026-11-22 23:07:22 -0500 = 04:07 UTC**,
  65 minutes later.
- `finetune/e6/data/*` directories: created 22:48:33-22:49:25 -0500, 46 minutes after 3a51700. The E6 kernel ran
  02:37-03:40 UTC (log), so the amendment was committed while it was still running, and its log prints no S value
  (follow-up review).
- The E7/E8 code (0873fed, 22:09) also predates the data and has not changed since.

**Yes.** Condition A's timing check passes. (Commit times are author-set; the push time is not checkable here.)

## 5. What `pilot_prepare.json` already decides

The directions, σ, cells, separation, r and the upstream probe are fitted on the selection half and the report half of
E6's stored activations; none depends on N_items or on any intervention. The registered run uses the same selection
half and the same L_c, so **these pilot values are the registered values for A_s0, A_s1, B_s0, B_s1**. The other six
models' values are fixed by E6 too but not yet computed.

**(i) "Uninformative" for condition A: decided.** Rule: "A condition's "not shown" is labelled **uninformative** ... if,
in any of its models: r < 0.10". A_s1 has `r_projected_norm` = **0.0970** (< 0.10). So any "not shown" for E7(A) or
E8(A) is "uninformative" whatever A_s2-4 show. A claim is unaffected. For B: r = 0.114, 0.118; separation g against
median control 0.132 vs 0.102 and 0.068 vs 0.058 (above, narrowly); median |cos| 0.786, 0.832 (≤ 0.9). Not triggered for
B_s0 and B_s1; B_s2-4 still open. (A_s0: r 0.154, separation 0.064 vs 0.059, cos 0.776; A_s1: separation 0.068 vs 0.039,
cos 0.716.)

**(ii) Upstream qualifier for condition B: decided for both pilot models.** Rule: "If the site value is ≥ the
final-position value − 0.02, every statement about c carries the qualifier". B_s0: 0.981 ≥ 0.9957 − 0.02 = 0.976; B_s1:
0.984 ≥ 0.975. Not triggered for A (0.844 vs 0.961; 0.862 vs 0.973). The text does not say whether the qualifier is
judged per model, on the seed mean, or "in any model"; on the two B values it is triggered under any of these unless
B_s2-4 differ by more than about 0.01, so it is practically decided for B.

**(iii) E8 descriptive-only: decided for both conditions, at every N.** Rule: "More than 30% of items skipped in any
model → E8 for that condition is descriptive only and enters the family with p = 1." The donor rule uses only item
metadata (string, gold, adjacency, premise, token count), so the skip rate is the same for every model. Pilot: 72 of
198 items skipped (**36.4%**; `n_donor_missing` 64 opposite, 25 same). Recomputed for the registered sets with the
committed code and pool: N = 300: 38.3%; 600: 37.8%; 1,000: 35.6%; all 2,162 left: 35.6%. Every candidate N exceeds 30%.
**E8(A) and E8(B) are descriptive-only and enter Holm with p = 1**, whatever the pilot shows. (This needs no E6 number;
it could have been known before the amendment.)

Consequences: the only tests that can still be claimed are E7(A) (and a null there is "uninformative") and E7(B) (with the
upstream qualifier). If the pilot drops both E7 tests for power -- the follow-up review's estimate, and the synthetic run's
outcome -- the family is four p = 1 entries; the code would still return an N_items set by the descriptive E8 (M9).
Also descriptive, not a rule: report-half separation along g is 0.06-0.13 σ, so the projected patch moves h·g by about
0.1 σ, and an E8 effect of 0.10 s_LD would need an E7 slope near 1 s_LD per σ; E8 had little room even before the
donor rule. None of this may change the design now without a new dated amendment that says it was written after these
diagnostics were seen.

## Verdict

**PASS WITH CONDITIONS.** The text meets A-F (B's "the pilot reports" sentence aside) and closes the first review's
conditions 1, 4, 5 and 9; condition 3 is closed in the text but not in the code. The timing is clean. The code implements
the core rules correctly, but not the halt rule, check (b)'s order, the loading and norm reports, and the registered
analysis, and the real diagnostics already settle three labels that should be recorded before any intervention number
exists.

Conditions, all before the pilot is launched:

1. **Halt.** `e7e8_pilot_analysis.py` refuses to compute the power rule unless all four pilot models have status OK (no
   df = 1 fallback). `e7e8_gpu.py` stops at the first failed check (exit non-zero) rather than running the next model.
2. **Check (b) first.** Compute the zero-vector LD for every pilot item, and apply (b), before any steering or patching
   (or record the post-hoc order in a dated amendment).
3. **Loading and norms (M3, M4).** Add to `e7e8_prepare.py`: within-string Spearman of h·g with the selection-half tuned
   stack's report-half score, and the four loading correlations for the 20 controls (median and range); and the relative
   norm of each steered residual. Re-run prepare and re-pin the notebook.
4. **Dated amendment (Amendment 3a), saying it was written after `pilot_prepare.json` was seen and before any
   intervention:** (a) float32 merge with float16 autocast (M5); (b) the donor order is the seed-0 permutation of pool
   items, and the pilot's donor pool is the report half minus the pilot premises (M8); (c) whether a descriptive-only E8
   counts towards N_items (recommend: it does not; then the code follows) (M9); (d) the hooked-layer tolerance and the
   single-item check (a) as coded, or extend check (a) to the missing v1.0 asserts (M6); (e) site rule NaN handling as
   coded, noting no NaN occurs (M7); (f) whether the upstream qualifier is judged per model or on the seed mean.
5. **Record the decided labels now** (log and the amendment above), from E6 alone: E8(A), E8(B) descriptive-only with
   p = 1 (skip 35.6-38.3% at every N); condition A's "not shown" is uninformative (A_s1 r = 0.097); B carries the upstream
   qualifier. Compute the Section 1-2 diagnostics for A_s2-4 and B_s2-4 too and record them, so no label is left to be
   set after intervention numbers exist.
6. **Registered-run code (M10)**, as Section 4 requires before the pilot: the E7(c)/E8(c) test with s_LD and slopes
   recomputed on 2,000 premise resamples, Holm m = 4 with p = 1 entries, outlier rule, labels, and the half swap, with
   unit tests on synthetic input.
7. The owner's explicit go-ahead for the pilot (HALT, quota), as the amendment already says.

Non-blocking: report the two item SDs and their correlation (M11); fix the notebook docstring (M12); the "Earlier status"
paragraph and heading time in the registered text; outcome-table rows for "not run", "dropped for power" and "E8
descriptive only"; the outcome table's "uninformative" wording covers only the r trigger.

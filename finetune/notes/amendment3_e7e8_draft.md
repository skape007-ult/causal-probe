# Amendment 3 (version 3): E7 (steering) and E8 (patching), registered before any intervention on a fine-tuned model

Status: **version 3**: version 2 met the first review's 12 conditions, and version 3 meets the follow-up review's
conditions A–F (`finetune/notes/audit_amendment3_followup_raw.md`; v2 is commit 9878c35), marked [A]–[F].
Earlier status: **version 2, revised to meet the 12 conditions of the first review** (`finetune/notes/audit_amendment3_raw.md`;
version 1 is commit a3ad4b0). It goes into `finetune/PREREG.md` after a follow-up review confirms the conditions are
met. The registered runs need that **and** the owner's go-ahead (owner, 23 Nov: draft, review, and a pilot only if
quota is left; no intervention runs). This version is committed before `e6_analysis.py` has run on real activations
(condition 1). Every quantity that depends on E6 is fixed by a rule given here.

PREREG Section 4 requires: (i) a site upstream of where the answer is decodable; (ii) the model's own-answer
direction and the text prediction's direction as controls, with the test on the gold direction after projecting them
out; (iii) a power figure from a small GPU pilot. Bracketed numbers [c1]–[c12] mark where each review condition is met.

**Superseded sentences [A].** Section 4 ends with "If that does not happen by the deadline, the page says the causal
tests were not run". The deadline (22 Nov 18:00) has passed, and that sentence is replaced by this amendment: the page
says E7/E8 are registered and not yet run until they are. Section 5's order (amendment after E6 **and its audit**) is
changed: Amendment 3 is registered while E6 is still being extracted, before any E6 number has been seen. That is
earlier than Section 5 required, so no E6 result can have shaped it.

## 0. Question, and the ladder [c10]

Is the gold-label information that a linear probe reads at a site upstream of the answer **used** by the fine-tuned
model, beyond what its own answer and a text-only program already carry? E7 asks this by steering, E8 by patching.

**Amendment to Section 0:** the L4 row ("an intervention shows the label direction is used") is withdrawn. E7 and E8
are reported **outside the ladder**, with their own family (Section 3 here). No rung above L3 exists.

## 1. Models, items, site

- **Models:** A_s0–4 and B_s0–4. Not run: base, A48, C, Bswap.
- **Items [c6, c9]:** the report half of POOL-LB (v1.0 `data.selection_mask`, as in E6), minus the pilot's premises.
  N_items per model is set by the power rule (Section 4). Items are drawn as whole premises in a seeded order (seed 0), truncated at the premise boundary at or below N_items [F], and are **the same items for E7, E8 and all ten models**. E8 donors come from all registered report-half items (pilot premises excluded), not only the N_items set, and an item can be a donor [F].
- **Position:** `content_end`, the last token of the hypothesis. The answer is read at the final position.
- **Layer, by rule [c2]:** for condition c, take E6's per-run `S_selection_oof` at content_end (stride 1: every
  layer 0–28) and average over the five seeds, giving m(l); layers where m(l) is NaN are ignored [F]. Excess e(l) = m(l) − 0.5. L_c is the first layer with
  e(l) ≥ 0.9 × max_l e(l). **If max_l e(l) ≤ 0.02, condition c is not run**, and its two tests enter the family with
  p = 1 (Section 3), reported as "no gold signal at the site on the selection half". This uses selection-half numbers
  only. Report-half S never enters the choice.
- **How upstream the site is [c9]:** a probe on the model's own answer sign (v1.0 `probes.fit_predict`, trained on
  the selection half), at (content_end, L_c) and at (final, layer 28). The score is the AUROC within string (the S
  statistic with the own-answer sign as the label) on the report half. If the site value is ≥ the final-position value − 0.02 [F], every statement about c carries the qualifier "at a site where the model's answer is already
  as decodable as at the answer position". The tests are run unchanged.

## 2. Directions and controls

All directions are unit vectors in raw activation space at (content_end, L_c), fitted per model on the selection
half.

- `d_gold`: difference of class means of the gold label (v1.0 `steer.diff_of_means`).
- `d_ans`: difference of class means of the model's own answer sign (answer logit difference > 0).
- `d_text` [c9]: difference of class means of the sign (decision > 0) of the tuned text stack. The stack is computed
  out of fold inside the selection half: 5 premise-grouped folds (`data.grouped_folds(sel_items, 5, seed=0)`) [F], the stack of `finetune/scripts/text_references.py`
  fitted on 4 folds and scored on the fifth, with its n-gram C chosen inside the 4 folds.
- **Tested direction:** g = unit(P d_gold), where P projects out span{d_ans, d_text} (Gram–Schmidt, in that order).
- **Degenerate g [c4, B]:** r = ‖P d_gold‖ / ‖d_gold‖ is reported per model. A condition's "not shown" is labelled
  **uninformative**, not "not shown", if, in any of its models: r < 0.10; or the report-half separation of class
  means along g (in σ) is not above that of the median control; or the median |cos(g, control)| > 0.9.
- **Controls (K = 20 per model) [c3]:** labels are shuffled within cells of hypothesis string × own-answer sign ×
  text-prediction sign on the selection half (seeds 1 to 20). For each shuffle, d = diff of means is computed and then
  projected and normalised exactly as g. A control therefore has the same dependence on the model's answer and on the
  text prediction as the gold labels have, inside each string. Only the gold-versus-other split inside those cells is broken. Cells with a single gold class are left unchanged by
  the shuffle [B]. Reported per model [B]: the share of selection-half items in cells with both gold classes, the
  share of gold-label variance that lies within cells, and the median |cos(g, control)|.
- **Loading of g, reported [c3]:** within string (Spearman, averaged with item weights) on the report half,
  correlation of h·g with the model's answer logit difference, with the selection-half tuned stack's score, with log
  token count, and with adjacency. The same is reported for the controls (median and range).
- **Scale:** σ = SD of the selection-half activations projected on that direction. Steps α ∈ {−2, −1, 0, +1, +2} σ.
  The norm of each steered residual relative to the item's own norm is reported.

## 3. Statistics

**Units [c6].** LD is the answer logit difference (Yes − No) at the final position. s_LD is the SD of the model's zero-vector LD over the items used (E8: over the items not skipped) [F]. Every E7 and E8 statistic is divided by s_LD, so both are in "SD of the
model's own answer score".

### E7, steering
- b_i(v): the least-squares slope of LD_i(α) on α (in σ units) over the five steps, from v1.0
  `steer.steered_logit_diffs` (KV-cache form).
- β(v) = mean_i b_i(v) / s_LD.
- Model statistic X7 = β(g) − mean over the 20 controls of β(control).
- **E7(c):** D = mean over the 5 seeds of X7 against 0, by the registered test (Section 3 of PREREG). The premise
  variance comes from 2,000 premise resamples of the items, with every direction's slopes and s_LD recomputed on each
  resample, models fixed.
- Reported beside it: the rank of β(g) among its 20 controls for each seed, β(d_gold) unprojected, β(d_ans), and
  **X7 / β(d_ans)** (the effect as a fraction of what moving along the model's own answer direction does) [c6]; the pilot reports this ratio as a number too [F].

### E8, patching
- **Donor [c9]:** for item i, the first item j in the seeded order such that: j is from another premise, has the same
  hypothesis string, the opposite gold label and the same adjacency, and its token count is within ±10% of i's. A
  same-label donor j' is chosen by the same rule with the same label. If either donor is missing, the item is skipped and counted [F].
- **Projected patch (the tested version):** h_i ← h_i + ((h_j − h_i)·g) g at (content_end, L_c). Then LD_i is read.
- e_i = { [LD_i(patch j) − LD_i] − [LD_i(patch j') − LD_i] } × (2 y_j − 1) / s_LD. This is the move towards the
  donor's label, minus the move a same-label donor causes (y_j is the opposite-label donor's gold label, 1 = valid).
- Model statistic X8 = mean_i e_i. **E8(c):** the registered test, 5 seeds, against 0. The full-vector patch is
  reported as descriptive.
- More than 30% of items skipped in any model → E8 for that condition is descriptive only and enters the family with
  p = 1.

### Family, outliers, replication
- **Family [c8]:** E7(A), E7(B), E8(A), E8(B). Holm over **m = 4 always**. A test that is dropped, not run, or made
  descriptive enters with p = 1. Claimed means Holm-adjusted p < 0.05.
- **Registered effect [c6]:** 0.10 (in s_LD units per σ for E7, in s_LD units for E8), for both.
- **Outlying seed [c7]:** a seed whose X lies more than 0.10 (the same units) from its condition's median is
  outlying. The test is then reported with and without it, and claimed only if both versions have Holm-adjusted p < 0.05, each within the family of four [F].
- **Replication [c11, D]:** half swap, pre-specified. Directions, d_text (5-fold out-of-fold stack inside the report
  half), the shuffle cells, controls and σ are all rebuilt inside the report half by the same rules. Steering and
  patching are run on items of the selection half (the same N_items rule, seed 0), with donors from the selection
  half, at the same L_c. **The replication sentence states that L_c was chosen on the half the swap tests on**, so the
  swap is not independent of the site choice. A
  claimed E7(c) or E8(c) is reported as "replicated" if the swapped D has the same sign and its 95% interval excludes
  0. Otherwise it is "not replicated", in the same sentence.

## 4. Power from the pilot [c5]

- **Pilot models:** A_s0, A_s1, B_s0, B_s1. 200 report-half items per model (whole premises, seed 0). Their premises
  are then excluded from the registered items.
- **Measured:** the item-level SD of b_i(g)/s_LD and of the controls' mean, and of e_i, together with their
  correlation. Also the per-model X7 and X8, used only to estimate a seed SD.
- **Seed SD for the power simulation [C]:** the upper 80% bound of the SD pooled over A and B (two seeds each, so 2
  degrees of freedom): s × sqrt(2 / q), where q = 0.446 is the lower 20% point of χ²₂, so a factor of 2.117. The
  floor of 0.03 applies to that bound: max(bound, 0.03).
- **Premise SE [C]:** from 2,000 premise-cluster resamples of the pilot items, scaled by sqrt(pilot premises /
  premises in N_items). It is computed per condition (A and B each with its own premise variance), with the seed SD
  pooled. **The pilot's X values do not enter otherwise**:
  no term in |X|, and no site, direction, step, item rule or test changes after the pilot.
- **Correlation across seeds:** item slopes are taken as fully correlated across seeds, so the premise SE of the
  seed mean does not shrink with the number of seeds.
- **Rule:** N_items is the smallest value in {300, 600, 1,000, all report-half items left} at which the simulated
  power of the registered test at α = 0.05/4 is at least 0.6, for the registered effect 0.10, per test (E7(A), E7(B), E8(A), E8(B)). N_items is the largest N needed by any test not dropped [C]. If none reaches 0.6 for a test, that test is dropped
  (standing order 2), enters the family with p = 1, and the page says why.
- **Before the pilot runs [C, D]:** the power script and the E7/E8 code (including the L_c rule, as a unit test on
  synthetic E6 output that must return the planted layer, with NaN layers present) are committed and tested on
  synthetic input. The pilot runs only if quota is left under the week's cap after E6 (owner, 23 Nov).
- **Quota:** one Kaggle session of at most 90 minutes, inside the week's cap.

## 5. Code and checks on the GPU [c9, c12]

- The adapter is merged into the float16 base (`merge_and_unload`) before any intervention, because `steer.py` hooks
  the plain model's layers.
- **Before anything else, on the Kaggle machine, the run halts if any check fails:**
  - (a) [E] A `finetune/` copy of the logic of all three tests in v1.0 `tests/test_steer.py` (KV-cache steering =
    full forward at both positions, and the hooked-layer test), run on the merged model, tolerance 0.02 in LD
    (float16).
  - (b) A zero vector gives an LD within 0.05 of the model's stored POOL-LB score, for every item used.
  - (c) [E] For the first 20 items, the unsteered activation h_live at (content_end, L_c) against E6's stored
    h_stored: |(h_stored − h_live)·v| ≤ 0.05 σ_v for g and for every control direction v, and
    ‖h_stored − h_live‖ / ‖h_stored‖ ≤ 0.5%.
  - A failed check halts the run. Any change to a check or a tolerance is a new, dated amendment.
- Activations for donors and directions come from E6's `resid.npy` (selection half), which check (c) confirms
  against the merged model.

## 6. Outcomes

| E7(c) | E8(c) | Statement |
|---|---|---|
| claimed > 0 | claimed > 0 | in c, moving the site along the gold direction, with the own-answer and text directions removed, moves the answer towards the gold label more than controls with the same answer and text dependence do, by steering and by patching |
| claimed > 0 | not shown | steering moves the answer beyond controls; patching that component from real items is not shown to |
| not shown | any | not shown that the projected gold direction at the site is used; intervals given, with the design's power |
| uninformative (c4) | any | g was nearly removed by the projection; nothing is said about use |
| claimed < 0 | any | reported as found |

"Used" is never written without "along this direction at this site, relative to these controls", nor without the
replication sentence and the upstream qualifier where they apply.

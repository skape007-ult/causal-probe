# DRAFT Amendment 3: E7 (steering) and E8 (patching), registered before any intervention on a fine-tuned model

Status: **draft for independent review**. It goes into `finetune/PREREG.md` only after the review passes, and the
registered runs happen only after that **and** with the owner's go-ahead (owner, 23 Nov: draft, review and pilot only;
no intervention runs). Written 23 Nov 2026, before E6's activations were downloaded. Every quantity below that
depends on E6 is fixed by a rule stated here, so it is decided before anyone has seen E6's numbers.

PREREG Section 4 requires three things: (i) a site upstream of where the answer is decodable; (ii) the model's
own-answer direction and the text prediction's direction as controls, with the test run on the gold direction after
both are projected out; (iii) a power figure from a small GPU pilot. Each is covered below.

## 0. Question

Is the gold-label information that a linear probe reads at a site upstream of the answer **used** by the fine-tuned
model, beyond what the model's own answer and a text-only program already account for? E7 asks this by steering
along the direction. E8 asks it by patching the site from another item. Neither result counts towards a rung of
Section 0. A positive result says that moving along the gold direction moves the answer more than matched controls
do. It does not say how the answer is computed.

## 1. Models, items, site

- **Models:** A_s0–4 and B_s0–4. The base model is not run (it does not do the task: E1, E6). A48, C and Bswap are
  not run.
- **Items:** the report half of POOL-LB (v1.0 `data.selection_mask`, as in E6). Directions and controls are fitted
  on the selection half only. **N_items** per model is set by the pilot's power figure (Section 4), drawn as whole
  premises in a seeded order (seed 0), with at most N_items items.
- **Position:** `content_end`, the last token of the hypothesis. Every template token after the user's text follows
  it, so it is upstream of the answer position in the sequence. The answer is always read at the final position.
- **Layer, by rule from E6 (Amendment 2, 6(b)):** for condition c, L_c is E6's "first layer reaching 90% of its
  maximum excess over 0.5" (`first_layer_reaching_90pct_of_max_excess`) at content_end. It is the same layer for all
  seeds of c. If E6 marks a model's activations unusable (Amendment 2, item 2), that model is dropped and the page
  says so.
- **How upstream the site is, reported beside the result:** a probe on the model's own answer sign at (content_end,
  L_c) against the same probe at (final, L_c), both on the report half, both S. If the own-answer probe at the site
  already matches its value at the final position within 0.02, the page says the site is "upstream in position, not
  in content".

## 2. Directions

All directions are unit vectors in raw activation space at (content_end, L_c), fitted per model on the selection
half.

- `d_gold`: difference of class means of the gold label (v1.0 `steer.diff_of_means`).
- `d_ans`: difference of class means of the model's own answer (sign of its answer logit difference).
- `d_text`: difference of class means of the sign of the selection-half-fitted tuned stack (E6 reference row
  `selection_half_fit|tuned_stack`, refitted out of fold on the selection half so that no item's own label enters
  its own score).
- **Tested direction:** `g = unit(P d_gold)`, where P projects out span{d_ans, d_text} (Gram–Schmidt, in that
  order). The cosines of d_gold with d_ans and with d_text are reported.
- **Controls (K = 20 per model):** labels shuffled within hypothesis string on the selection half, d computed in the
  same way and projected the same way. This keeps each string's label balance, the class sizes and the projection.
  Controls drawn with seeds 1 to 20.
- **Scale:** σ = SD of the selection-half activations projected on that direction. Steps α ∈ {−2, −1, 0, +1, +2} σ.

## 3. Statistics

### E7, steering
- For each item i and direction v: the answer logit difference LD_i(α) with α·v added at the site (v1.0
  `steer.steered_logit_diffs`, KV-cache form, checked against the full forward in v1.0 tests).
- Item slope: b_i(v), the least-squares slope of LD_i on α over the five steps.
- Per model: β(v) = mean over items of b_i(v).
- Model statistic: X = β(g) − mean over the 20 controls of β(control).
- **E7(c), for c ∈ {A, B}:** D = mean over the 5 seeds of X, against 0. The registered test (Section 3) gives seed
  variance, and v_premise comes from 2,000 premise resamples of the items, with every direction's item slopes
  resampled together.
- Reported beside it: the rank of β(g) among the 20 controls for each seed, β(d_gold) unprojected, β(d_ans), and the
  slope in units of LD's own SD.

### E8, patching
- For each item i: a donor j is drawn from another premise with the same hypothesis string, the opposite gold
  label, the same adjacency, and prompt length within ±10% (first such donor in a seeded order; no donor → item
  skipped and counted). A same-label donor j' is drawn by the same rule.
- Patch: replace the residual at (content_end, L_c) of i with the donor's at its own content_end, then read LD_i.
- Item effect: e_i = [LD_i(patch j) − LD_i] × sign(y_j − y_i) − [LD_i(patch j') − LD_i] × sign(y_j − y_i). This is
  the movement towards the donor's label, minus the movement from a same-label donor in the same direction.
- Model statistic: mean e_i. **E8(c)**: the registered test, 5 seeds, against 0.
- **Projected version (the one tested, by PREREG's requirement):** only the component along g is patched,
  h_i + ((h_j − h_i)·g) g. The full-vector patch is reported beside it as descriptive.

### Family and level
- E7(A), E7(B), E8(A), E8(B): four tests, Holm over these four. This is a separate family from the eleven of
  Section 3, because a different question is asked. Claimed means Holm-adjusted p < 0.05.
- Outlying-seed rule as in Section 3.

## 4. Power from the pilot (GPU, before registration is final)

- **Pilot models:** A_s0, A_s1, B_s0, B_s1 (two seeds each, for a first seed SD). Items: 200 report-half items per
  model, as whole premises (seed 0). They are not the registered items: the registered items are drawn after
  excluding the pilot's premises.
- **Pilot measures:** item-level SD of b_i(g) and b_i(control), the correlation between them, the seed-to-seed
  difference of X, and the same for e_i. The effect itself is **not** used to choose anything: the pilot's
  X values are reported as pilot numbers only, and no site, direction, step or test is changed after seeing them.
- **Power:** the registered test simulated with those SDs (premise SE scaled to N_items), seed SD = max(pilot,
  0.5 × |pilot X|, a floor of 0.02 in LD units per σ). It is computed at α = 0.05/4 for effects of 0.05, 0.10 and 0.20
  LD units per σ. **Rule:** N_items is the smallest of {300, 600, 1,000, all report-half items} at which power is at
  least 0.6 for an effect of 0.10. If none reaches 0.6, the test is dropped (standing order 2) and the page says
  why.
- **Pilot quota:** one Kaggle session of at most 90 minutes, inside the week's cap.

## 5. Outcomes

| E7(c) | E8(c) | Statement |
|---|---|---|
| claimed > 0 | claimed > 0 | in c, moving the site along the gold direction, with the own-answer and text directions removed, moves the answer towards the gold label more than matched controls do, by steering and by patching |
| claimed > 0 | not shown | steering moves the answer beyond controls; patching the same component from real items is not shown to |
| not shown | any | not shown that the projected gold direction at the site is used; intervals given, with the power the design had |
| claimed < 0 | any | reported as found |

"Used" is never written without "along this direction at this site, relative to these controls".

## 6. What would make the design fail, and how it is checked

- The projection removes the effect along with the confounds: reported via β(d_gold) unprojected, and the cosines.
- Steps off the activation manifold: α ≤ 2σ of the projected direction; the norm of the steered residual relative
  to the item's own norm is reported.
- Donor matching leaves too few items: the count of skipped items is reported. If more than 30% are skipped in any
  model, E8 for that condition is reported as descriptive only.
- Length: LD is also reported with E6's length residual (within string) removed. The test uses raw LD, because
  every item is compared with itself.

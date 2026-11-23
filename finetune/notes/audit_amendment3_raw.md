# Independent review of the draft Amendment 3 (E7 steering, E8 patching)

Reviewer: fresh, no earlier involvement. Date: 22 Nov 2026 (machine clock). The draft itself is dated 23 Nov.
Read: `finetune/PREREG.md` (whole, incl. Amendments 1-2), `finetune/notes/amendment3_e7e8_draft.md`,
`src/causal_probe/steer.py`, `src/causal_probe/models.py` (get_layers, body_forward), `tests/test_steer.py`,
`finetune/ftstudy/seedtest.py`, `finetune/ftstudy/stat.py`, `finetune/scripts/e6_analysis.py`,
`finetune/scripts/e6_extract.py` (adapter loading), `src/causal_probe/data.py` (selection_mask, grouped_folds),
`CLAUDE.md`, and `finetune/results/behaviour_registered_tables.md` (to know the regime: S of B's own answers on
POOL-LB is 0.905-0.924, of A's 0.760-0.794).

State of E6 at review time: `finetune/e6/data/` holds only `adapter_sha256.json`; `finetune/results/` holds only
`e6_synthetic.*`. So no real E6 number exists yet, and the draft's claim "before E6's numbers are seen" is true and
can be made verifiable (Condition 1).

## Verdict

**PASS WITH CONDITIONS.** The skeleton (one site, fixed directions, matched-by-construction controls, item slopes,
registered seed test, a donor-matched patch with a same-label baseline, a pilot that sets only N_items) is sound
and close to what Section 4 asks. But as written, (a) the site rule reads E6's report half, which then serves as the
test set; (b) the controls do not match the confound the projection is meant to remove, so a positive E7/E8 could be
produced by residual loading of g on the model's own answer; (c) the power rule lets the pilot's effect into the
design; (d) several quantities (d_text's sign, the "own-answer probe ... both S", E8's power effect, the outlying-seed
threshold, Holm under dropping) are undefined in E7/E8 units; (e) the text contradicts PREREG Section 0's L4 row; and
(f) the v1.0 steering code is not shown to work on a PEFT model. Each is fixable by an edit or a check before the
pilot. The conditions below must all be met before the amendment is appended and before any GPU pilot runs; a
follow-up check by a reviewer (as was done for version 2) is recommended because Condition 3 changes the controls.

## Conditions (each blocking, each checkable)

1. **Order of commits.** Append Amendment 3 (with these conditions met) to `PREREG.md` and commit it before
   `e6_analysis.py` is run on real activations, i.e. before `finetune/results/e6_descriptive.json` exists. Check:
   `git log` shows the amendment commit earlier than the first commit or file mtime of `e6_descriptive.json`.

2. **Site layer chosen on the selection half, not the report half.** `first_layer_reaching_90pct_of_max_excess` in
   `e6_analysis.py` (lines 241-243) is computed from `point`, the report-half S (`S_vec(e_rep, big)`). Amendment 2
   item 3 says layers are chosen on the selection half, and E7/E8 then test on the report half: as drafted the site is
   picked on the test items. Fix: define L_c by the same 90%-of-max-excess rule applied to the seed mean of
   `S_selection_oof` (already computed per run and layer at content_end), with `--layer-stride 1`. Also state a
   fallback: if the max selection-half excess at content_end is below 0.02 (or ≤ 0, where the rule returns layer 0,
   the embeddings), E7/E8 are not run for that condition and the page says so. Check: the amendment names the
   `S_selection_oof` key and the fallback; a unit test on synthetic E6 output returns the planted layer.

3. **Controls must carry the same answer and text association as the gold label.** Labels shuffled within hypothesis
   string have no relation to the model's own answer or to the text prediction, so the projection removes almost
   nothing from them, and they point in random sign (their mean slope is ≈ 0 by symmetry). Meanwhile g = unit(P d_gold)
   is orthogonal to d_ans only as a vector in raw space: because the activation covariance is not isotropic, h·g can
   still correlate strongly with the model's answer, and steering along g can move LD through the answer's own readout.
   With B's answers at S ≈ 0.91, d_gold ≈ d_ans and g is a small residual that is exactly where such loading would
   dominate. The test could then pass for a reason the amendment says it excludes. Required: (a) shuffle gold labels
   **within cells of hypothesis string × own-answer sign × text-prediction sign** (selection half), seeds 1-20, same
   diff-of-means, same projection, same σ rule. Then the controls share the gold label's association with the answer
   and the text, and only gold information beyond them separates g from the controls. If a cell has one class only, it
   is left unshuffled (state this). (b) Report, on the report half, the within-string correlation of h·g with the
   model's LD, with the text score, with log token count and with adjacency (standing order 3, loading), for g and for
   the controls' mean. Check: the amendment text names the cell definition, and the pilot output contains these
   correlations.

4. **Rule for a degenerate projected direction.** State before E6: ‖P d_gold‖ / ‖d_gold‖ (before unit
   normalisation) is reported per model; if it is below a fixed value (suggest 0.10), or the report-half separation
   of class means along g is not above that of the median control, g is called "not separable from the controls" and
   a "not shown" for that model is reported as uninformative, not as evidence of non-use. Check: threshold written in
   the amendment.

5. **Remove the effect from the power rule; fix the seed-SD assumption.** "seed SD = max(pilot, 0.5 × |pilot X|,
   0.02)" makes N_items (and the decision to drop the test) depend on the pilot's effect, which the draft says it
   will not use; a large pilot X even forces a drop. Delete the |pilot X| term. With two seeds per condition each SD
   has 1 df; pool the two seed differences of A and B (2 df) and use the upper 80% confidence bound of the pooled seed
   SD (chi-square, 2 df, factor ≈ 2.1), with the floor kept. For v_premise of the 5-seed mean, assume item slopes
   fully correlated across seeds (premise SE of D = per-model premise SE), which is conservative. Check: the formulas
   appear in the amendment and in the power script, and the script reads no X value.

6. **Units for the registered effect and for E8.** "0.05, 0.10, 0.20 LD units per σ" are given without anchor, and
   E8's e_i is in LD units, for which no effect is registered. State: E7's registered effect in LD per σ (as drafted)
   **and** what it is relative to β(d_ans) (report the ratio at the pilot as a number, not a choice); E8's registered
   effect in LD units, fixed now (e.g. 0.10 and 0.25 LD), with the same N_items rule. Also say that E7 and E8 use the
   same N_items items and the same items for every one of the 10 models. Check: both effects are in the text before
   the pilot.

7. **Outlying-seed rule in the right units.** Section 3's 0.10 is in S units. For X (LD per σ) and mean e_i (LD) set
   the distance now (suggest 3 × the pooled pilot seed SD, fixed when N_items is fixed, or a fixed number per test).
   Check: `seedtest.outlying_seeds(..., distance=...)` is called with the stated value.

8. **Holm when a test is dropped.** State that the family stays m = 4 whatever is dropped (power, the 30% donor rule,
   Condition 2's fallback, Condition 4), with dropped tests entered as p = 1. Check: the text says so.

9. **Undefined pieces.** (a) d_text: "sign of the tuned stack" needs a threshold (the stack's output is a probability
   or score, not signed); state it (e.g. within-string median of its out-of-fold score, or p > 0.5) and the folds
   (`data.grouped_folds(sel_items, 3, seed=0)`). (b) The "own-answer probe ... both S" in Section 1: S scores against
   gold labels; state that it is `stat.S` with labels = own-answer sign, which probe (v1.0 `fit_predict`, C by inner
   CV), and that the comparison is with the final position at the **last** layer (where the answer is read), not at
   L_c. (c) Say what the "upstream in position, not in content" finding changes: at minimum, the outcome sentence for
   that condition then reads "at a site where the answer is already decodable". (d) Donor pool for E8: from the
   registered report-half items only (pilot premises excluded), with the same seeded order for every model.
   (e) Where activations for directions, σ and patches come from (E6's stored fp16 residuals or a fresh pass) and a
   check that they agree (e.g. max |diff| on 20 items). Check: each item a-e has a sentence.

10. **Section 0's L4 row.** PREREG Section 0 defines L4 as "an intervention shows the label direction is used | E7,
    E8". The draft says neither result counts towards a rung. The amendment must explicitly amend the L4 row (remove
    it or redefine it in the draft's words). Check: the amendment names Section 0, L4.

11. **Replication on the other half (standing order 3).** The outcome table says "more than matched controls", i.e.
    beyond controls. Pre-specify the half-swap: directions and σ fitted on the report half, steering on N_items
    selection-half items (pilot-excluded), descriptive, reported beside the test; a claim whose half-swap X has the
    opposite sign is reported as "not replicated on the other half". Check: text present.

12. **Steering code on the fine-tuned models.** `steer.steered_logit_diffs` uses `models.get_layers` = `model.model.layers`
    and `model.model(...)` returning `last_hidden_state`. A `PeftModel` (as loaded in `e6_extract.py`) does not have
    that layout (`PeftModel.model` resolves to the `Qwen2ForCausalLM`, whose `.layers` does not exist and whose forward
    returns logits). `tests/test_steer.py` covers only base Qwen-0.5B, fp32, CPU. Required: load each adapter with
    `merge_and_unload()` (or a wrapper), and as the first step of the pilot run the two equality tests of
    `test_steer.py` (final and content_end) on a merged A_s0 in the dtype and transformers version used on Kaggle,
    plus a zero-vector check against the stored POOL-LB score (≤ 0.05, as Amendment 2). The pilot halts if they fail.
    Check: the pilot log contains these three results.

## Answers to the seven questions

**1. Section 4's three requirements.** (ii) Own-answer and text directions are projected out and the test is on
g: met formally, but see Condition 3 (raw-space vector projection does not remove the answer's linear readout from
h·g). (iii) Pilot: present; rule needs Conditions 5-6. (i) Upstream: met in position only. content_end is before the
answer position, but the layer is chosen where the gold label is already ~90% decodable, so the site is by
construction at, not upstream of, where the label is decodable; and in a model whose answer tracks gold at S ≈ 0.91
the answer is likely decodable there too. The draft's diagnostic only reports this; Condition 9(b-c) makes it have a
consequence and compares with the right reference.

**2. Forking paths.** Fixed before E6: position, method of directions, K, seeds, steps, slope, statistics. Open or
tunable: the layer rule reads report-half numbers (Condition 2), and two E6 variants exist (`..._max_S`, cited by
Amendment 2 6(b), and `..._max_excess`, the one the draft uses; the draft fixes one, which is fine, but cite it as a
code key, not as Amendment 2 6(b)); no fallback when the excess is ≈ 0 (rule returns layer 0); d_text's threshold;
the source of activations; N_items via |pilot X|; the threshold for "degenerate g"; Holm under dropping; outlying-seed
distance. Conditions 2, 4, 5, 7, 8, 9 close them.

**3. Statistics.** β(v) and X are well defined. The registered test applies unchanged to any per-model statistic:
D = mean of 5 per-seed X against 0, s²/5 + v_premise, df from the seed term (4). Valid for slopes and patch effects,
with v_premise from `stat.boot_indices` on the premises of the N_items items, the **same** resample for all 5 seeds
and for g and all controls (as `boot_D` does); the draft says "all directions resampled together", which is right;
add "and all seeds". The directions themselves are fixed (fitted on the selection half), so inference is conditional
on them; say so. Holm over four is defensible: E7/E8 ask a different question from E1-E5 and are a separate family,
which standing order 3 allows if every comparison that could be "the one" is in it. The descriptive rows (unprojected
d_gold, full patch, d_ans) are not tested and must stay unworded as claims. E7 and E8 (projected) are nearly the same
intervention (E8's projected effect ≈ slope × class separation along g), so "by steering and by patching" is two
looks at one linear effect, not two independent confirmations (remark R1).

**4. Controls.** See Condition 3. With within-string shuffles the control can hardly fail (mean slope ≈ 0 by
symmetry, whatever the model does) and the test can pass trivially through answer loading of g. A second structural
mismatch: POOL-LB is balanced within premise; gold labels therefore cancel premise identity in d_gold, but
within-string shuffled labels do not, so control directions pick up between-premise variance g does not have. The
cell shuffle of Condition 3 does not fix this; if POOL-LB items form (premise, string) pairs with one valid and one
invalid item, a per-pair random label swap restricted to cells of Condition 3 would preserve both margins (remark R2).
σ: per-direction σ makes steps "natural" but means a low-variance g gets small steps in norm; report the raw norm of
the 1σ step for g and controls and the class-mean separation along g in σ units (remark R3). Steps ±2σ with a
least-squares slope cancel symmetric off-manifold effects, which is good.

**5. Pilot.** It uses 4 of the 10 registered models; items differ, but seeds dominate, so the pilot X previews two of
five seed values. That is acceptable only if nothing but N_items depends on it and N_items does not depend on X
(Condition 5). Two seeds give a 1-df SD; hence the pooled upper bound. The power rule (smallest N_items in
{300, 600, 1000, all} with power ≥ 0.6 at 0.10 and α = 0.05/4) is coherent once the units are anchored (Condition 6);
note that at df = 4 and α = 0.0125 the critical t is about 4.6, so power 0.6 at 0.10 needs SE ≲ 0.02, i.e. seed SD
≲ 0.045 LD/σ: more items cannot rescue a large seed SD, and the drop rule will then fire, which is the right outcome.

**6. E8.** The sign algebra is correct: since y_j ≠ y_i = y_j', e_i = (LD_i(patch j) − LD_i(patch j')) × sign(y_j − y_i),
the baseline cancels, positive = towards the donor's label beyond a same-label donor. The projected patch
h_i + ((h_j − h_i)·g) g correctly sets the g-component of h_i to the donor's. Matching on string, adjacency, length
and premise (other) is good. It isolates "the g-component of a real item" only to the extent g does not load on the
answer (Condition 3), and the full-vector patch carries everything in the donor's content_end state, so it is right
that only the projected patch is tested. Donor pool, order and the use of pilot premises need stating (Condition 9d).

**7. Interpretability.** Positive uninterpretable without Conditions 2 (site picked on test items), 3 (loading), 12
(code path). Negative uninformative without Condition 4 (g may be pure estimation noise once d_ans is removed, very
likely in B) and Condition 6 (an effect that means something). With these, a positive says: along a direction that
carries gold information beyond the answer and text associations, at this site, moving the state moves the answer
more than matched directions do; a negative is "not shown" with the power the design had, and "uninformative" if g
is degenerate.

## Non-blocking remarks

- R1. Report E7 and E8 as two estimates of one linear effect along g (steered vs on-manifold magnitude), not two
  independent lines of evidence; reword the first row of the outcome table accordingly.
- R2. Check whether POOL-LB's balancing gives (premise, string) pairs; if so, prefer margin-preserving swaps for the
  controls (Condition 3 remains the requirement).
- R3. Report the raw L2 norm of 1σ along g and controls, and the class-mean separation along g in σ; consider also an
  equal-norm sensitivity row.
- R4. "LD also reported with E6's length residual removed" is vacuous for E7 (a per-item constant does not change a
  slope) and for E8 (differences within item). Drop it or say what it is meant to show.
- R5. "Slope in units of LD's own SD": say which SD (report-half LD of that model, unsteered).
- R6. Gram–Schmidt order does not affect P (projection onto a span); say so to avoid the impression it matters.
- R7. The "beyond own answer" framing is close to self-contradictory: information orthogonal to the model's answer
  that nonetheless moves the answer is, by definition, part of the answer pathway. State the hypothesis as "the gold
  mean difference, minus its component along the answer and text mean differences, still moves the answer beyond
  matched controls", which is what is tested.
- R8. fp16 on GPU: report the LD noise of the zero vector repeated (should be ~0) and the spread of LD(0) vs the stored
  score; slopes over ±2σ should be robust to it.
- R9. Compute: E7 needs about (1 + 20 + 3 descriptive) directions × 4 non-zero steps per item; with the KV-cache form
  only the template suffix is re-run, so 1,000 items × 10 models fits a few GPU hours; give the estimate from the pilot.
- R10. The draft is dated 23 Nov while the machine date of this review is 22 Nov; make sure the amendment's timestamp
  is the commit's.

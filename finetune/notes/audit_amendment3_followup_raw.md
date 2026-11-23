# Follow-up review of draft Amendment 3, version 2 (E7 steering, E8 patching)

Reviewer: fresh, no earlier involvement with the draft or the first review. Machine date 22 Nov 2026 (the commits
are dated 22 Nov 21:55 -0500, i.e. 23 Nov 02:55 UTC).

Read: `finetune/notes/audit_amendment3_raw.md` (first review, 12 conditions); `finetune/notes/amendment3_e7e8_draft.md`
(v2, commit 9878c35) and v1 (`git show a3ad4b0:...`); `finetune/PREREG.md` Sections 0, 3, 4 (E6, E7/E8), 5,
Amendments 1-2; `CLAUDE.md`; `finetune/scripts/e6_analysis.py`; `src/causal_probe/steer.py`; `src/causal_probe/models.py`
(`get_layers`, `body_forward`); `tests/test_steer.py`; `finetune/ftstudy/seedtest.py`; `finetune/scripts/text_references.py`
(`tuned_stack`); `finetune/scripts/e6_extract.py` (what the Kaggle log prints); `finetune/kaggle/queue.json` (working copy).

State at review time: `finetune/e6/data/` does not exist locally; `finetune/results/` has only `e6_synthetic.*`. The E6
extraction kernel `ft_e6_resid` is **running on Kaggle** (launched 23 Nov 02:37 UTC, i.e. before the v1 and v2 draft
commits). `e6_extract.py` logs only progress and the largest |answer − stored score|, no probe output, so nothing in
the kernel log reveals an E6 S value. No E6 number has been seen. That remains true only until the data are
downloaded and `e6_analysis.py` is run; this is the binding timing constraint (Condition A below).

## Code facts checked

- `S_selection_oof` exists: `e6_analysis.py` line 185, stored at `runs[<run>]["positions"][<pos>]["S_selection_oof"]`,
  one value per layer in `layers` (all 29 with the default `--layer-stride 1`). It is `e_sel.S(s_oof)`, where `s_oof` are
  3-fold premise-grouped out-of-fold probe scores **inside the selection half** (`probes.oof_scores(Xs, y[sel],
  sel_folds, C=C)`, folds from `data.grouped_folds(sel_items, 3, seed=0)`), the scaler fitted on `Xs` and C chosen by
  inner CV on the selection half. No report-half quantity enters it. (C is chosen on the whole selection half and then
  reused in the OOF folds, a mild optimism that is the same at every layer and does not touch the report half.) It is
  a Python list that can contain NaN if no string is eligible; the L_c rule should ignore NaN (trivial; Condition F).
- `steer.steered_logit_diffs` needs `model.model.layers` and `model.model(...)` returning `last_hidden_state`; a
  `merge_and_unload()` result is a plain `Qwen2ForCausalLM`, which satisfies both. Draft Section 5 is right on this.
- `tests/test_steer.py` is hard-wired to Qwen-0.5B, fp32, CPU, d = 896, `atol=2e-3`. It cannot be run "on the merged
  model" as is; a copy of its logic under `finetune/` is needed (v1.0 files may not be edited, standing order Scope),
  and 2e-3 is not a realistic fp16 GPU tolerance.
- `text_references.tuned_stack` returns a stacker decision whose sign is the stack's own decision rule
  (`fit_stacker(...)(Z) > 0`), and chooses the n-gram C inside its training side. "decision > 0" is well defined.
- `seedtest.holm` with p = 1 entries: they sort last, and adjusted p of the others is unchanged except through m; with
  m fixed at 4 the function does exactly what the draft says. `seedtest.outlying_seeds` default distance is 0.10.

## The 12 conditions

| # | Verdict | Where v2 meets it / what is missing |
|---|---|---|
| 1 | **Partly met** | "This version is committed before `e6_analysis.py` has run on real activations (condition 1)." True (commit 9878c35; no `e6_descriptive.json`, no `e6/data/`). But the condition is that the amendment is **appended to PREREG.md and committed** before `e6_descriptive.json` exists. Not yet done, and E6 extraction is already running. See Condition A. |
| 2 | **Met** | "take E6's per-run `S_selection_oof` at content_end (stride 1: every layer 0–28) and average over the five seeds ... L_c is the first layer with e(l) ≥ 0.9 × max_l e(l). If max_l e(l) ≤ 0.02, condition c is not run ... Report-half S never enters the choice." The key exists, is per layer and selection-half only (above). The unit test on synthetic E6 output was a check of the first review; it belongs to the E7/E8 code, which does not exist yet (Condition D). |
| 3 | **Partly met** | (a) "labels are shuffled within cells of hypothesis string × own-answer sign × text-prediction sign on the selection half (seeds 1 to 20) ... projected and normalised exactly as g." Met. The requested sentence on single-class cells is missing (harmless: a shuffle there is the identity), but the real problem is near-pure cells (new problem 2). (b) "Loading of g, reported ... correlation of h·g with the model's answer logit difference, with the selection-half tuned stack's score, with log token count, and with adjacency. The same is reported for the controls (median and range)." Met for the registered run; the first review's check ("the pilot output contains these correlations") is not stated. |
| 4 | **Partly met** | "r = ‖P d_gold‖ / ‖d_gold‖ ... If r < 0.10 ... labelled uninformative." The threshold is there. The second clause of the condition ("or the report-half separation of class means along g is not above that of the median control") was dropped without comment, and with the new cell-stratified controls it is exactly the clause that is needed (new problem 2; Condition B). |
| 5 | **Met** (text) | "the upper 80% bound of the SD pooled over A and B (two seeds each, so 2 degrees of freedom: s × sqrt(2 / χ²₀.₂₀,₂)). The floor is 0.03. The pilot's X values do not enter otherwise: no term in \|X\|"; "item slopes are taken as fully correlated across seeds". The power script does not exist yet, so "script reads no X value" is unchecked; notation and two specification gaps in new problem 4 (Condition C). |
| 6 | **Met** | "Every E7 and E8 statistic is divided by s_LD"; "Registered effect: 0.10 (in s_LD units per σ for E7, in s_LD units for E8)"; "X7 / β(d_ans)" reported; "the same items for E7, E8 and all ten models". The ratio at the pilot as a number is not stated as a pilot output; minor (Condition F). |
| 7 | **Met** | "a seed whose X lies more than 0.10 (the same units) from its condition's median is outlying." Fixed, in the statistic's units (`outlying_seeds` default). |
| 8 | **Met** | "Holm over m = 4 always. A test that is dropped, not run, or made descriptive enters with p = 1." |
| 9 | **Partly met** | (a) Met: "sign (decision > 0) of the tuned text stack ... 5 premise-grouped folds" — but the fold function and seed are not named (`data.grouped_folds(sel_items, 5, seed=0)`?). (b) Met: "v1.0 `probes.fit_predict` ... at (content_end, L_c) and at (final, layer 28) ... the S statistic with the own-answer sign as the label". (c) Met: "every statement about c carries the qualifier ...". (d) Partly: "E8 donors come from the same pool" — whether the pool is the N_items registered items or all report-half items left after the pilot exclusion is ambiguous, and the skip rate depends on it. (e) Met in substance ("Activations for donors and directions come from E6's `resid.npy` ... check (c)"), but the tolerance is too loose (new problem 6; Condition E). |
| 10 | **Met** | "Amendment to Section 0: the L4 row ... is withdrawn. E7 and E8 are reported outside the ladder." But two other PREREG sentences now conflict (Condition A). |
| 11 | **Met** | "Replication: half swap, pre-specified. Directions, controls and σ are refitted on the report half ... 'not replicated', in the same sentence." Leak and specification points in new problem 3 (Condition D). |
| 12 | **Met** (text) | "merged into the float16 base (`merge_and_unload`) ... (a) `tests/test_steer.py`'s two checks ... (b) zero vector within 0.05 of the stored POOL-LB score ... (c) ... agrees with E6's stored `resid.npy`". Tolerance for (a) in fp16 not stated (Condition E). |

## New problems introduced by v2

**1. Standardisation by s_LD.** Sound in itself. Dividing each model's slopes by its own unsteered LD SD makes A and B
comparable in "fraction of the model's own answer spread", which also makes pooling the seed SD over A and B (power
rule) more defensible. s_LD is recomputed on every premise resample ("with every direction's slopes and s_LD
recomputed"), which is correct for a ratio statistic. Three consequences to keep in mind, none blocking: (i) in a
confident, bimodal model (B, LD near ±several logits) 0.10 s_LD per σ is a large raw effect, so the registered effect
is harder to reach in B than in A in logit terms; this is a fixed choice and is acceptable, but the page should give
the raw-logit equivalent of 0.10 per model beside it. (ii) "unsteered LD" should be named: the zero-vector LD from the
steering code (which check (b) ties to the stored score) — say so. (iii) For E8, s_LD should be over the items not
skipped, and in the half-swap over the selection-half items used; one sentence.

**2. Stratified shuffle cells: degenerate, not empty.** Empty cells do not matter. The risk is the opposite: cells
that are nearly pure in the gold label. In B, whose answers track gold at within-string S ≈ 0.91, the cell
(string, answer = Yes, text = Yes) is mostly valid and (string, No, No) mostly invalid; the gold label then varies
inside cells only for the model's and the text stack's errors. A shuffle inside cells preserves every cell's class
counts, so each control label vector agrees with gold on most items, the controls' diff-of-means carries the whole
between-cell part of d_gold, and after projection each control is close to g. In the limit (all cells pure) every
control equals g and X7 ≡ 0, X8 ≡ 0 by construction. That is the intended logic (only within-cell gold information
separates g from its controls), but it means a "not shown" in B may be a property of the design, not of the model,
and the r-rule of [c4] does not detect this: r measures what the projection removed, not how much of g survives the
cell shuffle. When the answer is nearly constant within a string the cells reduce to string × text sign, and a
one-item cell simply keeps its gold label; neither breaks anything. What is missing is a pre-fixed degeneracy rule on
the controls: report the share of selection-half items in cells holding both gold classes, the within-cell share of
the gold label's variance, and the median |cos(g, control)|; and restore [c4]'s second clause (report-half class-mean
separation along g, in σ, not above the median control's ⇒ uninformative). A "claimed" result is unaffected (the
controls are matched, so a claim beyond them is still meaningful); this concerns only how a null is labelled.

**3. Replication half-swap: one real leak, two gaps.** (i) L_c is reused "at the same L_c", and L_c was chosen on
the selection half, which is the swap's test half: the swap's site was chosen on its own test items, which is the
very defect [c2] removed from the main test. The effect on a steering slope is indirect (the layer is chosen for gold
decodability, not for steerability), so the swap can stay, but the replication sentence must say the site was chosen
on the half it is tested on. The alternative (choose a swap layer from report-half S) would read report-half numbers
that are then the training half of the swap and would need its own rule; not worth it. (ii) d_text for the swap needs
its own out-of-fold stack inside the report half (same 5-fold rule), the swap's cells need the report-half answer and
text signs, and the swap's donor pool is the selection half; none is stated. (iii) No leak from the main test into the
swap or back: the main test's directions are fitted on the selection half and its outcome is on the report half; the
swap is fully pre-specified, so seeing the main result changes nothing. The pilot premises are in the swap's training
half only, which is harmless.

**4. Power rule and the χ² bound.** The arithmetic is right if χ²₀.₂₀,₂ is the **lower** 20% quantile (0.446): factor
sqrt(2/0.446) = 2.117, the first review's 2.1. Under the common upper-tail notation χ²₀.₂₀,₂ = 3.219, the factor is
0.79 and the "bound" would be **below** the estimate. Write the number. Also: (i) "The floor is 0.03" — floor on the
bound or on s? (on the bound, presumably). (ii) "the item-level SD of b_i(g)/s_LD" — the registered test's premise
term comes from premise resamples, and items within a premise are correlated; the power simulation must take its
premise SE from premise-cluster resamples of the pilot items (scaled to N_items by the number of premises), not from
an item-level SD, or it understates v_premise. (iii) Per-test or per-E: "for E7 and for E8 separately" while the seed
SD is pooled over A and B; state whether A and B are simulated with their own item-level (premise) variance, so that,
e.g., E7(B) can be dropped while E7(A) is kept, and that N_items is the largest N needed **by the tests not dropped**.
(iv) Realism: at α = 0.0125, df 4 (t_crit = 4.31; the first review's 4.6 is slightly off), effect 0.10 and zero
premise SE, power is 0.81 at seed SD 0.04, 0.61 at 0.05, 0.44 at 0.06. With the 2.117 factor, a test survives only if
the pooled pilot seed SD is below about 0.023 s_LD/σ, and the pilot's per-model X on 200 items includes item×model
noise that inflates it. The drop rule will likely fire; that is a legitimate pre-registered outcome, not a flaw, and
the page must then say the test was dropped for power. (v) The power script and the E7/E8 code do not exist; they must
be committed and tested on synthetic pilot output before the pilot runs, so that "reads no X beyond the seed SD" is
checkable.

**5. m = 4 Holm with p = 1 entries.** Correct and conservative; `seedtest.holm` handles it. Two interplays to state:
(i) with an outlying seed, "claimed only if both are" should read "only if both the with- and without-seed p are below
0.05 after Holm (each computed with the other three family members as they are)"; (ii) the "uninformative" label of
[c4] and the "not run" fallback of [c2] do not change p, they only label; the half-swap is outside the family and
carries no p in any claim. Both one-line edits.

**6. Activation agreement tolerance (check (c)).** "within 2% relative norm" is too loose for what the activations are
used for. The patch uses (h_j − h_i)·g and σ is the SD along g; Qwen residual norms are dominated by a few large
dimensions, so a 2% norm discrepancy can be several σ along a low-variance direction like g. fp16 storage error alone
is ~0.1%; the real question is whether the merged fp16 model reproduces the unmerged-adapter states along the
directions used. Required: agreement of the projections, |(h_stored − h_live)·v| ≤ 0.05 σ_v for v = g and every
control, on all 20 items (plus the norm check, at ≤ 0.5%). Likewise check (a) needs an fp16 tolerance fixed now (e.g.
0.02 LD), and it should include `test_hook_changes_exactly_the_intended_layer`'s logic, since the two equality tests
share the hook and would both pass with a wrong layer. A failed check halts the run; any change after that is a new
dated amendment (say so).

**7. Things still choosable after E6 or the pilot.** Mostly closed. Left open: the d_text fold function and seed; the
donor pool (N_items items or all remaining report-half items) and whether an item is skipped when **either** donor is
missing; whether the seeded premise order is truncated at a premise boundary; the swap's d_text, cells and donors
(3 ii); the per-test power simulation (4 ii-iii); the tolerances of checks (a) and (c) (6); the degeneracy rule for
controls (2). The upstream qualifier "within 0.02 of the final-position value" should read "at least the final value
minus 0.02" (as written, a site more decodable than the final position escapes the qualifier). Each is a sentence.

**8. Two PREREG sentences the amendment now contradicts (found in this review).** (i) Section 4: E7/E8 "will be
registered in an amendment ... If that does not happen by the deadline, the page says the causal tests were not run."
The deadline (Section 5: Sunday 22 Nov 18:00) has passed; the amendment must say that the 22 Nov page reported E7/E8
as not run, and that this registration is later, on the owner's instruction of 23 Nov. (ii) Section 5's order puts
"6. Amendment for E7 and E8" after "5. E6 and its audit"; [c1] deliberately reverses that, and the amendment should
say so. Without both, a reader of PREREG.md finds the amendment out of its own registered order.

## Non-blocking remarks

- R1 of the first review stands: the first outcome row still says "by steering and by patching"; E8's projected patch
  is approximately the E7 slope times the class separation along g, so these are two estimates of one linear effect.
- The owner's note as quoted in v2 ("a pilot only if quota is left; no intervention runs") reads as allowing the pilot
  but no registered interventions; confirm with the owner that the pilot, which is an intervention on fine-tuned
  models, is covered. Quota: 6.62 h recorded, minus the running E6 kernel (est. 75 min, cap 150).
- Loading (standing order 3): the correlations of h·g are reported but have no consequence; consider a pre-fixed
  qualifier if |ρ(h·g, LD)| within string for g exceeds every control's.
- Merge in fp32 and cast to fp16 to reduce merge rounding against the stored scores, if memory allows.
- Outcome table: add rows for "not run (no gold signal at the site)" and "dropped for power".

## Verdict

**PASS WITH CONDITIONS.** v2 meets conditions 2, 6, 7, 8, 10, 11, 12 and the substance of 5; conditions 1, 3, 4 and 9
are partly met. The design is now sound in its core (selection-half site, cell-matched controls, X-free power rule,
fixed family). The conditions below are edits to the text and checks before the pilot; none needs a further review
round if met as written, though a quick check of Condition B's numbers by the next reviewer would be prudent.

A. **Timing and PREREG consistency.** Append Amendment 3 (with B-F) to `PREREG.md` and commit it **before**
   `finetune/e6/data/` is downloaded or `e6_analysis.py` is run on real activations (the E6 kernel is running now).
   The amendment states that it supersedes Section 4's "by the deadline" sentence (the 22 Nov page reported E7/E8 as
   not run; this is a later registration on the owner's instruction) and Section 5's order (amendment before E6's
   analysis and audit). Check: commit time of PREREG.md earlier than the mtime of `finetune/e6/data/*` and of
   `e6_descriptive.json`.
B. **Degenerate controls.** State that single-class cells are unchanged by the shuffle. Report per model: share of
   selection-half items in cells with both gold classes, the within-cell share of gold variance, and median
   |cos(g, control)|. Restore [c4]'s second clause: if the report-half class-mean separation along g (in σ) is not
   above the median control's, or median |cos(g, control)| > 0.9, a "not shown" for that condition is "uninformative".
   The pilot reports these numbers and the loading correlations.
C. **Power rule specification.** Write the χ² quantile as the lower 20% point (0.446, factor 2.117); the 0.03 floor
   applies to the bound; premise SE from premise-cluster resamples of the pilot items scaled by premise count; power
   computed per test (A and B with their own premise variance, seed SD pooled); N_items is the largest N needed by the
   tests not dropped. Power script committed and tested on synthetic pilot output before the pilot.
D. **Half-swap specification.** d_text, cells and controls of the swap are built inside the report half by the same
   rules (5-fold OOF stack); swap donors from the selection half; the replication sentence states that L_c was chosen
   on the half the swap tests on. E7/E8 code (including the L_c rule, with NaN ignored, as a unit test on synthetic E6
   output returning the planted layer) committed and tested before the pilot.
E. **GPU checks.** Check (c): |(h_stored − h_live)·v| ≤ 0.05 σ_v for g and every control, and relative norm ≤ 0.5%, on
   20 items. Check (a): a `finetune/` copy of the logic of all three tests of `tests/test_steer.py` (including the
   hooked-layer test) with an fp16 tolerance fixed in the text (e.g. 0.02 LD). A failed check halts; any change is a
   new dated amendment.
F. **One-line fixes.** d_text folds = `data.grouped_folds(sel_items, 5, seed=0)` (or as chosen, named); donor pool
   named (all registered report-half items, pilot premises excluded, or the N_items set), item skipped if either donor
   is missing; seeded order truncated at a premise boundary; s_LD = SD of the zero-vector LD over the items used (E8:
   items not skipped); upstream qualifier when site ≥ final − 0.02; outlier rule: both with- and without-seed p Holm
   below 0.05; the pilot reports X7/β(d_ans) as a number; the L_c rule ignores NaN layers.

After A-F are met, the amendment may be appended and the GPU power pilot may run under it (subject to the owner's
explicit go-ahead for the pilot, and the quota).

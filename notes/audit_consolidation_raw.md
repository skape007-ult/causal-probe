# Sixth audit: the corrected round-2 analysis, RESULTS.md and the consolidation (raw report)

Reviewer: independent. Written 13 Nov, 20:45 to 21:40. No language model was loaded (tokenizer only, for the patching pair
list). Everything below was recomputed from `cache/*/c2c_debiased/resid.npy`, `meta.json`, `probe_oof_C*.npz`,
`cache/qwen0.5b/c2c_debiased/steering_adjacency_*.npz`, `steering_controls*.npz`, `cache/qwen0.5b/patching*.npz` and
`data/raw/corr2cause/{train,test}.csv` (streamed) with my own scripts in
`/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit6/`
(`a0_pool.py`, `a1_adj.py`, `b_controls.py`, `b2_premise.py`, `c_patch.py`, `c2_example.py`, `d_ladder.py`, `e_claim1.py`,
`f_cells.py`, `g_start.py`; runner and ingest dry runs under `g/`, `ing/`, `ing2/`, `ing3/`).

**Versions.** I started at commit `9efddfb`. The author committed four times while I worked (`1de96dc`, `e2a832b`, then
`bb3dd2a` at about 20:50: the fourth adjacency site rerun with all arms; then `ddfa165`). `RESULTS.md` did not change. The
fourth site (`hyp_first50`) was recomputed from both its old raw file (`_before_audit5/`) and the new one.

**One thing I did wrong.** `a1_adj.py` first wrote two 4 kB scratch arrays (`split_final_best.npz`, `split_hyp_best.npz`)
into the repository root through a relative path (20:45). They are matched by `.gitignore`, were never committed, and I
moved them to the scratch directory two minutes later. Nothing else in the repository was written except this file.

---

## 1. What I verified and found correct

- **Column layout of the two new blocks** (`item_vectors` against the slices in `analyse`), at all four sites, from the
  saved norms and outputs: translation norm = |delta_g| (rel. error 5e-8); control translation norms = `shuf_delta_norm`;
  erasure norm = |c_i| |delta_g| (my own c_i from the cache); transfer column of the item's own template = translation
  column (max difference 0.0 logits); split: |along| = |delta_g . u| with my own ridge direction (1e-6) and
  along² + remainder² = |delta_g|² (2e-7); fair-control norms = |c_i| |delta_g| for all 20. `delta` and `W` reproduce from
  the cache (7e-8, 4e-5). The 20 within-string shuffles reproduce from the same seed (norms to 2e-7) and preserve the
  adjacent count of every hypothesis string.
- **Every headline number of Section 4.4, four sites**, from the raw arrays: held-out R² 0.960 / 0.791 / 0.425 / 0.314;
  held-out adjacency AUROC before → after erasure 0.855 → 0.697, 0.918 → 0.700, 0.775 → 0.688, 0.795 → 0.681 (my folds;
  report 0.858 → 0.698 etc.) and 0.852 → 0.650, 0.912 → 0.641, 0.745 → 0.642, 0.757 → 0.618 on the evaluated items;
  translation effects, control ranges and shares; pre-specified contrast +0.1102 / +0.0129 / +0.0004 / +0.0045; uniform-sign
  contrast +0.0959 / +0.0155 / -0.0067 / +0.0025; gaps before and after within string; mean absolute gap 0.1022 → 0.0697
  (my interval [-0.009, +0.072]; report [-0.010, +0.077]) and 0.1024 → 0.0972 ([+0.000, +0.015]); the directly measured
  fair control (shares 75 / 55 / 5 / 95 and every per-template share in the third table); the split (final L15: along
  +0.022, -0.012, +0.047, -0.157, -0.131; remainder +0.017, -0.032, -0.083, -0.097, -0.051; hyp L13: +0.005, -0.005,
  +0.003, -0.012, -0.010 and -0.013, +0.003, -0.005, -0.025, -0.015; norms 0.061 s and 0.285 s for common effect).
  Each of `audit5_checks`' readouts is what its name says, with the exceptions in F4, m9 and m11.
- **Steering controls (4.1).** The shuffle is within hypothesis string: replaying the script's random stream gives the
  natural SDs of the 20 control differences of means stored in the raw file to 1e-7, at both tokens. Slopes and shares:
  end of hypothesis +0.095 (45%), +0.287 (80%; largest control 0.420); final token -0.229 (75%), +0.150 (70%); model-answer
  direction 100% at both.
- **Patching (4.2).** Pair lists rebuilt (streamed CSV, tokenizer filter) match the saved premise ids for all four sets;
  the swap direction was re-derived by diffing the two prompts. Premise relabellings, label-changing: +0.252 [+0.176,
  +0.336], 27 of 27 positive (later); -0.165 [-0.242, -0.088], 8 of 33 positive (earlier). Label-preserving: +0.223 [+0.172,
  +0.275], 53 of 60. Matched on direction +0.029 [-0.064, +0.124] (prem), +0.065 [-0.075, +0.202] (hyp). Within template
  and direction: +0.252 against +0.246 (prem).
- **RESULTS.md numbers.** Hypothesis-only classifier on the official test split, recomputed from the two CSV files: AUROC
  0.895 [0.862, 0.920] (21 premises; 0 unseen strings; no test premise in train). Probe minus my own parsed-feature
  classifier (own folds) on the Corr2Cause pool: -0.106, -0.043, -0.049, -0.051 for the four models, all intervals below
  zero (files: -0.113, -0.049, -0.056, -0.057); the four CLadder cells were checked against the result files only.
  Debiased pool: cached probe scores give 0.752 / 0.763 / 0.765 (ensembles; single seeds 0.736 to 0.763), my refit of
  Qwen2.5-3B 0.744 and 0.755, my own lookup 0.762 to 0.767, answers' AUROC 0.505 / 0.510 / 0.492 / 0.507.
- **The move to the appendix.** Appendix A, B, C are line-for-line identical to the old Sections 3.2, 4.3, 5.1; every line
  of old 4.2 is in new 4.2 or Appendix D. No reference to an old section number is left dangling in the report. Section 8
  holds only its one-line note.
- **Runner (`gpu_round2.py`), dry run on scratch directories.** `--tiny` runs under `<run>_tiny` (markers, logs, status,
  environment copy, zip); a pre-planted stale `export` / `all_ok` / `finished` is gone from the status file before the
  first stage starts; `finished` is written at the end. The ingest refuses `--run <run>_tiny`.

---

## 2. Findings

### Major

**F1. RESULTS.md claim 3: "inside their range" is false, and the result file says so.**
Where: `RESULTS.md` / `README.md` claim 3 ("larger in absolute value than 95% of 20 random directions of the same size:
inside their range"); `results/adjacency_qwen0.5b_c2c_debiased_hyp_best.json`, `translation/contrast/vs_controls_norm_matched`.
Evidence: the contrast is +0.0129; the 20 controls run from -0.0210 to +0.0108. The value is above all 20 controls
(`share_of_controls_below` = 1.0, `outside_control_range` = true in the author's own file) and one control is larger in
absolute value. The contrast is signed by prediction, so the directional reading (0 of 20 above) is the pre-specified one;
the uniform-sign contrast (+0.0155 against -0.0177 to +0.0106) is also above all 20. The same holds for the reduction of
the mean absolute gap: +0.005 is above 19 of 20 fair controls at L13 and +0.032 above all 20 at L15 (`share_of_controls_below`
0.95 and 1.0). What is inside the control range is each hypothesis type's own effect (fig12's dots). So the end-of-hypothesis
result is the weakest one 20 controls can register in either direction, not a clean null, and it is one measurement
reported four ways.
Should say: see Section 4.

**F2. RESULTS.md claim 2: "hold about what a lookup ... gives" asserts equal content; only equal accuracy is shown, and the
content differs.**
Evidence (cached out-of-fold probe scores against my per-template lookup on the six pair features, report half): the two
classifiers agree on 68.4% (Qwen2.5-1.5B), 69.2% (Gemma-2-2B) and 70.2% (Qwen2.5-3B) of items. Two classifiers of 76.5%
accuracy agree on 64% if their errors are independent and on 100% if they hold the same thing. The probe is right on 64
to 65% of the items the lookup gets wrong; inside template × lookup-prediction cells the probe's AUROC is 0.77 to 0.78; a
two-parameter stack of the two scores 0.809 / 0.814 / 0.810 against 0.767 for the lookup alone. The report (3.6.2, 3.6.4)
and EXPLAINER step 6 ("equal accuracy; that the content is the same is not shown") already say this; the headline does not.
Also not in the headline: the quoted probe accuracies are three-seed ensembles (single probes 1 to 2 points lower).
Should say: "a linear probe on the larger models' activations is as accurate as a lookup ..., on partly different items".

**F3. RESULTS.md claim 2: "the models answer at chance" repeats, pooled, what retraction row 4 says cancels.**
Evidence, `results/probe_content_*`, `answers/by_template/*/within_string_auroc_for_gold`: direct causation 0.628 [0.581,
0.673] (Qwen2.5-1.5B), 0.387 [0.347, 0.432] (Gemma-2-2B), 0.561 [0.517, 0.609] (Qwen2.5-3B); common cause 0.415 [0.348,
0.489] (Qwen2.5-3B); common effect 0.586 [0.536, 0.636] (Qwen2.5-0.5B). Report 3.6.5: "it is not accurate to say the
model's answers ignore what the probe reads". The same page retracts "The answers do not express what the probe reads"
as "false per hypothesis type; opposite signs cancel".
Should say: "pooled over hypothesis types the models answer at chance; within types they deviate in both directions".

**F4. Report 4.4 point 8: the interpretation of the split is not licensed.** ("So the part of the class difference that
reproduces the observed, type-specific pattern is the part that already predicts the answer; the rest acts as a uniform
push towards No. That is what 'the class difference is an answer shift' predicts, and it is not what a separately encoded
adjacency variable that is used later would predict.") The numbers are right; three things are wrong with the reading.
(a) *The first half is true by construction.* delta_g . coef is the fitted adjacent-minus-not answer gap of type g on the
half both were fitted on (final L15: +0.027, -0.015, +0.055, -0.190, -0.162 against in-sample gaps +0.049, -0.012, +0.057,
-0.206, -0.159). The along-component is (delta_g . u) u, so its effect has the sign of that gap whenever moving along u
raises the answer, which is the positive control. Null, from files on disk: for the 20 shuffled-label class differences,
the along-effect (measured slope along u × dl_k . u) correlates with that shuffle's in-sample answer gap at r = 0.975 over
100 type × shuffle cells at L15 (signs agree in 95%) and r = 0.88 at L13 (86%). Any label's class difference, split this
way, has an along-part that "reproduces" that label's answer gap. The real one is larger than the null (common effect
-0.157 against -0.040 to +0.059) only because the real gap is larger, which is the observational fact the experiment
started from. Both hypotheses predict it.
(b) *The second half is not distinguishable from a random direction.* The remainder's effect is inside the range of the
20 label-free directions rescaled to its norm for every type at both sites (shares smaller in absolute value: 5, 50, 80,
65, 35% at L15; 55, 10, 30, 70, 60% at L13). "Towards No for four types" at L15 is also "the observed sign for four of
five types" (+0.017 / +0.050, -0.032 / -0.042, -0.097 / -0.230, -0.051 / -0.113; only descendant differs); the text gives
one description. Whether a remainder "follows the item's type" was not tested: the split was applied to an item's own
type only, with no transfer arm.
(c) *The remainder is not small, and "answer shift along the ridge direction" does not predict that.* It carries 38% of
the common-effect translation effect at L15 and 67% at L13. The ridge direction is one predictive direction, not the causal
gradient: the measured slope along it is 1.80 to 1.94 logits per unit norm at L15 against a regression slope of 2.26, and
0.11 to 0.36 against 2.65 at L13.
`notes/log.md` (20:30) already says "Not controlled ... no shuffled-label version of the split was run. The split is
descriptive." The report dropped that sentence.
Should say: "The split is descriptive. Its along-component reproduces the observed pattern by construction (a shuffled-label
class difference does the same for its own in-sample gap, r = 0.98); the remainder, 38% of the common-effect effect at
layer 15 and 67% at the end of the hypothesis, is inside the range of random directions of its size. Neither part
discriminates between an answer shift and a separately encoded variable; that needs the split with shuffled labels and
with the transfer arm."

**F5. `kaggle_ingest.py`: a run from an earlier pin silently overwrites newer local result files with stale copies.**
Where: step 4 compares every member of the results zip with `git_blob(commit, n)`, `commit` being the notebook's current
pin, although the zip holds `results/` as of the commit the run used.
Evidence, reproduced on scratch directories (`ing/`): a results folder whose environment record names `bab7d1b`, containing
`results/steering_controls_qwen0.5b_c2c_debiased.json` as of `bab7d1b`; scratch results pre-loaded with the current file
(`control_labels_shuffled: within hypothesis string`). `kaggle_ingest.py --run r2_qwen3b --no-fold` printed "accepted ...
DONE: everything expected is present and consistent", reported no conflict, and replaced the file by the globally
shuffled one. Files exposed the same way (changed between the old pin and `cdb08d7`, unchanged since): 9 for `bab7d1b`, 21
for `5c31791`, 58 for `3c7b87d`, among them `cross_model.json`, `probe_sensitivity_qwen3b.json` and both Qwen2.5-0.5B
steering-control files. Step 7 would then rebuild the tables and RESULTS.md from them.
Fix: compare with `git_blob(got, n)` (the run's own commit) and treat "differs from HEAD and from the run's commit" as new.

**F6. RESULTS.md claim 3, title and parenthesis, against the report.** "There is no evidence that what a probe reads is what
the answer is computed from": report 3.6.5 gives observational evidence both ways (answer logit against probe score within
string: +0.359 [+0.283, +0.425] for Qwen2.5-1.5B direct causation, -0.262 for Gemma-2-2B, +0.211 for Qwen2.5-3B; "for the
simplest relation they use it in the right direction") and EXPLAINER question 30 says "The evidence is mixed". What is
absent is interventional evidence, in one model, for adjacency, plus weight-vector steering in two models that the report
itself calls uninformative about use. "(the one structural property the answers depend on)": no result shows it is the
only one; 3.6.5 says the answers also track validity for some types.
Should say: "No intervention shows ..." and "a structural property".

**F7. Report 4.2, summary item 6, README "Patching", study question 41: the new explanation misdescribes the pairs that give
the 0.2.** "In the label-changing premise relabellings, D is +0.252 when the hypothesis letter is replaced by a later letter";
"the Yes−No logit is about 0.2 higher when the queried variable has the earlier letter"; "its sign follows which variable
letter is asked about".
Evidence: in all 60 + 60 premise relabellings the hypothesis text is identical in the two prompts (`c2_example.py`); the
queried letter never changes. What changes is its place in the premise: the swap puts it where a later (or earlier) letter
stood. Where the letter itself changes (hypothesis edits) the effect is +0.149 (18 of 27 positive), -0.106 (21 of 33
negative), +0.084 (37 of 60): about 0.1, not 0.2. The direction split is also confounded with hypothesis type in the
label-changing set (all 9 common-effect pairs are later-letter, all 5 common-cause pairs earlier-letter), which the
within-type figure above settles for premise pairs.
Should say: "the Yes−No logit is about 0.2 higher when the variable named in the hypothesis stands earlier in the premise
(premise relabellings, letter unchanged) and about 0.1 higher when the hypothesis names an earlier letter (hypothesis
edits), whether or not that makes the hypothesis valid."

### Minor

**m1. "Checked by audit 1" covers half of claim 1.** `notes/audit_tier1_raw.md`: scope "qwen0.5b + qwen1.5b"; pool-level
0.859. The test-split 0.895 (`14_benchmark_baselines.py`) and the four Gemma-2-2B and Qwen2.5-3B cells appear in no earlier
audit. Both are now recomputed (Section 1). "Audit 5 (four models, recomputed)" in claim 2 is true, but that audit's own
refit gave 0.736 and 0.742, not 0.763 and 0.765.

**m2. Figure under claim 1.** `fig5_confound_c2c_probe.png` shows three models on the Corr2Cause pool: 3 of the "eight
cells", no Qwen2.5-3B, and not the test-split number the claim leads with. In it the hypothesis-string prior is level with
the probe for two models; the bar the claim is about is the green one.

**m3. A smoke test can still be ingested.** The status file is now `round2_status_<run>_tiny.json`, so with `--run <run>`
the ingest's `status.get("tiny")` check never sees it. Reproduced (`ing2/`): without a flag the ingest stops for "missing"
files and recommends `--allow-partial`; with `--allow-partial` it wrote the ten-item
`steering_controls_qwen3b_c2c_debiased.json` into results. `notes/kaggle_ingest.md`: "is refused by the ingest". Fix: refuse
any archive that contains `round2_status_*_tiny.json`.

**m4. The ancestor check.** (i) It takes `git_commit` from the archive and passes it to git as a revision: the string
"HEAD" is accepted as "an earlier round-2 pin" (`ing3/`); require 40 hex characters. (ii) For runs from pins before
`cdb08d7` the status has no `finished`, and `"export" in status` counts as finished: the stale-`export` hole the hardening
closed is open again for exactly those runs. (iii) "accepted, raw outputs are re-analysed locally": the ingest runs no
re-analysis (a manual step in `notes/kaggle_ingest.md`), and for `07c` none is possible: a run from before `bab7d1b` has
globally shuffled control directions in its raw file.

**m5. `results/kaggle_environment.json` no longer describes the Kaggle run.** At `81ef2d7` it recorded Linux, two Tesla
T4s; since `1f06b3d` it records macOS, no GPU, commit `2ff9e0f`: the runner's `environment` stage writes that file name
on every round-2 run, including local ones. The round-1 record survives only in git history and in
`kaggle_ingest_report.json`.

**m6. A smoke test still shares stage outputs with the real run** (code reading; the fourth audit's F1 named it).
`feature_select_*` and `steer_select_*` are keyed by model, tag and position only, and `choose_layer` / `choose_site`
reuse an existing file. A `--tiny` run (stride 6) followed by a full run in the same session gives the full run a
stride-6 layer choice where no selection file is committed (`feature_select_*` exists for Qwen2.5-0.5B only). `--tiny`
stage results also carry the real file names in the results zip (m3).

**m7. Steering controls are of the same kind at one token only.** Report 4.1: "the natural spread of the control
differences of means is now 0.134 s against 0.132 s for the real one at this site". At the final token it is 0.264 s
against 0.113 s (the report's own table). The gold label is balanced within every premise as well (two items per premise,
one valid); a within-string shuffle is not, and premise identity explains 83% of the activation variance at the final
token (52% at the end of the hypothesis). "With controls of the same kind, neither gold direction stands out at either
token" should be "... (at the final token the control family is still 2.3 times wider)"; the per-natural-SD share there is
45%, so the conclusion stands.

**m8. "Stays within the range of matched controls" holds per unit of norm.** Per natural standard deviation the gold-probe
direction exceeds all 20 matched controls in two of six rows of the 4.1 table (Qwen2.5-1.5B original pool, 5.6 times the
median; Qwen2.5-0.5B original pool at the end of the hypothesis, 4.8 times). The second is in no sentence of 4.1.

**m9. "75% of the fair controls stay below" (4.4 point 3), "55% ... stay below" (point 6).** These are shares smaller in
absolute value. Below in the signed sense: 100% and 95% (F1).

**m10. "(correlation over items and controls 0.98 at the final token)"** is the best of four: 0.76 (hyp L13), 0.84 (final
L11), 0.37 (hyp L5).

**m11. `translation_contrast_under_other_sign_choices`**: three of the six variants are the same contrast (point-estimate
signs on all items, on the selection half and on the report half are all +,-,+,-,-). The fifth audit's variants were
about membership by interval on each half; "drop direct causation" is not among them.

**m12. 4.4 point 1, "the difference between them is one of degree".** As a readout, yes (R² 0.96 and 0.79). Causally the
split arm says otherwise: moving along the answer-predictive direction moves the answer by 82% of what the regression
implies at the final token, layer 15, and by 4 to 13% at the end of the hypothesis, layer 13 (3 to 11% at final L11).
"Linearly predictable from this token" is not "formed at this token". This supports claim 3's localisation sentence and
should be in the first table next to R².

**m13. Stale or inconsistent sentences.**
- Report 4.1 "May not" bullet: "the probe is stronger at the end of the hypothesis, which was not steered" (it was, same section).
- Report 7 "Steering": "Debiased-pool controls were shuffled over the whole pool" (now Qwen2.5-1.5B only).
- Report 7 "Models and scale": "all at chance on both tasks" against summary item 1 (Qwen2.5-3B above chance on CLadder).
- Report 3.6.6: "nothing measurable to a stronger text-only model" against 3.6.2 ("a little that this text model lacks").
- Appendix C "With SAEs that fit poorly"; EXPLAINER step 9 "fit these activations poorly", "dictionaries this poor":
  against summary item 8 and the fifth audit's F12.
- EXPLAINER question 30 "that was not done"; question 32 "is what is missing": both were done for Qwen2.5-0.5B.
- EXPLAINER figure guide, fig11/12: bottom row "would mirror them" (the figure now plots change in |gap|; its own caption
  says "below zero"); "near the output".
- EXPLAINER step 10 title "the one thing the models do read" (F6).
- `study/02_questions.md` intro and question 48, `study/01_reading_order.md` session 6: "section 5" is a stub; Appendix C.

**m14. Retraction table.**
- Missing: "at the final token translation and erasure reproduce almost all of the dependence (0.99 of full mediation)",
  withdrawn in report 4.4 point 3 after the fifth audit; "patching shows where the answer is formed" (second audit;
  section retitled).
- Row 5 reads as if the fault were fixed; the second version's eraser leaves adjacency at 0.64 to 0.70 on held-out items.
- Row 8, "a fix reported but not applied ... audit 5": not in the fifth audit's report; the fourth audit's header recorded
  at 18:00 that `gpu_round2.py` was unchanged.

**m15. Limitations missing from RESULTS.md:** probe accuracies are three-seed ensembles; two of the three larger models
have final-token activations only; the eraser removes about half of the readable adjacency on held-out items; the 0.895
interval rests on 21 premises.

### Nit

- 4.2: the same number has two intervals (+0.223 [+0.173, +0.270] in the table, [+0.173, +0.268] in the text).
- 4.4 first table, "(Probe trained on the half the eraser was fitted on)": 0.434 and 0.535 are numerical noise; in float64
  I get 0.565 and 0.453.
- "20 random directions": they are class-mean differences for shuffled labels at the real norm.
- EXPLAINER D30 "Nothing new enters RESULTS.md before an independent audit" against claim 3's "reruns ... not yet audited".
- The runner declares `all_ok` and `finished` when the export wrote nothing (dry run: `zip_megabytes: {}`); the ingest
  catches the missing cache zip.

---

## 3. Verdicts on the three headline claims

**Claim 1. Survives as worded** (number and file right; figure shows 3 of 8 cells). Change the audit clause (m1).
Strongest sentence: "On the official Corr2Cause test split a classifier that sees only the hypothesis reaches AUROC 0.895
[0.863, 0.919] (21 premises); on the balanced pool the hypothesis string alone predicts 86% of labels, and gold-label
probes are 5 to 11 points (Corr2Cause) and 13 to 24 points (CLadder) below the strongest text-only classifier in all eight
model × benchmark cells, every interval excluding zero."

**Claim 2. Survives with rewording** (F2, F3). Strongest sentence: "With every single-input shortcut removed, a linear probe
on the three larger models' activations is as accurate as a lookup of the premise statements about the two queried
variables (0.75 to 0.77 against 0.765), five points below a hand-written parse (0.816), and adds at most one point to the
strongest text-only classifier (0.865). Probe and lookup agree on only 68 to 70% of items, so equal accuracy is not equal
content. Pooled over hypothesis types the models' answers are at chance (AUROC 0.49 to 0.51); within types they deviate in
both directions."

**Claim 3. Survives with rewording** (F1, F6, m8). Strongest sentence: "No intervention shows that what a probe reads is what
the answer is computed from. In Qwen2.5-0.5B, shifting an item at the end of the hypothesis by its type's
adjacent-minus-not difference moves the answer towards No for all five hypothesis types (-0.002 to -0.037 logits), the
observed direction for two types and the opposite for two; the pre-specified contrast (+0.013 against a full-mediation
benchmark of 0.112) is above all 20 label-shuffled directions of the same size and smaller in absolute value than one of
them, and a contrast that ignores the hypothesis type fits as well. At the answer position, layer 15 of 24, the same shift
removes the dependence for common-effect hypotheses beyond all 20 controls and creates one for common-cause hypotheses;
the answer is a linear function of the activation there (held-out R² 0.96), so this locates the answer. Per unit of norm,
steering along the gold-label probe's weight vector stays within the range of 20 matched controls in two models. One model
for the adjacency test."

---

## 4. Sentences in RESULTS.md (`report/RESULTS_template.md`) that should change

1. Claim 1, last sentence → "*Checked by* audit 1 (two Qwen models, on the pool: it found the shortcut and forced the
   retraction of my first positive result) and audit 6 (the test-split number and the Corr2Cause cells of all four models,
   recomputed)."
2. Claim 2, headline → "**2. With every single-input shortcut removed, a probe on the larger models' activations is as
   accurate as a lookup of the premise statements about the two queried variables, no more, and pooled over hypothesis
   types the models answer at chance.**"
3. Claim 2, add after the numbers → "Probe and lookup agree on 68 to 70% of items (a stack of the two scores 0.81), so equal
   accuracy is not equal content; the probe accuracies are three-seed ensembles (single probes one to two points lower).
   Within a hypothesis type the answers are not at chance: for direct causation their AUROC for validity is 0.63
   (Qwen2.5-1.5B), 0.56 (Qwen2.5-3B) and 0.39 (Gemma-2-2B)."
4. Claim 3, headline → "**3. No intervention shows that what a probe reads is what the answer is computed from.**"
5. Claim 3, "(the one structural property the answers depend on)" → "(a structural property the answers depend on, with a
   sign that differs by hypothesis type)".
6. Claim 3, "has an effect that is larger in absolute value than 95% of 20 random directions of the same size: inside their
   range." → "moves the answer towards No for every hypothesis type; the pre-specified contrast (+0.013 of a possible 0.112)
   is above all 20 label-shuffled directions of the same size and smaller in absolute value than one of them, the weakest
   result 20 controls can register, and it does not depend on the hypothesis type."
7. Claim 3, "At the answer position it removes the dependence for one hypothesis type" → "At the answer position (layer 15
   of 24) it removes the dependence for one hypothesis type and creates one for another".
8. Claim 3, "Steering along the gold-label probe direction stays within the range of matched controls." → "Per unit of norm,
   steering along the gold-label probe's weight vector stays within the range of 20 matched controls (two models; per
   natural standard deviation it exceeds all 20 in two of six cases, both on the original pool)."
9. Claim 3, "*Checked by* audits 2, 4 and 5; the reruns after audit 5 are not yet audited." → "*Checked by* audits 2, 4, 5
   and 6 (the reruns after audit 5 recomputed from the raw files)."
10. Retraction table, row 5, second cell → "Its 'removal' left the property readable (AUROC 0.84); the second version still
    leaves 0.64 to 0.70 on held-out items".
11. Retraction table, add → "| 'At the answer position the intervention reproduces almost all of the dependence' | Errors of
    opposite sign cancelled in the summary; one gap was created, not removed | Audit 5 |" and "| 'Patching shows where the
    answer is formed' | The models do not prefer the valid prompt of a pair; it shows where a prompt difference arrives |
    Audit 2 |".
12. Retraction table, row 8, last cell → "Ingest check, own review, audit 4".
13. Limitations, add → "Probe accuracies are three-seed ensembles. For Gemma-2-2B and Qwen2.5-3B only final-token activations
    exist. The eraser removes about half of the linearly readable adjacency on held-out items. The 0.895 interval rests on
    21 test premises."

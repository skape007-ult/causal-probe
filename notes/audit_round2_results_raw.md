# Audit of the round-2 RESULTS and the claims drawn from them (raw report)

Reviewer: independent, results only (the code review is `notes/audit_round2_code_raw.md`). Written 13 Nov, 19:30 to 20:10.
No language model was loaded. Everything below was recomputed from `cache/*/c2c_debiased/resid.npy`, `meta.json`,
`cache/qwen0.5b/c2c_debiased/steering_adjacency_*.npz`, `steering_controls*.npz`, `cache/qwen0.5b/patching*.npz` and
`data/raw/corr2cause/train.csv` (streamed; tokenizer only) with my own scripts in
`/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit5/`
(`a0_basic.py`, `a1_leace.py`, `a2_effects.py`, `a3_more.py`, `b_patching.py`, `b2_rebuild.py`, `d_probe.py`).

**Versions.** The documents changed under me at 19:43 (commit `1d51dd2`, after the `hyp_best` rerun with the transfer block).
"v1" = the wording I was briefed on (report 4.4 before 19:43; log entry 19:20: "specific to the encoding", "some of the
information a probe reads there is used downstream"). "v2" = the wording now in `report/report.md` 4.4 point 5 and the
May / May-not bullets, the README adjacency bullet, EXPLAINER step 10 and question 37. Each A(2) finding says which version
it hits. The new `hyp_best` raw file is identical to the old one on the 219 shared columns (max difference 0.0), so every
number I computed on the old file holds for the new one.

**One thing I did wrong.** `a1_leace.py` first wrote two scratch arrays (`recon_final_best.npz`, `recon_hyp_best_old.npz`,
17 MB each) into the repository root (relative path, 19:32). They are matched by `.gitignore` (`*.npz`), were not in commit
`1d51dd2` (`git ls-files | grep recon_` is empty), and I moved them to the scratch directory at 19:44. Nothing else in the
repository was written except this file.

---

## 1. What I verified and found correct

- **Raw file to JSON to table, adjacency test.** From the raw arrays I reproduce: translation contrast +0.1102 (final, L15)
  and +0.0129 (end of hypothesis, L13); erasure contrast +0.0959 and +0.0156; benchmarks 0.112 / 0.105; per-template
  translation effects (+0.040, -0.044, -0.037, -0.255, -0.183 and -0.009, -0.002, -0.003, -0.037, -0.026); control ranges;
  both transfer matrices; cosines between class-mean differences. `delta` and `W` in the raw files are reproduced from the
  cache to 0.0. All 500 evaluated items are report-half items; 51/59/60/61/56 of 100 are adjacent.
- **Linearity (the assumption behind "size-matched").** Small-step / large-step slope ratio for the real class-mean
  direction 0.997 (final) and 1.031 (hyp); for the 20 control class-mean directions 0.96 to 1.01 (final) and 0.99 to 1.02
  (hyp, the 9 with a measurable slope). Item-level correlation of the two slopes 0.99 to 1.00. Rescaling a control effect
  by a factor of 2 to 3.5 is justified. This part of the design is sound.
- **Lambda.** The erasure contrast predicted for lambda = 1e-4 ... 1 (and for a plain mean-difference projection) is
  +0.091 to +0.133 (final) and +0.0146 to +0.0155 (hyp). The result does not depend on the LEACE regulariser.
- **Observational gaps.** My block bootstrap (own seed) gives parent +0.050 [+0.003, +0.098], ancestor -0.042 [-0.105,
  +0.015], descendant +0.055 [+0.023, +0.087], collider -0.230 [-0.282, -0.179], confounder -0.113 [-0.193, -0.040]; the
  report's intervals agree to 0.003. The within-string AUROC table at the top of 4.4 is reproduced exactly for Gemma-2-2B
  and Qwen2.5-3B (my own pairwise-concordance code).
- **Final token, L15, is robust to a control that can fail** (see F1): contrast 0/20 controls larger, collider 0/20,
  confounder 1/20.
- **Steering controls (4.1).** All slopes and percentiles in the controls table reproduce from `steering_controls*.npz`.
- **Patching.** Mean D, mean |D|, fractions, the 31/29 split and the within-template numbers reproduce from the npz files.
  The rebuilt label-changing pair list matches the saved premise ids for both kinds.
- **Section 3.6, four models.** Every number in the prose I checked matches `results/probe_content_*` and
  `results/confound_*` (probe accuracies, differences to bag-of-words / lookup / hand parse, incremental accuracy,
  adjacency AUROCs, stratified AUROCs, answers by template). Independent recomputation with my own folds and classifier
  code: Qwen2.5-3B adjacency AUROC 0.978 (report 0.980), random-init 0.604 (0.601); Gemma 0.961 (0.963), 0.601 (0.605);
  hand parse 0.810 (0.816); my own six-feature per-template pair lookup 0.765 (0.765); share of answer-logit variance
  explained by the hypothesis string 0.874 and 0.943 (identical).
- **"Qwen2.5-3B is the only model above chance on CLadder"** survives the two-draw problem: on the 2,271 items shared by
  both draws its accuracy is 0.532, against 0.494, 0.500 and 0.506 for the other three.
- **CLadder caches.** `qwen3b/cladder`, `qwen3b-rand/cladder` and `qwen3b/cladder_sub_fewshot` hold the other draw (2,271
  of 3,000 shared; 386 of 500); all Gemma and local Qwen CLadder caches hold the pinned draw. Same 47 stories, label
  balance 0.500 in both.

---

## 2. Findings

### Critical

**F1. The erasure arm is the translation arm again, and its control cannot fail. "Outside all 20 size-matched control
erasures" does not survive a control that can.** (v1 and v2.)
Where: `report/report.md` 4.4 point 5 ("for common-effect hypotheses the gap changes by +0.042 [+0.034, +0.051] and for
common-cause hypotheses by +0.028 [+0.024, +0.032], both outside all 20 size-matched controls"); May bullet ("where equally
large control erasures do nothing"); README adjacency bullet ("where 20 equally large control erasures remove between
-0.003 and +0.008"); EXPLAINER step 10 and question 37 ("where equally large control erasures remove nothing");
`results/adjacency_*_hyp_best.json` key `erasure/contrast_reduction_of_dependence/vs_controls_norm_matched`.
What is wrong, in two steps.
(a) The LEACE edit of item i is `-c_i * delta_g` with `c_i = (w_g.x_i)/(w_g.delta_g)`; the translation is `+/- delta_g`.
The response is linear, so the erasure effect is `-c_i` times the translation effect. Measured: per-item correlation
between the erasure effect and `-c_i x (translation effect)` is r = 0.998 (final) and 0.94 (hyp), slope 0.999 and 0.98;
the five changes in gap are predicted to the third decimal (hyp: observed +0.0058, -0.0011, +0.0022, +0.0425, +0.0278;
predicted +0.0066, -0.0004, +0.0021, +0.0424, +0.0280). Erasure and translation are one measurement of one number, the
answer's sensitivity along `delta_g`. They are presented as two lines of evidence, and where they disagree with their
controls (translation: one control larger; erasure: none) the report sides with the erasure.
(b) They disagree only because the controls differ. A control erasure moves each item by a coefficient `w_c.x_i` that is
unrelated to adjacency (labels were shuffled), so it cannot change the adjacent-minus-not gap beyond sampling noise,
whatever its direction does to the answer. Rescaling by 2.4 to 3.5 does not change that. The control SD is 0.0023 for
that reason, not because "equally large" edits are harmless: the same rescaled control erasures move individual answers by
0.011 to 0.022 logits on average, more than the real erasure does (0.0097).
My evidence: a control that can fail keeps the real, adjacency-correlated coefficient `c_i` and replaces the direction by
each of the 20 shuffled-label class-mean directions at the norm of `delta_g` (computed from the step columns of the same raw
file; the method reproduces the real erasure at r = 0.93 to 0.998). At the end of the hypothesis: contrast +0.0156 against
controls -0.024 to +0.009, 1 of 20 larger in absolute value; common effect +0.0425 against -0.060 to +0.053, 3 of 20 larger;
common cause +0.0278 against -0.037 to +0.047, 4 of 20 larger. That is the translation result (95%, 85%, 80%), as it must be.
Should say: "Erasure and translation measure the same quantity. Against random directions of the same size applied in the
same way, the end-of-hypothesis effect is exceeded in absolute value by 1 of 20 controls for the contrast and by 3 and 4 of
20 for the two hypothesis types that carry it. Against the sampling noise of the class-mean difference itself (controls as
fitted) the contrast is +0.013 with a control SD of 0.003." Delete "outside all 20" for this site in all three documents.

**F2. "About a sixth" (hyp) and "0.096 of 0.105" / "the whole dependence" (final) are produced by a contrast that books a
move AWAY from zero as a reduction.** (v1 and v2.)
Where: report 4.4 points 2 and 5, May bullet ("About a sixth of that dependence is removed"; "By layer 15 the whole
dependence is present at the final token in linear form"); README ("removes 0.016 of the 0.105 logits"; "Near the output the
whole dependence is present"); EXPLAINER step 10 and Q37; JSON key `erasure/contrast_reduction_of_dependence`.
What is wrong: the contrast is mean over four templates of (predicted sign) x (gap before - gap after), with the sign taken
from all 2,400 items and the gap from the 500 evaluated ones, pooled over strings. For common-cause hypotheses the predicted
sign is minus but the evaluated items' gap is +0.031 (pooled; -0.010 within string). The erasure pushes it to +0.059 (hyp)
and +0.160 (final), further from zero, and that is counted as +0.028 and +0.131 of "dependence removed".
My numbers (evaluated items):

| Site | Template | Gap before (pooled / within string) | After | What happened |
|---|---|---|---|---|
| hyp L13 | parent | +0.097 / +0.077 | +0.103 / +0.084 | larger |
| hyp L13 | descendant | +0.042 / +0.056 | +0.044 / +0.057 | unchanged |
| hyp L13 | collider | -0.313 / -0.264 | -0.270 / -0.221 | 14% smaller |
| hyp L13 | confounder | +0.031 / -0.010 | +0.059 / +0.018 | further from zero |
| final L15 | parent | +0.097 / +0.078 | +0.071 / +0.055 | 27% smaller |
| final L15 | descendant | +0.041 / +0.054 | +0.061 / +0.076 | 48% larger |
| final L15 | collider | -0.312 / -0.263 | -0.065 / -0.006 | 79% (98%) smaller |
| final L15 | confounder | +0.029 / -0.011 | +0.160 / +0.122 | a gap created |

Mean absolute gap over the four contrast templates: hyp 0.1207 -> 0.1190 (1.4% smaller; within string 6.6%; cluster
bootstrap of the reduction +0.0016 [-0.0004, +0.018], includes zero); final 0.120 -> 0.089 (25%; within string 36%;
[+0.0004, +0.085]). String-centred AUROC of the answer for adjacency, before -> after: hyp collider 0.264 -> 0.299,
confounder 0.474 -> 0.501, the rest unchanged to 0.005; final collider 0.267 -> 0.504, confounder 0.473 -> 0.604 (overshoot),
parent 0.542 -> 0.518, descendant 0.553 -> 0.583, ancestor 0.415 -> 0.426.
The translation share "0.99" is the same kind of coincidence: +0.040 against an observed +0.050, -0.037 against +0.055 (wrong
sign), -0.255 against -0.230, -0.183 against -0.113 (162%). Errors of -167% and +62% cancel in the average.
Should say: "At the end of the hypothesis the erasure shrinks the common-effect gap by 14% (-0.313 to -0.270) and does not
shrink any other; the mean absolute gap falls by 1 to 7%, not distinguishable from zero. At the final token (layer 15) it
removes 79% of the common-effect gap, 27% of the direct-causation gap, enlarges the descendant gap and creates a positive
gap for common-cause hypotheses; the mean absolute gap falls by a quarter to a third." "The whole dependence" and "almost
all" should go; "about a sixth" should go from the README, the EXPLAINER and the May bullet.

### Major

**F3. "The erasure removes the information" is measured on the half the eraser was fitted on. On the evaluated items
adjacency is still readable at AUROC 0.64 to 0.70.** (v1 and v2; also EXPLAINER Q38.)
Where: report 4.4 point 1 ("a fresh probe reads adjacency at AUROC 0.876 before and 0.434 after the erasure ... 0.926, 0.535");
design item 2 ("Whether the erasure worked is measured, not assumed"); first table, column "after LEACE erasure"; EXPLAINER
Q38 ("The second version measures what the erasure did to the representation"); JSON `leace/adjacency_auroc_of_fresh_probe_report_half/after_erasure`.
What is wrong: the fresh probe is trained on erased SELECTION-half activations, where LEACE has set the class-mean
difference to exactly zero in sample (I get 0.000 for all five templates), and tested on the report half. A probe trained
where nothing is left learns noise; its report-half AUROC lands on either side of 0.5 (0.319 for ancestor at the final
token, 0.637 at the end of the hypothesis), which is itself a sign that information remains. Lambda was chosen by the same
protocol.
My numbers, erased activations:
- probe cross-validated within the erased report half (4 block folds, 5 fold seeds): pooled 0.698 to 0.705 (final; 0.858 to
  0.867 before) and 0.701 to 0.705 (hyp; 0.912 to 0.920 before); per-template probes up to 0.80 (ancestor, final) and 0.86
  (confounder, hyp);
- probe trained on the 748 erased report-half items that were not evaluated, tested on the 500 evaluated erased items:
  0.650 (final; 0.852 before) and 0.640 (hyp; 0.912 before);
- report-half class-mean difference after erasure, in units of s: ancestor 0.368 (0.399 before) at the final token: not
  erased at all;
- no lambda on the grid does better (0.67 to 0.71 at both sites).
So the eraser, fitted on 49 to 197 items per class in 896 dimensions, removes roughly half of the excess AUROC on held-out
items. The first version's fault ("left adjacency readable at 0.84") is reduced, not removed.
This cuts both ways and should be said: at the final token the common-effect gap is nearly gone while adjacency is still
readable at 0.65 to 0.70 on the same items, which supports the localisation reading of A(1) better than the transfer check
does; at the end of the hypothesis "erasing adjacency removes x" is "an edit that removes about half of the linearly
readable adjacency removes x".
Should say (point 1): "On held-out items the erasure lowers the AUROC of a fresh probe from 0.86 to 0.70 at the final token
and from 0.92 to 0.70 at the end of the hypothesis (probe cross-validated on erased report-half activations); the 0.434 and
0.535 in the table come from a probe trained on the half the eraser was fitted on and overstate the erasure."

**F4. "Here the answer is not yet formed at the steered token" is contradicted by the cache, and the heading of point 5
contradicts its own last sentence.** (v2; the first sentence was also in v1.)
Where: report 4.4 point 5, heading "*a small part of the dependence, specific to the encoding*" and sentence 1.
Evidence: a ridge regression (alpha = 1) of the string-centred answer logit on the string-centred activation at the last
hypothesis token, layer 13, fitted on the selection half, predicts the report half with R^2 = 0.79 (r = 0.89). The same at
the final token, layer 15: R^2 = 0.96. The difference between the two sites is one of degree. The regression direction
assigns the class-mean differences answer shifts of +0.044, -0.034, +0.064, -0.191, -0.147 (parent ... confounder), i.e. the
observed template-specific pattern is already linearly present at the end of the hypothesis; translating by those
differences moves the answer by -0.009, -0.002, -0.003, -0.037, -0.026, about a fifth of it for the two large types and
nothing (wrong sign) for the two small ones. That is what "an answer precursor, partly passed on" predicts and it is what
the v2 May-not bullet now says. The heading still says "specific to the encoding", which F1 removes.
Should say: heading "a small effect, not specific to adjacency"; sentence 1 "The within-question answer is already 79%
linearly predictable from this token at this layer (96% at the final token, layer 15)".

**F5. The contrast's membership and signs are fragile, the two statistics in Section 4.4 disagree about direct causation,
and a template-blind alternative fits the end-of-hypothesis data as well.** (v1 and v2.)
Where: report 4.4 Design ("the average over hypothesis types whose observed gap excludes zero"), second table.
Evidence: (i) parent enters on an interval of [+0.003, +0.098] (mine; report [+0.005, +0.096]); on the selection half alone
the interval is [-0.033, +0.122]; and the AUROC table of the same section gives 0.531 [0.489, 0.573] for the same
dependence, which includes 0.5. (ii) confounder is -0.157 on the selection half and -0.068 [-0.186, +0.044] on the report
half, -0.010 on the evaluated items; the "benchmark" inherits that uncertainty. (iii) signs are computed on all 2,400
items, the 500 evaluated ones included; with selection-half signs parent drops out, with report-half signs confounder does.
(iv) hyp L13 translation contrast under other choices: drop parent +0.020; collider and confounder only +0.032; all five with
point-estimate signs +0.011 (2 of 20 controls larger); uniform sign "adjacent -> No" for all five +0.0155 (1 of 20 larger).
All five translation effects at the end of the hypothesis are negative; for the two types with positive observed gaps the
effect has the wrong sign. A single "towards No" push fits as well as the pre-specified template-specific pattern.
(v) the intervals quoted for contrasts ([+0.012, +0.014]) resample evaluated items only; they are conditional on the fitted
directions and on the signs and are not the uncertainty of the claim. The relevant spreads are the control SDs: 0.003 (as
fitted) and 0.008 (same norm). (vi) 20 shuffles give a smallest one-sided p of 1/21; the translation contrast is above all
20 one-sided and below one two-sided, at one of four sites and three arms, unadjusted.
Should say: state (iv) next to the contrast, and call the tight intervals "conditional on the fitted directions".

**F6. Label-preserving pairs: the "+0.223 with no label change" is an artefact of how the pairs are ordered, and the
explanation given for it is refuted by the label-changing pairs.** (Report 4.2 as it stands, including the new last sentence.)
Where: report 4.2 "Label-preserving pairs": "For the relabelled premises there is even a consistent direction with no label
change at all ... +0.223 [+0.173, +0.270] ... Relabelling puts the variable letters out of alphabetical order, and the model
reacts to that."; table header "Mean D = LD(first) - LD(second)"; `notes/log.md` 19:20.
What is wrong: `patch.build_pairs(label_mode="same")` keeps a pair only if the swapped hypothesis sorts after the original
(`q2 < q: continue`), so in all 60 label-preserving pairs the hypothesis letter is replaced by a LATER letter. In the
label-changing pairs the direction is mixed (27 later, 33 earlier; I rebuilt the list, it matches the saved premise ids).
The premise swap itself is symmetric in the two letters, so every relabelled premise is "out of alphabetical order" in the
same way in both directions.
My numbers (D = original - edited):

| Kind | Pairs | Later-letter swaps | Earlier-letter swaps |
|---|---|---|---|
| prem, label changes | 60 | +0.252 (n = 27, 27 of 27 positive) | -0.165 (n = 33, 24% positive) |
| prem, label preserved | 60 | +0.223 [+0.173, +0.274] (all 60) | none in the set |
| hyp, label changes | 60 | +0.149 (n = 27) | -0.106 (n = 33) |
| hyp, label preserved | 60 | +0.084 [+0.010, +0.161] (all 60) | none in the set |

So the direction effect is present, equally large, in the label-changing pairs (+0.252 [+0.181, +0.335] against +0.223); it
averages to +0.02 there only because both directions are sampled. Out-of-order relabelling cannot be the cause, since the
earlier-letter half of the same relabellings moves the answer the other way. What the data show is that the Yes-No logit is
about 0.2 higher when the queried variable holds the earlier-lettered role.
The paragraph's main inference ("the model responds to the edit, not to the change of label") survives, and on better
evidence than the |D| comparison: matched on direction, the signed D of label-changing minus label-preserving pairs is +0.03
(prem) and +0.07 (hyp), parent only: +0.246 against +0.296 (prem). The |D| comparison is a second-order statistic: adding a
validity shift of 0.1 logits to the label-preserving pairs raises mean |D| by 0.011, a shift of 0.2 by 0.055; the reported
interval for the within-type difference, [-0.105, +0.061], cannot exclude a label effect of 0.2 logits (80% of mean |D|).
"No difference in either" should be "no detectable difference". The two pair sets also share 21 premises, so the bootstrap
that treats them as independent samples is slightly off.
Should say: "In premise relabellings the sign of D follows the direction of the letter change (27 of 27 later-letter swaps
positive, 25 of 33 earlier-letter swaps negative, in the label-changing pairs). The label-preserving set contains
later-letter swaps only, by construction, hence its mean of +0.223; matched on direction the two kinds do not differ
(+0.252 against +0.223). The model responds to which variable is asked about, not to validity." Delete the alphabetical-order
sentence. Head the table column "LD(alphabetically earlier hypothesis) - LD(later)".

**F7. "Fully accounted for by what text-only classifiers extract" is stronger than the result files.**
Where: README "What this shows": "a linear probe's above-chance accuracy for the gold label is fully accounted for by what
text-only classifiers extract from the prompt"; report 3.6.2 "no clear excess over text in any model"; 3.6.6 "nothing
measurable to a stronger text-only model"; summary item 4 "nothing to a stronger text-only model".
Evidence, `results/probe_content_*_c2c_debiased.json`, `incremental_value/over_strongest_text_only/gain_auroc`: Qwen2.5-1.5B
+0.0027 [+0.0006, +0.0048], Gemma-2-2B +0.0032 [+0.0012, +0.0056], Qwen2.5-3B +0.0045 [+0.0024, +0.0067]; all three intervals
exclude zero (0.5B: +0.0001 [-0.0004, +0.0006]). The report quotes the accuracy gains only. Same files, `stratified`: inside
25 cells of the two text scores the probe keeps AUROC 0.607 [0.550, 0.664], 0.661 [0.603, 0.715], 0.656 [0.605, 0.708], all
above 0.5.
Should say: "almost entirely accounted for: over the strongest text-only model the three larger models' probes add 0.003 to
0.004 AUROC (intervals exclude zero) and zero to one point of accuracy (intervals reach zero)."

**F8. Steering controls on the debiased pool are not matched in kind, and the end-of-hypothesis paragraph leaves out the
one direction that exceeds all of them.**
Where: report 4.1 "Steering at the end of the hypothesis" ("So for this model the earlier site gives the same reading as the
final token"); summary item 5 ("The difference-of-class-means direction moves the answer, about as much as differences of
means for shuffled labels"); controls table.
Evidence: (i) `results/steering_controls_qwen0.5b_c2c_debiased_hyp.json`: gold difference of means +0.287 [+0.262, +0.313],
larger than all 20 shuffled-label differences of means (max 0.215; 4.3 times their median), positive for 97% of items. On
the original pool at the final token it is also at 100% (0.865 against a maximum of 0.628). Two of six rows contradict
"about as much as". (ii) `07c_steering_controls.py` shuffles labels over the whole pool (`probes.shuffled_labels`), not within
hypothesis string. On the debiased pool the gold label is balanced within every string and premise and a global shuffle is
not, so the control directions carry string identity. The table shows it: natural SD along the gold difference of means
0.132 s against a control median of 0.509 s at the end of the hypothesis (0.113 against 0.369 at the final token). The
code review's F9 fixed this for script 17 only. "Within the range of its controls" on the debiased pool is therefore a
comparison with a wider family than the real direction belongs to.
Should say: add the difference-of-means row to the paragraph; say that the debiased-pool controls are shuffled globally;
change summary item 5 to "about as much as ... in four of six cases, more than all 20 in two".

**F9. The CLadder two-draw account is right in substance but the affected numbers are not flagged where they are used, and
"same stratification" is not exact.**
Where: report Section 1 item 1; tables at report lines 87, 123, 126, 145, 167, 168, 233, 278, 297, 323, 324, 359 to 362, 384
(line numbers of the 19:43 version; every Qwen2.5-3B CLadder row in 3.1 to 3.4); EXPLAINER D28 ("same stratification, same size"); report Section 7.
Evidence: the word "draw" or "D28" occurs in the report only at 4.1 (Gemma CLadder steering), 5.1 (SAE) and Section 7. The
two draws differ in: rung 1 / rung 2 counts 938/936 against 936/938; `exp_away` / `collider_bias` 48/46 against 46/48; nine of
ten graph types (confounding 296 against 285, IV 229 against 216, fork 273 against 284); 43 of 47 stories; nonsense-story
items 1,147 against 1,155. Not listed in D28: the random-init control `qwen3b-rand/cladder` (same Kaggle draw, so the
comparison is internally consistent) and the 500-item subset (386 shared). `steering_analysis_gemma2b_cladder.json` analyses
228 of the 300 steered items; the report says "its slopes are exact" without the 228.
Should say: mark the Qwen2.5-3B CLadder rows (a dagger and one line under each table); in item 1 add "on the other draw;
0.532 on the 2,271 shared items, where the other models score 0.49 to 0.51"; in D28 "same strata definition; graph-type
and story composition differ by up to 13 items per graph type".

### Minor

**F10. "Near the output" is layer 15 of 24.** README adjacency bullet; EXPLAINER step 10 and Q37 ("Near the output, the
answer is almost a linear function ..."); report 4.4 point 3 ("Near the end of the network"). Relative depth 0.62, nine
layers before the unembedding. The linearity is an empirical fact here (R^2 0.96; slope ratio 1.00), not a consequence of
being near the output. Say "at the final token from layer 15 on".

**F11. The transfer check at the final token is a restatement of linearity, not a second check.** Report 4.4 point 3(b).
With a response that is linear with the same gradient for all items (ratio 1.00, per-item SD of the translation effect
0.007 to 0.014), the effect of template h's vector on any item is gradient x delta_h; constant columns follow. It cannot
come out otherwise once the linearity column of the table is known. At the end of the hypothesis it is informative, and
there the columns are less constant than the text says: the collider vector moves descendant items by -0.010 and the other
four types by -0.037 to -0.057; the confounder vector -0.013 against -0.023 to -0.039. "By a similar amount" should be "with
the same sign; descendant items respond a quarter to a third as much".

**F12. SAE section (5.1), sentence by sentence against `results/sae_ablation.json` / `sae_features.json`.**
- "changes the answer logit difference by 0.223 to 0.442 logits on items of the features' own graph type": mediation is
  0.190. The range is 0.190 to 0.442.
- "The effect is not specific to the graph type: it is as large on items of other graph types"; "equally for all graph
  types" (summary item 7, README, EXPLAINER step 9): false for collision, 0.223 [0.199, 0.241] on collision items against
  0.137 [0.111, 0.188] on others (top minus random +0.097 against +0.025). True for the other four. (The intervals resample
  5 stories and are fragile.)
- "The change is mostly a uniform shift of the logit difference in one direction": signed / absolute change is 0.97
  (confounding), 0.95 (collision), 0.72 (mediation), 0.51 (chain), 0.39 (fork). Not "mostly" for the two largest effects.
- "somewhat more than removing matched random features": one random set (mediation, set 2) moves the answer by 0.512, more
  than any selected set (max 0.442). Four of five selected sets exceed their own three random sets; with three sets that
  is all that can be said, as the caveat states. Random sets are not disjoint from other types' selected features (16346,
  13295, 2804, 12877, 13897, 7507 occur in them).
- "I found no feature that is specific to one graph type": the nearest is block 18, end of context, feature 4602: fires on
  85% of collision items and 38% of others, held-out AUROC 0.78, all eight top examples collision. Its top examples are
  `collider_bias` and `backadj` questions, so it tracks question wording, which fits the report's reading; it should be named,
  and the ablation (block 12 only) did not test it.
- Reading of the reconstruction numbers ("The SAEs fit these activations poorly", attributed to base-versus-IT mismatch):
  the relative squared error is 0.16 to 0.26 and the cosine 0.86 to 0.92; `fraction_variance_unexplained` in `11_sae.py`
  divides the error by the variance ACROSS THESE ITEMS at one token, a small part of the squared norm. 0.65 to 0.87 says the
  reconstruction error is comparable to the between-item signal at that token; it does not show that the dictionaries fit
  worse here than on their own training distribution, and no in-distribution reference was computed. Should say:
  "reconstruct 74 to 84% of the squared norm but only 13 to 35% of the variance between items".
- `results/sae_top_examples.md` announces "'My reading'" labels; the file contains none.

**F13. Gemma's isotropic maximum is misquoted.** Report 4.1 "Gemma-2-2B and Qwen2.5-3B": "single isotropic directions in the
same runs reach +0.553 (Gemma-2-2B)". `steering_analysis_gemma2b_c2c_probe.json`: random slopes 0.969, -0.094, 0.063. On the
original pool one isotropic direction reaches +0.969, four times the gold-probe slope (+0.247); +0.553 is the debiased pool.

**F14. A(3) wording.** "about a thousandth of a logit at every site": 0.0002 to 0.0033; at the end of the hypothesis four
of five types are at 0.002 to 0.003. "the steering nulls of Section 4.1 ... were uninformative for the same reason": the
reason (a natural-size step along a read-out direction is tiny: the class gap along the probe is 5 to 48 times shorter
than the class-mean difference) explains 4.1's natural-scale statement, not its main comparison, which used steps of one
s (63 to 322 natural SDs) and compared slopes per unit norm. README and EXPLAINER step 7 go further ("a later experiment
showed why it could not: a probe's weight vector is not where the information lives"): shown for adjacency probes in one
model (AUROC 0.873 after removing the direction), asserted for gold-label probes in four.

**F15. "A little more is left over in the two largest models than in Qwen2.5-1.5B"** (report 3.6.4): 0.661 [0.603, 0.715]
and 0.656 [0.605, 0.708] against 0.607 [0.550, 0.664]. Not distinguishable.

**F16. The headline probe accuracies are three-probe ensembles.** Report 3.6 robustness paragraph says so for Qwen2.5-1.5B
("one to two points lower"). For the two new models single fold assignments are 0.746 to 0.759 (Gemma) and 0.754 to 0.763
(Qwen2.5-3B) in the JSON, and 0.736 and 0.742 with my own fold assignment; the text baselines gain less from the same
averaging (hand parse 0.810 single against 0.816; lookup 0.765 against 0.765). "Indistinguishable from the per-template
lookup" holds for the ensemble; a single probe is 0.2 to 2.9 points below it. Say so for all three models.

**F17. Stale or inconsistent sentences (claims pass, part F).**
- Report Section 1 item 4: reports two models ("Pooled, both models answer at chance") where 3.6 and the README report four.
- Section 1 item 5: "At the probed layer and final token"; the end-of-hypothesis site exists and is in the README.
- Section 1 item 6: "in both Qwen models"; 4.2 and the README have three models.
- Section 1 has no item for the adjacency test or the label-preserving pairs, which the README and EXPLAINER report as
  results; item 8 still lists them as "reported where its results have arrived".
- Section 7 "Steering. One layer ..., final token only": no longer true for Qwen2.5-0.5B.
- README "exactly at chance on the debiased item set": AUROC 0.49 to 0.51 with intervals; "indistinguishable from".
- README steering bullet "at the final token or at the end of the hypothesis ... (Qwen models on the laptop ...)": the end
  of the hypothesis was steered for Qwen2.5-0.5B only.
- EXPLAINER step 10 "the answer ... is complete a few layers later" and the one-sentence version "can be followed from the
  end of the question to the output": two sites, one of them at 62% depth; see F2 for "complete".
- EXPLAINER step 6 "each of these models has, internally, done roughly that lookup": equal accuracy is shown, not equal
  content; the probe keeps AUROC 0.72 to 0.76 inside quintiles of the parse score.
- Report 4.4 History "before any GPU time was spent on it": the first version had run at two sites on the laptop's GPU
  (`results/feature_causal_*`, 17:45 and 18:02). "Before any Kaggle time".
- Report 4.4 pre-registration bullet: "each makes the positive reading harder to reach" is true of the transfer arm; the
  switch to LEACE with a control that cannot fail (F1) made the positive reading easier.

### Nit

- `gap_eval` in `17_adjacency_test.py` pools over strings while the observational gap and the predicted signs are within
  string; for confounder the two have opposite signs on the evaluated items (+0.031 against -0.010). Use one definition.
- Report 4.4 first table: the columns "after LEACE erasure" for per-template values contain 0.319 and 0.637; a reader who
  sees values on both sides of 0.5 should be told why (F3).
- Report 4.2: "label-preserving valid/valid pairs exist only for direct-causation and common-effect hypotheses": the
  label-preserving set also has 8 `child` pairs that the label-changing set lacks; they are dropped from the within-type
  comparison, which the text does not say.

---

## 3. Verdicts on claims A(1) to A(3)

**A(1). Final token, layer 15: translation and erasure reproduce / remove almost all of the dependence; this is a
localisation, not evidence of an adjacency feature.**
Verdict: **survives with rewording.** The negative half (not evidence of an adjacency feature) survives and gains support
from F3 (the common-effect gap goes while adjacency stays readable at 0.65 to 0.70) and from R^2 = 0.96; the transfer matrix
adds nothing beyond linearity (F11). The quantitative half does not survive: "almost all", "0.99", "0.096 of 0.105", "the
whole dependence" are averages of compensating errors (F2), and "near the output" is wrong (F10).
Strongest sentence the evidence supports: "At the final token from layer 15 (of 24) the within-question answer is a linear
function of the activation (held-out R^2 0.96). Shifting an item by a hypothesis type's adjacent-minus-not mean difference
moves the answer by -0.26 and -0.18 logits for common-effect and common-cause hypotheses, the sign and rough size of the
observed gaps (-0.23, -0.11) and beyond every size-matched control; for the other three types the shifts (0.04) are inside
the control range and one has the wrong sign. Erasure removes four fifths of the common-effect gap on held-out items and
does not reduce the others, while adjacency itself stays readable at AUROC 0.65 to 0.70. Each type's class difference acts
as that type's answer shift on any item; nothing here indicates an adjacency variable."

**A(2). End of hypothesis, layer 13.**
- v1 ("erasing adjacency removes about a sixth of the dependence, outside all 20 size-matched control erasures; some of
  the information a probe reads there is used downstream"; heading "specific to the encoding"): **does not survive.**
  "Outside all 20" is an artefact of a control that cannot fail (F1); "a sixth" is 1 to 7% of the absolute dependence with
  an interval that includes zero, one type moving away from zero (F2); "adjacency erased" is half erased (F3); "the
  information a probe reads" is not what was moved (the probe direction is inert and has cosine 0.06 to 0.19 with the
  class-mean differences).
- v2 ("About a sixth of that dependence is removed by erasing the linear adjacency information ..., where equally large
  control erasures do nothing; so part of what distinguishes adjacent from not-adjacent items at that token is passed on to
  the answer"; "behaves like an early part of the answer signal ... not like a variable"): **survives with rewording.** The
  conclusion clause and the May-not bullet are supported; the three premises in front of it are not (F1, F2, F3), and the
  heading "specific to the encoding" and "the answer is not yet formed at the steered token" must go (F4).
Strongest sentence the evidence supports: "At the last token of the hypothesis, layer 13, the within-question answer is
already 79% linearly predictable. Shifting an item by its type's adjacent-minus-not mean difference moves the answer
towards No for all five hypothesis types (-0.002 to -0.037 logits), which is the observed direction for common-effect and
common-cause hypotheses (16% and 23% of their observed gaps) and the opposite of it for direct-causation and descendant
hypotheses. The pre-specified contrast is +0.013, four control SDs above what the sampling noise of the mean difference
gives, and is exceeded in absolute value by 1 of 20 random directions of the same norm; per type, by 3 to 4 of 20. So the
difference between adjacent and not-adjacent items at this token has a small causal effect on the answer that is no larger
than that of an arbitrary direction of equal size, does not depend on the item's hypothesis type, and lowers the absolute
dependence of the answers on adjacency by 1 to 7% (interval includes zero). One model, one layer."

**A(3). The probe's weight vector is inert and this is uninformative about use.**
Verdict: **survives with rewording** (F14). Supported: effects across the class gap are 0.0002 to 0.0033 logits and inside
the control ranges at every site; removing the direction leaves adjacency at 0.873 / 0.918. Worth adding, because it makes
the point sharper: per unit of norm the probe direction moves the answer as much as the class-mean difference does at the
end of the hypothesis (+0.032 against -0.028, opposite signs); the whole difference is that the class gap along it is 5 to
16 times shorter. Not supported as worded: "about a thousandth ... at every site"; "the steering nulls of Section 4.1 were
uninformative for the same reason" (true of their natural-scale statement only); README / EXPLAINER "a later experiment
showed why it could not".
Strongest sentence: "Moving an item across the class gap along the adjacency probe's weight vector changes the answer by
0.0002 to 0.003 logits, inside the range of shuffled-label probe directions at every site. The gap along that vector is 5
to 48 times shorter than the class-mean difference, so this is a statement about step size and says nothing about use."

---

## 4. Analyses that would settle the open points

### From files already on disk (CPU, minutes)
1. Replace the erasure control in `17_adjacency_test.py --analyse-only` by "real coefficient x shuffled-label direction at
   the real norm" (the step columns give every control direction's slope per item); report erasure and translation as one
   arm. Done here for two sites (`a2_effects.py`); needs the two first50 sites.
2. Residual adjacency AUROC by cross-validation within the erased report half, and on the evaluated items, in place of the
   selection-to-report number (`a1_leace.py`, `a3_more.py`).
3. Report gap before / after per template, within string, and the change in mean absolute gap with a cluster bootstrap,
   next to the signed contrast (`a2_effects.py`).
4. Contrast under selection-half-only signs, report-half-only signs and a uniform sign (same script).
5. Held-out R^2 of the string-centred answer logit from the string-centred activation at every layer and all three cached
   tokens for Qwen2.5-0.5B and 1.5B: where does the answer become linearly present, and does the adjacency effect ever
   precede it? (Cache only.)
6. Label-preserving against label-changing pairs matched on the direction of the letter change, signed D, within template
   (`b2_rebuild.py`; needs `train.csv` and the tokenizer only).
7. `incremental_value/over_strongest_text_only/gain_auroc` is already in the result files; put it in the 3.6.2 table.
8. Natural SD and string-identity content of globally shuffled against within-string shuffled difference-of-means
   directions on the debiased pool, from the cache, to size F8(ii) before any rerun.
9. Qwen2.5-3B CLadder behaviour and probe numbers restricted to the 2,271 shared items, next to the other models on the
   same items (behaviour done here: 0.532 against 0.494 / 0.500 / 0.506).
10. For Qwen2.5-3B and Gemma: per-template class-mean-difference geometry and the ridge answer direction at the final
    token, from their caches, to predict what their adjacency runs will show before they return.

### Needs a GPU
1. A direct (not linearly extrapolated) control erasure with the real coefficients and random directions, and at least
   200 shuffles at the end-of-hypothesis site: 20 cannot separate 95% from 99%.
2. An eraser that works on held-out items: cross-fitted LEACE (fit on the report-half items not being evaluated) or a
   rank-k eraser chosen by held-out AUROC; then rerun the erasure. If the dependence removed does not grow when the residual
   AUROC falls from 0.70 to 0.5, the "half erased" caveat is moot.
3. The discriminating experiment for F4: at the end of the hypothesis, split each class-mean difference into its component
   along the answer-predictive (ridge) direction and the remainder, and translate by each. "Answer precursor" predicts the
   effect sits in the first component; "adjacency information that is used" predicts it survives in the second.
4. Layers 9 to 15 at the end of the hypothesis and the same edit at all hypothesis tokens at once, to see whether "a
   fifth passed on" is one token's share or the route's.
5. Qwen2.5-3B, where the gaps are ten times larger (+0.90, +1.61, +0.64, +0.21, -2.34) and the class differences share a
   direction: the only model in which the target-versus-source question is not settled in advance by linearity.
6. Label-preserving pairs with both swap directions, and both pair kinds for the other models.
7. `07c` on the debiased pool with labels shuffled within hypothesis string.
8. SAE ablation with 20 or more random sets and per-item export; block 18 with feature 4602, with collision items split by
   query type.

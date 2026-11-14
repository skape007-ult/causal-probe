# Claims pass, 14 November (raw report)

Scope: `report/report.md`, `README.md`, `RESULTS.md` as generated, read against `results/*.json`, `report/tables/*.md`, `figures/fig5, fig7, fig10, fig11, fig12`, array shapes under `cache/`, and the sixth and eighth audit reports. Templates were used to tell placeholder numbers from typed ones. No model was loaded; nothing in the repository was changed except this file. `notes/log.md`, `EXPLAINER.md` and `study/` were not used as evidence.

Abbreviations: "rank r/21" = place of a direction's |slope| among itself and its 20 matched controls (table `steering_controls.md`). Severity: wrong (contradicted by the results), overstated, stale, inconsistent, unsupported.

Counts: report 43 items, README 9, RESULTS.md 9. The ones that are contradicted by a result file, as opposed to stale or loosely scoped: A2, A16, A19, A20, A26, A35, A37, A39.

---

## A. `report/report.md`

### Section 1, summary

**A1. Item 5, "At the probed layer and final token (Qwen2.5-0.5B and 1.5B), and at the end of the hypothesis (Qwen2.5-0.5B), adding the gold probe's weight direction shifts the answer by an amount within the range ..."**
Evidence: `steering_controls.md`. Final token, gold probe, all three Qwen models: ranks 17, 6, 5, 12, 5, 17 of 21 (true, and Qwen2.5-3B is missing from the sentence). End of the hypothesis: 0.5B 9 and 12; 3B 4 and 16; 1.5B original pool 2/21, debiased pool **1/21** (-0.251, largest control 0.155; `steering_controls_qwen1.5b_c2c_debiased_hyp.json`). The sentence is true as scoped and leaves out the one arm that contradicts it.
Severity: stale. Fix: "final token (three Qwen models)"; "at the end of the hypothesis within the range in five of six arms and larger than all 20 in one (Qwen2.5-1.5B, debiased pool; 2 first ranks among the 12 gold-direction comparisons on that pool, 0.57 expected by chance)".

**A2. Item 5, "The difference-of-class-means direction moves the answer about as much as differences of means for shuffled labels on the debiased pool"** (no model named).
Evidence: debiased pool, gold difference of means: 0.5B 7/21 and 5/21, 1.5B 2/21 and 10/21, 3B 5/21 and, at the end of the hypothesis, **1/21** (+0.738, largest control 0.595).
Severity: wrong for one of six arms. Fix: add "except Qwen2.5-3B at the end of the hypothesis (first of 21; Section 4.1)". The closing parenthesis "rewritten after the second audit, wording tightened after the third" should add the eighth.

**A3. Item 6, "reaches the answer position at about 0.6 of the network's depth in three models."**
Evidence: `patching_summary.md` has four models: 0.62, 0.64, 0.62 and Qwen2.5-3B 0.72 (layer 26 of 36 on a grid of even layers, so between 24 and 26, 0.67 to 0.72; `patching_qwen3b.json`, `layers`). fig7 has four columns.
Severity: stale. Fix: "at 0.6 to 0.7 of the network's depth in four models".

**A4. Item 6, "With pairs whose correct answer does not change (Qwen2.5-0.5B so far), the answer logit moves as much, and its sign follows where the queried variable stands in the premise, not validity."**
Evidence: all three Qwen models have these pairs. "As much": template-standardised label-preserving minus label-changing |D| is -0.025 [-0.105, +0.061] and +0.013 [-0.053, +0.076] in 0.5B, but +0.970 [+0.095, +1.949] (3B, hypothesis edits) and -0.366 [-0.722, -0.033] (1.5B, premise relabellings) (`patching_by_template_<model>.json`, `standardised_over_common_templates/preserving_minus_changing`). "Sign follows position": `signed_D_by_direction_of_letter_change`, premise relabellings, label-changing later-letter / label-preserving later-letter / matched difference: 0.5B +0.252 / +0.223 / +0.029 [-0.061, +0.126]; 1.5B **-0.747 [-1.258, -0.266] / -0.022 [-0.259, +0.194] / -0.725 [-1.341, -0.137]**; 3B -0.558 [-1.786, +0.527] / +0.379 [-0.328, +1.059] / -0.937 [-2.246, +0.415]. Hypothesis edits, matched difference: 1.5B -1.080 [-1.840, -0.344]. So the sign pattern is a Qwen2.5-0.5B result; the files for the other two models contain the same analysis, it is reported nowhere, and in 1.5B the direction-matched difference excludes zero for both kinds of pair (with the known caveat that its 40 label-changing pairs are another draw).
Severity: stale ("so far"); unsupported for two of three models. Fix: "(in Qwen2.5-0.5B; in the two larger models the movement is of the same order and the sign pattern is not reproduced)".

**A5. Item 7, "(Qwen2.5-0.5B so far.)" ... "at the end of the hypothesis the same intervention is not distinguishable from a random direction of the same size. Two earlier readings ... were corrected by the fourth and fifth audits".**
Evidence: `adjacency_contrast.md` has 12 rows, three models. For 0.5B at the end of the hypothesis Section 4.4 point 6 and RESULTS.md say the contrast is above all 20 controls in signed value and 95% in absolute value ("not a clean null"), which is not "not distinguishable". The controls are shuffled-label class-mean differences, not random directions (eighth audit F11; table header already corrected). Readings were also corrected by the sixth audit (point 8, the split) and the eighth (point 12, target-sign contrast).
Severity: stale; inconsistent with 4.4. Fix: "(three Qwen models.)"; "beyond every control in Qwen2.5-0.5B and 3B, inside them in 1.5B"; "at the end of the hypothesis borderline in Qwen2.5-0.5B and inside the controls in the two larger models at the one layer tested"; "fourth, fifth, sixth and eighth audits".

### Section 2

**A6. 2.2, "Qwen2.5-3B does not fit on this machine; it and all Gemma interventions were run on a Kaggle GPU ... Models run in float16 on MPS".**
Evidence: `results/round2_environment_r2_qwen15b.json` (Linux, two Tesla T4, cuda true): Qwen2.5-1.5B's matched controls, adjacency test and label-preserving patching also come from a GPU run on a rebuilt cache.
Severity: stale. Fix: add "the second-round interventions for Qwen2.5-1.5B also ran on a Kaggle GPU".

### Section 3

**A7. 3.1, bold lead "No model answers above chance, and for different reasons."**
Evidence: summary item 1 ("Qwen2.5-3B is the only model above chance on CLadder, barely"); `answer_bias.json`: accuracy 0.526 [0.502, 0.548], AUROC 0.543 [0.518, 0.568], reading "informative".
Severity: inconsistent. Fix: "No model answers clearly above chance".

**A8. 3.1, "Renaming the variables removes it for three of the four models (Qwen2.5-0.5B ..., Gemma-2-2B ..., Qwen2.5-3B ...)".**
Evidence: same table, test set before renaming: Gemma-2-2B 0.536 [0.495, 0.566] and Qwen2.5-3B 0.534 [0.493, 0.573], both already "at chance (AUROC CI includes 0.5)". Only 0.5B goes from informative (0.589 [0.531, 0.660]) to chance (0.511).
Severity: overstated. Fix: "removes it for Qwen2.5-0.5B; Gemma-2-2B and Qwen2.5-3B show none on the test set before or after".

**A9. 3.3 (2), "at the end of the premise ... it is at chance as it must be".**
Evidence: `probe_positions.md`, Corr2Cause pool, end of premise: 0.517 [0.504, 0.532] (0.5B), 0.534 [0.517, 0.552] (1.5B), 0.499 (Gemma). Two intervals exclude 0.5.
Severity: overstated (minor). Fix: "it is near chance (0.50 to 0.53)".

**A10. 3.6.3, "The end-of-premise check is not available for these two models (final-token activations only)."**
Evidence: `cache/qwen3b/c2c_debiased/resid.npy` is (2400, 37, 2, 2048), `meta.json` positions `['final', 'content_end']` since the 14 November ingest. The end-of-premise position is still absent, so the first half holds.
Severity: stale (parenthesis only). Fix: "(no end-of-premise activations)".

### Section 4, opening

**A11. Status box, "every number was recomputed from the raw outputs and agrees; four table labels and two readings were wrong or misleading and are corrected here."**
Evidence: the eighth audit lists 15 findings, two critical and five major, among them non-converged probe fits in seven stages (F7) and a contrast whose signs are read from the evaluated items (F5); neither is a label or a reading. Both are stated further down.
Severity: overstated by omission (minor). Fix: add "; it also found non-converged probe fits at seven stages and that the contrast's signs use the evaluated items (both below)".

**A12. Second paragraph, "Results for Gemma-2-2B and Qwen2.5-3B come from the Kaggle run and are added when it returns".**
Evidence: both are in the first steering table (`steering.md`, 72 rows for the two models) and discussed in 4.1.
Severity: stale. Fix: "came from the first Kaggle run and are in the first table".

### Section 4.1

**A13. Design, "At one layer (the gold probe's selected layer)".**
Evidence: Qwen2.5-3B, original pool: steered at layer 25 (C 0.1; `steering_controls_qwen3b_c2c_probe.json`, `steer_select_qwen3b_c2c_probe.json`), while the table of 3.3 gives the selected probe as layer 27, C 1 (eighth audit F13).
Severity: inconsistent (minor). Fix: add "(Qwen2.5-3B, original pool: layer 25 from an earlier selection; Section 3.3 lists 27)".

**A14. Point 1, heading "The gold-probe weight direction shifts the answer by an amount within the range of label-free directions of the same kind."**
Evidence: the four numbers are final-token rows of 0.5B and 1.5B and are correct. The table under which it stands has 12 gold-probe rows; one is first of 21 (A1).
Severity: overstated as a heading. Fix: begin "At the final token, ...".

**A15. Point 2, "The shuffled-label directions have the same natural spread as the gold direction on the debiased pool ... Per natural standard deviation, the gold direction for Qwen2.5-1.5B on the original pool exceeds all 20 shuffled controls (5.51 times their median); in the other three cases it is at the 50%, 75% and 50% percentile."** (Check 2b.)
Evidence: numbers match the table. But the table's own note ‡ and the "Three cautions" paragraph say this column is not a matched comparison for probe directions (eighth audit F2): the gold probe's natural SD there is 0.0158 against a control median of 0.0052, so the "100" is produced by the spread. "Same natural spread on the debiased pool" holds for 0.5B (0.0031 / 0.0031), less so for 1.5B (0.0035 / 0.0025, 1.4 times).
Severity: inconsistent with the table note; overstated. Fix: append "; this is not a matched comparison (‡): the gold probe's natural SD is three times that of its controls there, whatever its causal role", or delete the sentence.

**A16. Point 5, "The model-answer probe direction exceeds all 100 isotropic and all 20 shuffled-label probe directions in every case."**
Evidence: Qwen2.5-0.5B, original pool, end of the hypothesis: 95 and 95, rank 2/21 (`steering_controls_qwen0.5b_c2c_probe_hyp.json`, `real/model_out/versus/*/percentile_of_abs_slope_per_s` = 0.95). The other 11 arms are 100/100.
Severity: wrong as a universal. Fix: "in 11 of the 12 arms (Qwen2.5-0.5B, original pool, end of the hypothesis: second of 21)".

**A17. "May not" bullet, "With twenty controls and one site".**
Evidence: two tokens in three models (same bullet, last sentence).
Severity: stale (minor). Fix: "one layer per token".

**A18. "Gemma-2-2B and Qwen2.5-3B (first Kaggle run)": "They can therefore not be placed 'inside or outside the range of label-free directions', and I draw no conclusion from them yet ... The matched controls for these two models ... are added to the controls table when they arrive."**
Evidence: Qwen2.5-3B has 12 rows in the controls table.
Severity: stale for Qwen2.5-3B. Fix: restrict both sentences to Gemma-2-2B; "Qwen2.5-3B's matched controls are in the controls table".

**A19. "With the matched controls ...": "On the original pool, per natural standard deviation, the gold-probe direction exceeds all 20 controls in two rows of the table (Qwen2.5-1.5B at the final token; Qwen2.5-0.5B at the end of the hypothesis)".** (Check 2b.)
Evidence: `steering_controls_multiplicity.json`, `by_arm/*/abs_rank_per_natural_sd` = 1 for the gold probe in five original-pool rows (0.5B end of hypothesis; 1.5B final and end of hypothesis; 3B final and end of hypothesis) and one debiased-pool row (1.5B end of hypothesis). "Two rows" was the sixth audit's count for a six-row table. And the sentence reads the unmatched column as evidence (F2).
Severity: wrong count; overstated. Fix: "in five of the six original-pool rows (‡: not a matched comparison)".

**A20. Same paragraph, last sentence: "For Qwen2.5-1.5B the debiased-pool rows in the table still use globally shuffled controls until its second-round run returns; the table's last column says which."**
Evidence: `steering_controls_qwen1.5b_c2c_debiased.json` and `..._debiased_hyp.json`, `control_labels_shuffled` = "within hypothesis string"; the table's last column says the same.
Severity: wrong. Fix: delete the sentence.

**A21. "How much two first ranks mean": "What does repeat is the sign: the gold difference of means has a positive slope in 12/12 arms ... the gold-probe direction has a negative slope in 10/12."**
Evidence: correct for the 12 Qwen arms (`steering_controls_multiplicity.json`). The report's first table has the only other-family values: Gemma-2-2B, debiased pool, gold difference of means **-1.766 [-1.814, -1.718]**, gold probe +0.468 and +0.247 (`steering.md`): both signs reversed.
Severity: overstated by scope. Fix: "in the three Qwen models" (and, if wanted, "Gemma-2-2B's first-run slopes have the opposite signs").

**A22. Controls-table note ‡, "a probe fitted to a decodable label has 1.0 to 5.5 times the natural SD of probes fitted to shuffled labels".**
Evidence: true for the 12 gold-probe rows (1.0 to 5.5). The ‡ also marks the 12 model-answer rows, where the ratio is 5.4 to 110 (for example 0.2754 against 0.0025; recomputed from `real/*/own_sd_over_s` and `families/shuffled_probe/own_sd_over_s_median`).
Severity: overstated (minor). Fix: "a gold-label probe has 1.0 to 5.5 times (a model-answer probe 5 to 110 times)".

**A23. Two values for one quantity.** Gold-probe slope, Qwen2.5-1.5B, debiased pool, final token, layer 22: -0.211 [-0.224, -0.198] in the first table (laptop, 300 items) and -0.142 [-0.161, -0.126] in the controls table and in point 1 (GPU, 150 items); point 3 takes its natural-scale figures (280.42) from the laptop file. The "Three cautions" paragraph explains a laptop/GPU difference but does not say that both values are printed.
Severity: inconsistent (disclosed in part). Fix: in that paragraph, "(-0.211 in the first table, -0.142 in the controls table)".

### Section 4.2

**A24. "What the data show": Qwen2.5-3B is absent.**
Evidence: `patching_summary.md` and fig7 include it: mean D +0.673 [-0.022, +1.376] (hyp), +0.002 [-0.593, +0.580] (prem), D > 0 in 0.60 and 0.52 of pairs, first layer at 0.5 recovery 26 of 36 (0.72, even layers). "The models do not prefer the valid member of a pair" still holds, narrowly, for its hypothesis pairs.
Severity: stale (incomplete). Fix: one sentence with those numbers after the Gemma sentence.

**A25. "The answer logit moves as much when the correct answer stays the same as when it changes: mean |D| is 0.233 against 0.253 ..."** (Check 2d.)
Evidence: these are Qwen2.5-0.5B's numbers; the sentence stands directly under a three-model table in which 1.5B premise pairs are 0.686 against 1.055 and 3B hypothesis pairs 2.643 against 1.884. The scope comes only from a heading three blocks above.
Severity: overstated by scope. Fix: begin "In Qwen2.5-0.5B the answer logit moves as much ...".

**A26. Same paragraph, "Those intervals are wide: they could not exclude a label effect of 0.2 logits, about the size of |D| itself."**
Evidence: the intervals are [-0.105, +0.061] and [-0.053, +0.076]; both exclude ±0.2. What they cannot exclude is about 0.1 (hypothesis edits) and 0.08 (premise relabellings).
Severity: wrong. Fix: "they could not exclude a label effect of 0.1 logits, about 40% of |D|".

**A27. Sign paragraph, "So the Yes−No logit is about 0.2 higher when the variable named in the hypothesis stands earlier in the premise, and about 0.1 higher when the hypothesis names an earlier letter, whether or not that makes the hypothesis valid."**
Evidence: as A4. Qwen2.5-0.5B only; the paragraph headed "Qwen2.5-1.5B and Qwen2.5-3B" is silent on the sign, and the result files do not repeat it.
Severity: overstated by scope; unreported contrary rows. Fix: "So in Qwen2.5-0.5B ..." and one sentence: "In Qwen2.5-1.5B the label-changing premise pairs have the opposite sign (-0.747 [-1.258, -0.266]) and the label-preserving ones none (-0.022 [-0.259, +0.194]); in Qwen2.5-3B neither interval excludes zero."
Nit, unchanged since the sixth audit: +0.223 [+0.173, +0.268] in this paragraph, [+0.173, +0.270] in the table.

### Section 4.4

**A28. Design, "with the signs taken from the selection half only, no conclusion changes: table note".** (Check 2a.)
Evidence: the table note and the eighth audit (F5) give this for Qwen2.5-1.5B and 3B only. For Qwen2.5-0.5B, whose +0.013 is the headline number of claim 3, the F5 variant is in no file. Nearest stored value (`adjacency_qwen0.5b_c2c_debiased_hyp_best.json`, `audit5/translation_contrast_under_other_sign_choices/selection_half_point_estimate_signs_all_five`): +0.011, above all 20 in signed value, 90% in absolute value (final token: +0.097, 100%). Selection-half gaps have the same signs as all-item gaps for all five types (`audit5/observational_gap_by_half`).
Severity: unsupported for one of three models. Fix: "(checked for the two larger models; for Qwen2.5-0.5B the selection-half signs of all five types give +0.011, 90%)".

**A29. Remaining uses of "pre-specified".** (Check 2a.)
Places: point 3 "The pre-specified contrast (+0.110 against a benchmark of 0.112)"; point 6 "The pre-specified contrast is +0.013"; Status bullet "the design, the contrast and the expectations were written down before the full-size runs". (Also RESULTS.md and README claim 3; list C.) The Design paragraph and the contrast-table note already carry the corrected wording.
Severity: inconsistent with F5 and with the report's own Design paragraph. Fix: "The contrast" in points 3 and 6; Status: "the design, the form of the contrast (not its templates and signs) and the expectations".

**A30. "20 random directions of the same size"** (point 3, twice in point 8, summary item 7).
Evidence: they are class-mean differences for adjacency labels shuffled within hypothesis string, rescaled (table headers, corrected after F11; fig11 and fig12 titles say it correctly).
Severity: inconsistent (minor). Fix: "20 shuffled-label class-mean differences of the same norm".

**A31. Point 5 and first "May" bullet, "Final token, layer 11: nothing ... The adjacency-dependent part of the answer reaches the final token between layers 11 and 15" / "Four layers earlier it cannot. That locates the arrival".**
Evidence: contrast-table note: "Where the positive control does not exceed its controls (the earlier-layer sites), a null result for the other directions says nothing." At Qwen2.5-0.5B layer 11 the positive control is +0.255, 95% (19 of 20), held-out R² 0.43.
Severity: inconsistent. Fix: either exempt this site in the note ("the earlier-layer sites other than Qwen2.5-0.5B layer 11, where it exceeds 19 of 20") or add that qualification to the bullet.

**A32. Second "May" bullet, "per hypothesis type 3 to 4 of 20 are larger, does not depend on the item's hypothesis type, and lowers ...".**
Evidence: `adjacency_effects.md`, Qwen2.5-0.5B, end of hypothesis: controls smaller in absolute value 45, 5, 20, 85, 80%. "3 to 4 of 20 larger" is true for common effect and common cause only; for the other three types 11, 19 and 16 of 20 are larger. The clause is also grammatically broken.
Severity: overstated. Fix: "for the two types with the largest effects 3 and 4 of 20 are larger; the effect does not depend on ...".

**A33. Point 10, heading "The dependence to be explained is much larger than in Qwen2.5-0.5B."**
Evidence: in logits yes (benchmark 0.353 and 1.375 against 0.112). Relative to the SD of the answer logit (0.39, 2.03, 7.83; `adjacency_sites.md`) it is smaller: 0.29 for 0.5B, 0.17 and 0.18 for the larger models. The same point says "a tenth to a third of that spread"; for 0.5B the range is 0.11 to 0.59.
Severity: overstated. Fix: "much larger in logits (not relative to the spread of the answer)".

**A34. Point 11 heading "the same localisation as in Qwen2.5-0.5B" and "All three Qwen models, may: ... beyond all 20 controls in Qwen2.5-0.5B and Qwen2.5-3B and inside them in Qwen2.5-1.5B".** (Check 2e.)
Evidence: true for the contrast (100%, 85%, 100%). The 0.5B statement ("beyond every control") rests on one hypothesis type, common effect. By the same per-type standard Qwen2.5-1.5B's direct-causation type is beyond all 20 for translation (+0.470 against -0.253 to +0.413) and for erasure (change -0.267 against -0.238 to +0.139); `adjacency_effects.md`; visible in fig11. For Qwen2.5-3B the type is common cause. For 0.5B the reduction of the mean absolute gap is at 75%, not beyond all.
Severity: inconsistent standard. Fix: "inside them in Qwen2.5-1.5B (contrast; its direct-causation type alone is beyond all 20)".

**A35. Point 12, "In the transfer matrices of both models at this site every column has one sign".**
Evidence: `adjacency_transfer.md`: true for Qwen2.5-1.5B. Qwen2.5-3B: descendant column -0.008 and collider column -0.004 on common-cause items, all other cells of those columns positive.
Severity: wrong to the letter. Fix: "every column has one sign (two cells of Qwen2.5-3B within 0.01 of zero apart)".

**A36. Point 13 heading "no effect at the layer tested" and "All three Qwen models, may: ... At the end of the hypothesis, at the one layer tested per model, nothing exceeds its controls in the two larger models."** (Check 2e.)
Evidence: true for translation and erasure. Not for the third arm in Qwen2.5-1.5B, layer 18 (`adjacency_qwen1.5b_c2c_debiased_hyp_best.json`): `steps/probe/per_unit_norm_from_large_step` -0.028, controls -0.022 to +0.031, `outside_control_range` true (95% smaller in absolute value); `probe_direction_across_class_gap/contrast` +0.0056, controls -0.0048 to +0.0050 (100%); by type outside the range for four of five. The fit at this site converged (eighth audit F7). The effect is 0.006 to 0.013 logits and the per-natural-scale comparison has the F2 problem, but per unit of norm it has the same borderline status as the 0.5B contrast that claim 3 quotes. No sentence mentions it.
Severity: overstated. Fix: "neither the translation nor the erasure exceeds its controls in the two larger models".

**A37. "Limits" bullet, "the probe-step arm at seven sites of the two GPU runs rests on probes that had not converged".**
Evidence: eighth audit F7 and Q2 (iv): seven stages in all, of which four are adjacency sites (3B final earlier layer; 1.5B final best, final earlier, end-of-hypothesis earlier) and three are steering-control stages. The 4.1 wording ("seven sites of the two GPU runs") is right; this one is not.
Severity: wrong count. Fix: "at four of the eight sites".

### Section 6

**A38. The section describes five audits; "Raw reports" names three files; "What the audits did not cover": "The corrected analysis of Section 4.4 and the reruns that followed the fifth audit (within-string steering controls, the split by answer direction) have not been audited."**
Evidence: eight raw reports exist in `notes/`; the report body cites the sixth and eighth. `notes/audit_consolidation_raw.md` section 1 recomputed "every headline number of Section 4.4, four sites", the within-string controls and the split; `notes/audit_round2_runs_raw.md` covers the new rows.
Severity: stale; the quoted sentence is now wrong. Fix: one line each for audits 6 to 8; replace the sentence by what is unaudited now (the text added on 14 November apart from this pass; the probe rows at non-converged sites; Gemma-2-2B's pending rows).

### Section 7 (check 2f)

**A39. "Steering. ... final token, and for Qwen2.5-0.5B also the last token of the hypothesis ... Debiased-pool controls are shuffled within hypothesis string for Qwen2.5-0.5B and still over the whole pool for Qwen2.5-1.5B".**
Evidence: A20; end-of-hypothesis rows exist for all three Qwen models.
Severity: wrong. Fix: "final token and last token of the hypothesis (three Qwen models) ... Debiased-pool controls are shuffled within hypothesis string; original-pool controls over the whole pool (not interpretable for validity)".

**A40. "Adjacency test. One model so far".**
Severity: stale. Fix: "Three models of one family; one informative layer per token; templates and signs of the contrast read from all items".

**A41. "Patching. ... every second layer for the 1.5B model."**
Evidence: `layers` in `patching_gemma2b.json`, `patching_qwen3b.json` and all three `*_same.json` files are every second layer; only the 0.5B label-changing run has every layer.
Severity: stale (incomplete). Fix: "every second layer except the first Qwen2.5-0.5B run". ("The models do not distinguish the pairs by gold label" holds: all eight mean-D intervals include zero.)

**A42. "Remote caches hold the final token only. For the larger model ... the position comparison is unavailable."**
Evidence: A10. Qwen2.5-3B's debiased-pool cache holds the end of the hypothesis; `probe_positions.md` has no Qwen2.5-3B row.
Severity: stale. Fix: "The first remote caches hold the final token only; for the larger model the position comparison was not computed (no end-of-premise activations)".

**A43. Missing.** (i) "Two machines" names Gemma only; Qwen2.5-1.5B now mixes laptop results (probes, first steering table, 40 label-changing patching pairs) and GPU results (controls, adjacency, 60 label-preserving pairs). (ii) Non-converged probe fits at seven GPU stages (in 4.1 and 4.4, not here; "Numerics" mentions only the 300-iteration limit with "under one point on the one fold checked"). (iii) Multiplicity of the 24 steering comparisons (in 4.1, not under "Multiplicity"). (iv) The eraser bullet and the "20 shuffles" bullet are fine.
Severity: incomplete. Fix: one clause each.

### Section 8 (check 2h)

Empty apart from "*(Deliberately left empty: to be written by the project owner.)*" in both `report.md` (line 1463) and the template. Correct.

### Appendices

A (logit lens): the typed ranges were recomputed from `lens_*.json` and are correct (pool maxima 0.594, 0.598, 0.598 and 0.632; CLadder 0.471 to 0.526 and 0.544). B: correct (0.862 to 0.870; depths 0.63 to 0.86). C: one typed range, "relative squared error 0.16 to 0.26", covers the end-of-context rows only; the table also has 0.15 and 0.27 at the final token (minor). D: "of each model and pair kind" now includes Qwen2.5-3B (r 0.99 in the table); the sentence about random components was not rechecked for it.

---

## B. `README.md` (lines outside the included results page)

**B1. Status line, "GPU runs for three models are still returning".**
Evidence: `results/round2_status_r2_qwen15b.json` and `..._qwen3b.json`: `all_ok` and `finished` true. Outstanding by the report's own status box: Gemma-2-2B's rows, and the 7B behaviour check.
Severity: stale. Fix: "for two models (Gemma-2-2B, Qwen2.5-7B)".

**B2. "Further results", Patching: "about 60% of the network's depth (three models). The answer logit moves as much ... and its sign follows where the queried variable stands in the premise, not validity (Qwen2.5-0.5B so far)".**
Evidence: A3, A4.
Severity: stale; unsupported for two of three models. Fix: "60 to 70% (four models)"; "(Qwen2.5-0.5B; in the two larger Qwen models the movement is of the same order and the sign pattern is not reproduced)".

**B3. Table caption, "('Corr2Cause debiased pool' rows appear once those runs finish)".**
Evidence: the table below it has the debiased-pool row for all four models.
Severity: stale. Fix: delete the parenthesis.

**B4. "Added after seeing results ...: ... the adjacency test, its second version and its transfer arm ... Each can only make the claim harder to pass."**
Evidence: report 4.4, Status bullet: "The move to LEACE with its first control made a positive reading easier, not harder".
Severity: inconsistent. Fix: "Each of D18 to D25 can only make the claim harder to pass; the adjacency test is exploratory".

**B5. "Does not show": "The largest model analysed here has 3 billion parameters and answers at chance."**
Evidence: A7 (CLadder 0.526 [0.502, 0.548]).
Severity: inconsistent (minor). Fix: "at or barely above chance".

**B6. Documents table, "`notes/audit_*_raw.md` | ... (Tier 1, Tier 2, debiased pool, round-2 code)" and "`study/` | Reading order, 40 graded questions with answers, five predict-then-check exercises".**
Evidence: eight audit files in `notes/`; `study/02_questions.md` runs to question 52, `study/03_exercises.md` has six exercises (file counts only).
Severity: stale. Fix: "(eight audits)"; "52 graded questions ..., six ... exercises".

**B7. "How to read each figure: `EXPLAINER.md`".** (Check 2g.)
The README has no caption or how-to-read line for figures 10 to 12; neither has the report (image links only). The entry it points to says "fig11 is at the answer position at layer 15 of 24", which describes one of the three columns now drawn (15/24, 19/28, 25/36). The text printed inside the figures is correct: fig10 legends 0.25/0.23, 0.24/0.24, 1.11/0.99, 1.06/0.69, 1.88/2.64, 1.88/1.95 match `patching_label_preserving.md`; fig11 and fig12 panel lines match `adjacency_contrast.md` and `adjacency_sites.md` in all six panels, and their titles describe the controls correctly.
Severity: stale, in the file the README sends the reader to. Fix (EXPLAINER): "fig11 is at the answer position at each model's best layer".

**B8. Reproduction, "(`scripts/build_cache.sh` for the two Qwen models; `KAGGLE.md` for Gemma and the larger model) ... *(run_all.sh: pending.)*".**
Evidence: `scripts/run_all.sh` exists (61 lines). Qwen2.5-1.5B's round-2 results need the GPU route as well (A6).
Severity: possibly stale; I could not tell whether "pending" means unwritten or unverified.

**B9. The results page included at the top** (lines 8 to 55) repeats every item of list C.

---

## C. RESULTS.md (not to be edited tonight; for the owner)

**C1. "Run: ... six independent audits."**
Evidence: eight raw reports; the report cites the eighth.
Severity: stale. Fix: "eight".

**C2. Claim 1, "Figure (three of the eight cells): `fig5_confound_c2c_probe.png`."**
Evidence: the figure was regenerated in commit 8d4cb5b and shows four models on the Corr2Cause pool.
Severity: stale. Fix: "four of the eight cells".

**C3. Claim 3, "gives a pre-specified contrast of +0.013".** (Check 2a.)
Evidence: eighth audit F5; report 4.4 Design paragraph and table note. For this model the selection-half variant is not in any file (A28).
Severity: overstated. Fix: drop "pre-specified".

**C4. Claim 3, "in Qwen2.5-0.5B ..." and "One model so far."** (Check 2e.)
Evidence: `adjacency_contrast.md`. End of the hypothesis, best layer: 0.5B +0.013 of 0.112 (above all 20 signed, 95% absolute); 1.5B +0.020 of 0.353 (45%); 3B +0.008 of 1.375 (5%). Final token: +0.110 (100%), +0.225 (85%), +0.775 (100%). The number quoted is still correct for 0.5B; it is now the largest of three relative results at that token, and the two new ones are plain nulls at one layer each.
Severity: stale. Fix: "Three Qwen models; at the end of the hypothesis the two larger ones show nothing beyond controls at the one layer tested."

**C5. Claim 3 caveats, "Steering along the gold-label probe's weight vector stays within the range of matched controls per unit of norm."** (Check 2c.)
Evidence: true for Qwen2.5-0.5B (ranks 17, 6, 9, 12 of 21) and at the final token in all three models. False if carried over to three models unchanged: Qwen2.5-1.5B, debiased pool, end of the hypothesis, rank 1/21 (-0.251; sign lowers Yes when pushed towards valid). The gold difference of means is also first of 21 once (Qwen2.5-3B, same pool and token, +0.738). 2 first ranks in 12 debiased-pool comparisons, 0.57 expected, probability of two or more 0.109; none survives correction (`steering_controls_multiplicity.json`). No echo of the per-natural-SD reading here (check 2b: none in RESULTS.md or README).
Severity: true as scoped to one model; wrong if the scope is widened without the exception.

**C6. Claim 3 caveats, "at the answer position the same shift removes the dependence for one hypothesis type and creates one for another; ... (held-out R² 0.96)".**
Correct for 0.5B. For the record with three models: R² 0.96, 0.98, 0.96; the transfer matrices are 100%, 99% and 92% rank one (effect follows the vector, not the item).

**C7. Claim 3 figure, `fig12_adjacency_end_of_hypothesis.png`.**
The figure now has three columns; the claim text describes the first. Inconsistent until C4 is settled.

**C8. Limitations.** (Check 2f.)
- "Claim 3: one model, four sites, 20 controls": stale (three models, four sites each).
- "for Gemma-2-2B and Qwen2.5-3B only final-token activations exist": stale for Qwen2.5-3B on the debiased pool (A10); its steering directions at the end of the hypothesis were fitted on both pools in the GPU run.
- "(a 7B behaviour check is running)": no result file for it exists; unchanged as far as I can see.
- Missing, if claim 3 is widened: the contrast's signs use the evaluated items; probes not converged at seven GPU stages; 24 steering comparisons without correction.

**C9. Retraction table.**
Not contradicted by anything new. The two readings corrected by the eighth audit (the positive target-sign contrast; the per-natural-SD percentile) were not in the prose of `report.md` at the ingest commit (8d4cb5b has no "Qwen2.5-1.5B and Qwen2.5-3B" paragraphs; the tables of that commit did carry the unmarked per-natural-SD column and the target-sign contrast), so they are not retractions of sentences the report had made. "Checked by audit 6" for claims 1 and 2 matches section 1 of that audit.

The headline of claim 3 ("No intervention shows that what a probe reads is what the answer is computed from") is not contradicted by the three-model tables: the two effects beyond controls at the answer position act as answer shifts, and the two steering first ranks are undecided. Whether "no intervention shows" should acknowledge those two first ranks is the owner's call.

---

## Checked and found correct

- Section 1 items 1 to 4, 8, 9 and the pre-registration paragraph against the tables (typed ranges 0.56 to 0.59, 0.49 to 0.51, "about two points", AUROC 0.003).
- Section 3.1 to 3.6: every typed number I compared with a table (three-model ranges, selectivity, position differences, the 3.6.6 bullets, the adjacency-dependence table in 4.4 against `probe_content_*`). 3.6 was not re-derived from raw data; the fifth and sixth audits did that.
- 4.1: all percentile, ratio and slope values in points 1, 3, 4; the 0.5B paragraph's numbers; the whole "Qwen2.5-1.5B and Qwen2.5-3B at both tokens" block (slopes, ranks, z values, 12 and 24 comparisons, 0.57 and 1.14 expected, probability 0.109 recomputed by hand and 0.104 read from the file, signed ranks at most 5 in six debiased arms, "5 of 21 at the final token", "10 of 21 and 5 of 21"), the direction-geometry and size paragraphs against `direction_geometry_qwen3b_c2c_debiased_hyp.json` and the eighth audit, the first-run slopes for Gemma-2-2B and Qwen2.5-3B. 150 items and layers in every controls file.
- Check 2c, final token: the gold probe is within its controls in all six final-token arms; the two first ranks on the debiased pool are exactly the two the brief names, and the report's new paragraphs state both.
- 4.2: table rows, the within-type differences for three models, "two of six intervals exclude zero, with opposite signs", "eight pairs of a type the label-changing set lacks", same first layer for both kinds of pair at stride two, 16 of 60 and 6 of 40 donors.
- 4.4: points 1 to 4, 6 to 8 (numbers), 9 (inside the control range at all four 0.5B sites), 10 (gaps and SDs), 11, 13 and 14 (numbers), the rank-one shares and discordant-cell contrasts, "about half of the excess decodability goes" for all three models, the Limits bullet apart from A37.
- Figures 10, 11, 12 against their tables; fig5 and fig7 for the number of models shown.
- RESULTS.md claims 1 and 2 and the claim 3 numbers for Qwen2.5-0.5B.

## Could not check

- Anything that needs a model run, and the raw arrays behind the JSONs (not recomputed; the eighth audit did so for the new rows).
- The count "four table labels and two readings" in the Section 4 status box.
- "about 210 to 230 per kind" for the prepared patching run, and that `07d_steering_settle.py`'s primary test was fixed in advance (the only source is the log).
- The sixth audit's "four fifths" and "a tenth" in 4.4 point 1 (taken from its report).
- Appendix C beyond two table comparisons; Appendix D's statement about random components for Qwen2.5-3B.
- Whether the 7B run or Gemma-2-2B's second round has produced anything outside `results/`.
- `report/benchmark_text_baselines.md` (not in scope).

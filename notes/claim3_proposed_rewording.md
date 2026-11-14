# Claim 3 of RESULTS.md after the Qwen 1.5B and 3B runs: proposed rewording, for your decision

Written Sat 14 Nov, about 01:15, after the eighth audit (`notes/audit_round2_runs_raw.md`). **`RESULTS.md` has not been changed.** Every number below is in `results/*.json`; the file and key are given so that you can check them. Nothing here has been audited as wording: the audit checked the numbers, not this note.

## What RESULTS.md says now

> **3. No intervention shows that what a probe reads is what the answer is computed from.**
> *Number:* in Qwen2.5-0.5B, shifting items at the end of the hypothesis along the encoding of adjacency gives a pre-specified contrast of +0.013 out of a possible 0.112: above 100% of 20 label-shuffled directions of the same size, larger in absolute value than 95%. […]
> *Caveats:* […] Steering along the gold-label probe's weight vector stays within the range of matched controls per unit of norm. One model so far.

Three things in it are now out of date or too strong, whatever you decide about the headline:

1. "One model so far": there are three Qwen models.
2. "pre-specified contrast": the audit found (F5) that the templates and signs of the contrast are read from the observed gaps of all items, including the evaluated ones. With signs from the selection half alone no conclusion changes, but "pre-specified" should go.
3. "Steering along the gold-label probe's weight vector stays within the range of matched controls": true for five of six probe comparisons on the debiased pool. In Qwen2.5-1.5B at the end of the hypothesis it is larger in absolute value than all 20 controls, with a negative sign.

## The new evidence, sorted

### For keeping the headline

- **Adjacency test at the end of the hypothesis, the two new models: nothing.** Contrast +0.020 of 0.353 for Qwen 1.5B (45% of 20 controls smaller in absolute value) and +0.008 of 1.375 for Qwen 3B (5%). The erasure changes the mean absolute gap by +0.009 (30% of fair controls change it less) and −0.011 (5%). `results/adjacency_<model>_c2c_debiased_hyp_best.json`, keys `translation/contrast`, `audit5/gaps_on_evaluated_items_within_string/mean_absolute_gap_all_five_templates`. The audit confirmed this is not produced by sign conventions, template choice or a weaker eraser.
- **At the answer position the effect is an answer shift, in all three models.** The contrast is beyond all 20 controls for Qwen 0.5B (+0.110 of 0.112) and Qwen 3B (+0.775 of 1.375; controls −0.597 to +0.422), inside for Qwen 1.5B (+0.225 of 0.353; 85%). But the answer is a linear function of the activation there (held-out R² 0.96, 0.98, 0.96), and a class-mean difference pushes the answer the same way for items of every hypothesis type: on the cells of the transfer matrix where "follows the item's type" and "follows the vector's type" predict different signs, the contrast has the vector's sign in every model (−0.115 / +0.115, −0.313 / +0.313, −0.435 / +0.435 for target / source), and the matrix is rank one to 100%, 99% and 92%. Table `report/tables/adjacency_transfer_summary.md`. That is the pattern of "this is where the answer sits", not of "a variable that is read".
- **Gold-probe direction, final token, debiased pool: inside its controls in all three** (75%, 45%, 20% of controls smaller). `results/steering_controls_<model>_c2c_debiased.json`.
- **Multiplicity.** On the debiased pool 12 comparisons of a gold direction with 20 matched controls were made; 0.57 first ranks are expected by chance and 2 are observed; the probability of 2 or more is 0.109. `results/steering_controls_multiplicity.json`.

### Against keeping it as it stands

- **Qwen 3B, end of the hypothesis, layer 25: the gold difference of class means has slope +0.738 [+0.641, +0.832] per unit of s, larger than all 20 controls shuffled within hypothesis string** (largest 0.595; z = +2.06). Every one of the 150 items, all 13 blocks and all 48 hypothesis strings agree in sign; it is rank 1 in every block. `results/steering_controls_qwen3b_c2c_debiased_hyp.json`. This is an intervention on a gold-label direction, before the answer position, that exceeds its matched controls: on its face the thing claim 3 says does not exist.
- **Qwen 1.5B, end of the hypothesis, layer 19: the gold-probe direction has slope −0.251 [−0.272, −0.231], larger in absolute value than all 20 controls** (largest 0.155; z = −2.75, the largest standardised excess in the table). The sign is the wrong one for "towards valid raises Yes".
- **The sign of the gold difference of means repeats.** Positive in all 12 arms; among the 5 most positive of 21 in all six arms on the debiased pool. Noticed after the fact.
- **The nulls at the end of the hypothesis are narrow** (audit F6): one layer per model, where the answer is already linearly readable (R² 0.79, 0.91, 0.89) and where even the model-answer direction moves the answer 2.4 to 8 times less per unit of norm than at the final token. No layer between the early one and that one was tested.

### What weakens the two results that go against it

- Each is one arm with p = 1/21 at best; neither is first in any neighbouring arm (Qwen 3B: the same direction is 5th of 21 at the final token; at the same token the gold difference of means is 10th in Qwen 1.5B and 5th in Qwen 0.5B). Neither survives a correction for 12 comparisons.
- The Qwen 3B direction is loaded on adjacency, not on validity as such: cosine +0.688 with the adjacent-minus-not difference of direct-causation items (controls: SD 0.224, largest 0.549); 55% of its squared norm lies in the span of the five adjacency differences (controls: 18% on average, **but up to 52%**, so this is suggestive, not decisive). Within a hypothesis string the gold label goes with adjacency, and the within-string shuffle destroys that together with validity. So even if the effect is real, the comparison does not say that validity is used. `results/direction_geometry_qwen3b_c2c_debiased_hyp.json`.
- It is not an answer direction (cosine +0.066 with the model-answer probe direction), and on the half it was fitted on the gold label does not predict the model's answer within string (AUROC 0.510). So it is also not explained away as "any direction aligned with the answer".
- At natural scale it is tiny: moving an item across the gap between valid and invalid along it is worth +0.023 logits; the unsteered answer on those items has mean −15.09 and SD 7.56. A push of a whole unit of s (11.5 natural SDs) flips 1.3% of answers one way and 3.3% the other.
- Pushing in the adding direction alone, it ranks 3rd of 21.

**The auditor's bottom line (its words):** "Undecided between a chance top rank and a modest real effect of an adjacency-loaded direction; it is not a property of answer-correlated directions at this site, and it is not evidence about validity as such."

## Options

**A. Keep the headline, replace number and caveats (my recommendation if you decide before the settle run).** The headline says "no intervention *shows*"; an undecided arm at p = 0.05 among 12 does not show it. But the claim has to carry the two arms in its caveats, in the open.

> **3. No intervention shows that what a probe reads is what the answer is computed from.**
> - *Number:* at the last token of the hypothesis, shifting items along the encoding of adjacency (a structural property the answers depend on) moves the answer by +0.013 of a possible 0.112 in Qwen2.5-0.5B, +0.020 of 0.353 in Qwen2.5-1.5B and +0.008 of 1.375 in Qwen2.5-3B (contrast with the signs of the observed dependence; 95%, 45% and 5% of 20 label-shuffled controls are smaller in absolute value). One layer per model.
> - *Caveats:* at the answer position the same shift does move the answer, beyond all 20 controls in two of the three models; the answer is a linear function of the activation there (held-out R² 0.96 to 0.98) and the shift acts alike on items of every hypothesis type, so that locates the answer. Among 12 comparisons of a gold-label direction with matched controls on the debiased pool, two rank first of 21 (0.57 expected by chance): the gold class-mean difference in Qwen2.5-3B at the end of the hypothesis (+0.74 logits per unit of s; a direction loaded on adjacency; +0.02 logits at natural scale) and the gold probe direction in Qwen2.5-1.5B (−0.25, the opposite of "towards valid raises Yes"). Both are undecided. Three models of one family.
> - *Figure:* `fig12_adjacency_end_of_hypothesis.png`. *Checked by* audits 2, 4, 5, 6 and 8.

**B. Weaken the headline.** "No intervention shows a property that is read according to the question; two of twelve steering comparisons stand out and are undecided." More cautious, less quotable, and it puts an undecided result into a headline.

**C. Decide after the settle run (what I would do with the time there is).** `notebooks/kaggle_r3_settle_qwen3b.ipynb` (about 1.5 GPU hours) refits the Qwen 3B direction on the other half of the blocks and tests it once, one-sided, against 100 controls, with the rule written down beforehand (`notes/log.md`, "Pre-registration of the settle run"): p ≤ 0.05 replicates, p > 0.20 does not. It also runs controls that keep the label's association with adjacency. Then:
- *does not replicate:* option A, with "did not replicate on an independent draw of the direction (p = …)" in place of "undecided";
- *replicates, but not beyond the adjacency-keeping controls:* option A with the caveat rewritten as "a small effect of an adjacency-loaded direction at the end of the hypothesis in Qwen2.5-3B, replicated; it is an effect of adjacency, as at the answer position";
- *replicates beyond the adjacency-keeping controls:* the headline cannot stand for Qwen 3B. Then: "In the smallest two models no intervention shows …; in Qwen2.5-3B a gold-label direction at the end of the hypothesis has a small, replicated effect on the answer (… logits at natural scale)". That would be the first positive causal result of the project and would need its own audit before Sunday 08:00.

## What changes in RESULTS.md in every case (not done)

- Limitations, claim 3 bullet: "three models of one family, one layer per model at the end of the hypothesis, 20 controls, 12 comparisons without adjustment, an erasure that removes about half of the readable adjacency on held-out items".
- Retraction table, one row: "'pre-specified contrast'" / "templates and signs were read from all items, including the evaluated ones; conclusions unchanged with selection-half signs" / "Audit 8".
- Limitations, first bullet, once the 7B check is in.

## What I am unsure about

- Whether "shows" in the headline is honest wording for an arm at p = 0.05 among 12 that is consistent across every item, block and string. I think it is, because the uncertainty is in the draw of the direction and of 20 controls and not in the items; but it is a judgement, and it is yours.
- Whether the sign pattern (gold difference of means positive in 12 of 12) deserves a sentence in RESULTS.md. It was noticed after the fact and has an innocent explanation (within string the gold label goes with adjacency, and the adjacency difference of direct-causation items pushes towards Yes in every model). I have left it in the report (4.1) and out of the proposals above.

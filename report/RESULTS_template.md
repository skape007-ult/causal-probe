# Results in one page

**Question.** When small language models answer formal causal-reasoning questions wrongly, is the correct answer nevertheless linearly readable from their activations, and is it used?

**Run:** Qwen2.5-0.5B, 1.5B, 3B and Gemma-2-2B on Corr2Cause and CLadder; probes against text-only baselines, steering, patching; six independent audits. Numbers are from `results/*.json`; details and all caveats are in `report/report.md`.

## Three headline claims

**1. A probe's accuracy for the correct answer means nothing until it is compared with what the text alone gives away.**
- *Number:* on the official Corr2Cause test split a classifier that sees only the hypothesis, never the premise, reaches AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}. Gold-label probes are below the strongest text-only classifier in all eight model × benchmark cells.
- *Figure* (three of the eight cells): `fig5_confound_c2c_probe.png`. *Checked by* audit 1 (it found the shortcut and forced the first retraction) and audit 6 (recomputed for four models).

![probe vs text](figures/fig5_confound_c2c_probe.png)

**2. With every single-input shortcut removed, a probe on the larger models' activations is as accurate as a lookup of the premise statements about the two queried variables, no more; pooled over hypothesis types the models answer at chance.**
- *Number:* on the debiased pool Qwen2.5-3B's probe scores {{probe_content_qwen3b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}, the lookup {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|est}}, a hand-written parse {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}; the model's answers have AUROC {{answer_bias.json|qwen3b/c2c_debiased/auroc|ci}}.
- *Caveats:* equal accuracy is not equal content (probe and text models are right on partly different items); within a hypothesis type the answers deviate from chance in both directions.
- *Figure:* `fig13_debiased_ladder.png`. *Checked by* audit 3 (two models; it replaced my baseline list), audits 5 and 6 (four models, recomputed).

![ladder](figures/fig13_debiased_ladder.png)

**3. No intervention shows that what a probe reads is what the answer is computed from.**
- *Number:* in Qwen2.5-0.5B, shifting items at the end of the hypothesis along the encoding of adjacency (a structural property the answers depend on) gives a pre-specified contrast of {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|translation/contrast|signed}} out of a possible {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|full_mediation_benchmark|num}}: above {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|translation/contrast/vs_controls_norm_matched/share_of_controls_below|pct}} of 20 label-shuffled directions of the same size, larger in absolute value than {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|translation/contrast/vs_controls_norm_matched/share_of_controls_smaller_in_absolute_value|pct}}. That is the weakest result 20 controls can register, and the effect is a push towards No for every hypothesis type.
- *Caveats:* at the answer position the same shift removes the dependence for one hypothesis type and creates one for another; the answer is already a linear function of the activation there (held-out R² {{adjacency_qwen0.5b_c2c_debiased_final_best.json|audit5/within_string_answer_linearly_predictable/held_out_r2|num2}}), so that locates the answer. Steering along the gold-label probe's weight vector stays within the range of matched controls per unit of norm. One model so far.
- *Figure:* `fig12_adjacency_end_of_hypothesis.png`. *Checked by* audits 2, 4, 5 and 6.

![adjacency](figures/fig12_adjacency_end_of_hypothesis.png)

## What we retracted, and why

| Claim as first made | What was wrong | Caught by |
|---|---|---|
| "Qwen2.5-1.5B represents the answer but doesn't say it" | Baselines too weak: the hypothesis string alone predicts {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|pct}} of labels | Audit 1 |
| "The gold-probe direction is inert; the model-output direction flips answers" | One sign shown, three random directions, steps far beyond natural scale | Audit 2 |
| "Patching shows where the answer is formed" | It shows where a prompt difference arrives | Audit 2 |
| "The probe beats the shallow controls and adds nothing to the parser" | Controls at 50% by construction; an insensitive test | Audit 3 |
| "The answers do not express what the probe reads" | False per hypothesis type; opposite signs cancel | Audit 3 |
| First adjacency experiment | Its "removal" left the property readable | Audit 4 |
| "Erasing adjacency removes a sixth of the dependence, beyond all 20 controls"; "at the answer position, almost all" | A control that could not fail; a summary in which opposite errors cancelled | Audit 5 |
| "The model reacts to letters out of alphabetical order" | A one-sided pair construction | Audit 5 |
| "Splitting the class difference shows an answer shift, not an adjacency variable" | True by construction; no control | Audit 6 |
| Four bugs (a CLadder sample differing between machines; an SAE interval; a fix reported but not applied; an ingest that could overwrite newer results) | `notes/log.md` | Ingest check, own review, audits 4 to 6 |

## Limitations

- Models up to 3 billion parameters, at or near chance; two benchmarks; linear probes. Nothing here says what a model that can do the task would show (a 7B behaviour check is running).
- Claim 3: one model, four sites, 20 controls, and an erasure that removes about half of the readable adjacency on held-out items. "No intervention shows" is not "unused".
- The stronger baselines, the debiased pool and every corrected analysis were introduced after seeing results; the pre-registered analysis is reported next to them (report Section 3.4).
- Probe accuracies are three-seed ensembles; for Gemma-2-2B and Qwen2.5-3B only final-token activations exist.
- Qwen2.5-3B's CLadder numbers are on a different draw of the sample (marked † in the tables).

# Results in one page

**Question.** When small language models answer formal causal-reasoning questions wrongly, is the correct answer nevertheless linearly readable from their activations, and is it used?

**Run:** Qwen2.5-0.5B, 1.5B, 3B and Gemma-2-2B on Corr2Cause and CLadder: probes against text-only baselines, steering, patching. Qwen2.5-7B (4-bit): answers, and probes on the debiased pool. Fifteen reviews by fresh reviewer agents; those of results were given code and files, not my reading. Numbers are from `results/*.json`; details and all caveats are in `report/report.md`.

## Three headline claims

**1. A probe's accuracy for the correct answer means nothing until it is compared with what the text alone gives away.**
- *Number:* on the official Corr2Cause test split (21 premises) a classifier that sees only the hypothesis, never the premise, reaches AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}. Gold-label probes are below the strongest text-only classifier in all eight model × benchmark cells.
- *Figure:* four of the eight cells. *Checked by* audits 1 and 6.

![probe vs text](figures/fig5_confound_c2c_probe.png)

**2. With every single-input shortcut removed, a probe on the activations of the models from 1.5B to 7B is about as accurate as a lookup of the premise statements about the two queried variables, and less accurate than a hand-written parse; ranked across all items at once, the models' answers are at chance.**
- *Number:* on the debiased pool Qwen2.5-3B's probe scores {{probe_content_qwen3b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}, the lookup {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|est}}, a hand-written parse {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}; its answers have AUROC {{answer_bias.json|qwen3b/c2c_debiased/auroc|ci}}. Qwen2.5-7B: probe {{probe_content_qwen7b-4bit_c2c_debiased.json|baselines/gold_probe/accuracy|ci}} (its gain over the parse-plus-4-gram text model, {{probe_content_qwen7b-4bit_c2c_debiased.json|incremental_value/over_strongest_text_only/gain_accuracy|signedci}}, does not survive a correction over the five models and shrinks to {{probe_content_qwen7b-4bit_c2c_debiased.json|incremental_value/over_tuned_text_only/gain_accuracy|signedci}} when that text model is tuned); answers {{answer_bias.json|qwen7b-4bit/c2c_debiased/auroc|ci}}.
- *Caveats:* equal accuracy is not equal content (probe and text models are right on partly different items). Inside a hypothesis string the four smaller models' answers do not follow validity (AUROC {{answer_bias.json|gemma2b/c2c_debiased/within_hypothesis_string/mean_auroc|num2}} to {{answer_bias.json|qwen3b/c2c_debiased/within_hypothesis_string/mean_auroc|num2}}); they depend on adjacency of the queried pair. The 7B model's follow validity on average, weakly, and go against it for direct-causation hypotheses ({{answer_bias.json|qwen7b-4bit/c2c_debiased/within_hypothesis_string/by_type/parent|num}}): AUROC within string {{answer_bias.json|qwen7b-4bit/c2c_debiased/within_hypothesis_string/mean_auroc|ci}}; the valid item of a premise gets the higher answer logit in {{answer_bias.json|qwen7b-4bit/c2c_debiased/within_hypothesis_string/valid_item_above_invalid_item_of_the_same_premise|pct1}} of premises (Qwen2.5-3B: {{answer_bias.json|qwen3b/c2c_debiased/within_hypothesis_string/valid_item_above_invalid_item_of_the_same_premise|pct1}}).
- *Checked by* audits 3, 5, 6, 11 and 14.

![ladder](figures/fig13_debiased_ladder.png)

**3. No intervention shows that what a probe reads is what the answer is computed from.**
- *Number:* steering along a gold-label direction was compared with 20 matched control directions {{steering_controls_multiplicity.json|summary/all/n_comparisons|int}} times (four models, two item pools, two tokens, two kinds of direction). The gold direction exceeds all its controls in {{steering_controls_multiplicity.json|summary/all/n_rank_1_in_absolute_slope|int}}, where chance gives {{steering_controls_multiplicity.json|summary/all/expected_rank_1_by_chance|num2}}; on the debiased pool in {{steering_controls_multiplicity.json|summary/debiased_pool/n_rank_1_in_absolute_slope|int}} of {{steering_controls_multiplicity.json|summary/debiased_pool/n_comparisons|int}} ({{steering_controls_multiplicity.json|summary/debiased_pool/expected_rank_1_by_chance|num2}}). None survives a correction for multiple comparisons, and with 20 controls none could.
- *The four:* Qwen2.5-3B, end of the hypothesis, class-mean difference ({{steering_controls_qwen3b_c2c_debiased_hyp.json|real/gold_diffmeans/slope_per_s|signed}}; about half of it is the adjacency contrast of direct-causation hypotheses, so it cannot separate validity from adjacency; {{direction_geometry_qwen3b_c2c_debiased_hyp.json|natural_scale/effect_of_moving_across_the_class_gap_logits|signed}} logits at natural scale): undecided. Qwen2.5-1.5B, same token, probe direction ({{steering_controls_qwen1.5b_c2c_debiased_hyp.json|real/gold_probe/slope_per_s|signed}}: towards "valid" lowers Yes): in a test fixed beforehand, with the direction refitted on the other half of the data, it kept its sign at under half the size ({{steering_settle_qwen1.5b_c2c_debiased_hyp.json|primary_test/slope|signed}}) and lay inside its 500 controls (p = {{steering_settle_qwen1.5b_c2c_debiased_hyp.json|primary_test/p_one_sided|num2}}). Gemma-2-2B, answer position, probe direction ({{steering_controls_gemma2b_c2c_debiased.json|real/gold_probe/slope_per_s|signed}}; first by 3%, not robust): undecided. Qwen2.5-0.5B, original pool, answer position, class-mean difference ({{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/slope_per_s|signed}}; weaker controls): undecided.
- *Caveats:* taken together the gold class-mean differences do rank above their controls (mean rank {{steering_controls_multiplicity.json|summary/all/mean_rank/gold_diffmeans/mean_abs_rank|num2}} of 21; 11 expected). At the answer position on the debiased pool that is the answers' own within-question correlation with the label read back (three of four models) and does not show that validity is used; at the end of the hypothesis it is unexplained. Shifting items along the encoding of adjacency at the end of the hypothesis is outside its 20 controls in Qwen2.5-0.5B only ({{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|translation/contrast|signed}} of a possible {{adjacency_qwen0.5b_c2c_debiased_hyp_best.json|full_mediation_benchmark|num}}, a push towards No for every hypothesis type); Gemma-2-2B's layer there responds to nothing. At the answer position the same shift moves the answer (beyond all controls in two of four models), and a given shift acts the same way on items of every hypothesis type, so it locates the answer.
- *Checked by* audits 2, 4, 5, 6, 8, 11 and 15.

![adjacency](figures/fig12_adjacency_end_of_hypothesis.png)

## What we retracted, and why

| Claim as first made | What was wrong | Caught by |
|---|---|---|
| "Qwen2.5-1.5B represents the answer but doesn't say it" | Baselines too weak: the hypothesis string alone predicts {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|pct}} of labels | Audit 1 |
| "The gold-probe direction is inert; the model-output direction flips answers"; "patching shows where the answer is formed" | One sign shown, three random directions, steps far beyond natural scale; patching shows where a prompt difference arrives | Audit 2 |
| "The probe beats the shallow controls and adds nothing to the parser"; "the answers do not express what the probe reads" | Controls at 50% by construction, an insensitive test; opposite signs by hypothesis type cancel | Audit 3 |
| Two adjacency experiments: "erasing adjacency removes a sixth of the dependence, beyond all 20 controls" | The first "removal" left the property readable; the second used a control that could not fail | Audits 4 and 5 |
| "A pre-specified contrast"; "steering stays within the range of its controls" | Signs were read from the evaluated items too; false for three of the four models once both gold directions are counted | Audit 8 |
| Eight bugs that reached a result or a run | `notes/log.md` | Ingest check, own review, audits 4 and 6 |

More, including three of 14 November (first ranks read as chance; "the 7B model answers at chance"; an untuned "strongest text-only model"), are in report Section 6.

## Limitations

- Interventions on models of up to 3 billion parameters whose answers do not follow validity. The 7B model follows it weakly, was run in 4-bit quantisation, and was not intervened on (a test was costed and not run; report Section 9). Two benchmarks; linear probes. Nothing here says what a model that can do the task would show.
- Claim 3: four models, three of one family; at most one responding layer per model at the end of the hypothesis; an erasure that removes about half of the readable adjacency on held-out items. "No intervention shows" is not "unused".
- The stronger baselines, the debiased pool, the within-string statistics and every corrected analysis were introduced after seeing results; the pre-registered analysis is reported next to them (report Section 3.4).

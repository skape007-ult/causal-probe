# Results in one page

**Question.** When small language models answer formal causal-reasoning questions wrongly, is the correct answer nevertheless linearly readable from their activations, and is it used?

**Run:** Qwen2.5-0.5B, 1.5B, 3B and Gemma-2-2B on Corr2Cause and CLadder: probes against text-only baselines, steering, patching. Qwen2.5-7B (4-bit): answers, and probes on the debiased pool. Sixteen reviews by fresh reviewer agents; those of results were given code and files, not the conclusions. Numbers are from `results/*.json`; details and all caveats are in `report/report.md`.

**Not in this page:** the held-out test of the Qwen2.5-3B case and patching on all available pairs for Qwen2.5-3B (launched 14 Nov).

## Three headline claims

**1. A probe's accuracy for the correct answer means nothing until it is compared with what the text alone gives away.**
- *Number:* on the official Corr2Cause test split (21 premises) a classifier that sees only the hypothesis, never the premise, reaches AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}. Gold-label probes are below the strongest text-only classifier in all eight model × benchmark cells.
- *Figure:* [probe against text, four of the eight cells](figures/fig5_confound_c2c_probe.png).

**2. With every single-input shortcut removed, a probe on the activations of the models from 1.5B to 7B is about as accurate as a lookup of the premise statements about the two queried variables, and less accurate than a hand-written parse; ranked across all items at once, the models' answers are at chance.**
- *Number:* on the debiased pool Qwen2.5-3B's probe scores {{probe_content_qwen3b_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}, the lookup {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy|est}}, a hand-written parse {{probe_content_qwen3b_c2c_debiased.json|baselines/B_binding_text_reference/hand_parse_all_features/accuracy|est}}; its answers have AUROC {{answer_bias.json|qwen3b/c2c_debiased/auroc|ci}}. Qwen2.5-7B: probe {{probe_content_qwen7b-4bit_c2c_debiased.json|baselines/gold_probe/accuracy|ci}}; answers {{answer_bias.json|qwen7b-4bit/c2c_debiased/auroc|ci}}.
- *Caveats:* inside a hypothesis string the four smaller models' answers do not follow validity (AUROC {{answer_bias.json|gemma2b/c2c_debiased/within_hypothesis_string/mean_auroc|num2}} to {{answer_bias.json|qwen3b/c2c_debiased/within_hypothesis_string/mean_auroc|num2}}); they depend on adjacency of the queried pair. The 7B model's follow validity on average, weakly: AUROC within string {{answer_bias.json|qwen7b-4bit/c2c_debiased/within_hypothesis_string/mean_auroc|ci}}, about {{answer_bias.json|qwen7b-4bit/c2c_debiased/within_hypothesis_string/prompt_length_removed/quadratic_in_log_token_count/mean_auroc|num2}} with prompt length removed; the valid item of a premise gets the higher answer logit in {{answer_bias.json|qwen7b-4bit/c2c_debiased/within_hypothesis_string/valid_item_above_invalid_item_of_the_same_premise|pct1}} of premises (Qwen2.5-3B: {{answer_bias.json|qwen3b/c2c_debiased/within_hypothesis_string/valid_item_above_invalid_item_of_the_same_premise|pct1}}). Gemma-2-9B has the same raw figure ({{answer_bias.json|gemma9b/c2c_debiased/within_hypothesis_string/mean_auroc|num}}) and it is not shown beyond a response to prompt length ({{answer_bias.json|gemma9b/c2c_debiased/within_hypothesis_string/prompt_length_removed/quadratic_in_log_token_count/mean_auroc|num2}} with length removed).

![ladder](figures/fig13_debiased_ladder.png)

**3. No intervention shows that what a probe reads is what the answer is computed from.**
- *Number:* steering along a gold-label direction was compared with 20 matched control directions {{steering_controls_multiplicity.json|summary/all/n_comparisons|int}} times (four models, two item pools, two tokens, two kinds of direction). The gold direction exceeds all its controls in {{steering_controls_multiplicity.json|summary/all/n_rank_1_in_absolute_slope|int}}, where chance gives {{steering_controls_multiplicity.json|summary/all/expected_rank_1_by_chance|num2}}; on the debiased pool in {{steering_controls_multiplicity.json|summary/debiased_pool/n_rank_1_in_absolute_slope|int}} of {{steering_controls_multiplicity.json|summary/debiased_pool/n_comparisons|int}} ({{steering_controls_multiplicity.json|summary/debiased_pool/expected_rank_1_by_chance|num2}}). None survives a correction for multiple comparisons, and with 20 controls none could.
- *The four:* Qwen2.5-3B, end of the hypothesis, class-mean difference ({{steering_controls_qwen3b_c2c_debiased_hyp.json|real/gold_diffmeans/slope_per_s|signed}}); Qwen2.5-1.5B, same token, probe direction ({{steering_controls_qwen1.5b_c2c_debiased_hyp.json|real/gold_probe/slope_per_s|signed}}; inside its 500 controls in a test fixed beforehand, p = {{steering_settle_qwen1.5b_c2c_debiased_hyp.json|primary_test/p_one_sided|num2}}); Gemma-2-2B, answer position, probe direction ({{steering_controls_gemma2b_c2c_debiased.json|real/gold_probe/slope_per_s|signed}}, first by 3%); Qwen2.5-0.5B, original pool, class-mean difference ({{steering_controls_qwen0.5b_c2c_probe.json|real/gold_diffmeans/slope_per_s|signed}}). Three are undecided and one was not confirmed.
- *Caveats:* taken together the gold class-mean differences do rank above their controls (mean rank {{steering_controls_multiplicity.json|summary/all/mean_rank/gold_diffmeans/mean_abs_rank|num2}} of 21; 11 expected). At the answer position on the debiased pool that is the answers' own within-question correlation with the label read back (three of four models) and does not show that validity is used; at the end of the hypothesis it is unexplained. Shifting items along the encoding of adjacency before the answer position is outside its controls in one of four models ([figure](figures/fig12_adjacency_end_of_hypothesis.png)); at the answer position it moves the answer the same way for every hypothesis type, which locates the answer.

*Checked by:* claim 1, audits 1 and 6; claim 2, audits 3, 5, 6, 11 and 14; claim 3, audits 2, 4, 5, 6, 8, 11 and 15.

## What was retracted, and why

| Claim as first made | What was wrong | Caught by |
|---|---|---|
| "Qwen2.5-1.5B represents the answer but doesn't say it" | Baselines too weak: the hypothesis string alone predicts {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|pct}} of labels | Audit 1 |
| "The gold-probe direction is inert; the model-output direction flips answers" | One sign shown, three random directions, steps far beyond natural scale | Audit 2 |
| Two adjacency experiments: "erasing adjacency removes a sixth of the dependence, beyond all 20 controls" | The first "removal" left the property readable; the second used a control that could not fail | Audits 4 and 5 |

The other retractions, each with the review that caught it, are in report Section 6.

## Limitations

- Interventions on models of up to 3 billion parameters whose answers do not follow validity. The 7B model follows it weakly, was run in 4-bit quantisation, and was not intervened on. Inside a hypothesis type the pool still leaks through prompt length, which Gemma-2-9B's answers follow. Two benchmarks; linear probes. Nothing here says what a model that can do the task would show.
- Claim 3: four models, three of one family. "No intervention shows" is not "unused".
- The stronger baselines, the debiased pool, the within-string statistics and every corrected analysis were introduced after seeing results; the pre-registered analysis is reported next to them (report Section 3.4).

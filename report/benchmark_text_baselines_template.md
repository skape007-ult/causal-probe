# Text-only baselines on Corr2Cause and CLadder: a note for probing studies

This note reports how classifiers that see only the prompt text, and no language model, perform on
two causal-reasoning benchmarks. It was written as part of a probing study (see `report/report.md`):
when a probe is trained on a model's activations to predict the gold label, the right comparison is
what the same label can be predicted from without the model. The numbers below describe how such
baselines perform on the released data. They are a property of the data and the evaluation
protocol, and are stated as such; nothing here is a claim about the purpose or the quality of the
benchmarks.

## Data and reproduction

| | Version used |
|---|---|
| Corr2Cause | Hugging Face `causal-nlp/corr2cause`, revision `42ba12c769e11ff6427c9f52d7db58e3f9bf3e53`; files `train.csv`, `dev.csv`, `test.csv` (the v2.0 data of Jin et al., 2024, Table 3: 205,734 / 1,076 / 1,162 items) |
| CLadder | Hugging Face `causal-nlp/CLadder`, revision `18d684e444ed07cc57ae0f3a91ee983a79b8119a`; file `data/full_v1.5_default.csv` (10,112 items) |

Every number in this note is produced by one script and read from its output file
`results/benchmark_baselines.json` (CPU only, about four minutes, no model needed):

```bash
python scripts/download_data.py
```

```bash
python scripts/14_benchmark_baselines.py
```

The first command downloads the files at the revisions above and verifies their MD5 sums. Labels
are used as released. Public reports of label errors in both datasets exist (Hugging Face
discussion #6 on `causal-nlp/corr2cause`; GitHub issues #15 and #16 on `causalNLP/cladder`); we
have not verified them and have not corrected any label.

Confidence intervals are 95% percentile bootstrap intervals that resample whole premises
(Corr2Cause) or whole stories (CLadder), because items that share a premise or a story are not
independent. Seeds are fixed (seed 0; cross-validation seeds 0, 1, 2).

## Corr2Cause

### Structure of the released splits

<!-- TABLE:benchmark_c2c_description -->

Three properties of the splits matter for evaluation:

1. **Few distinct premises in dev and test.** The test split has {{benchmark_baselines.json|corr2cause_description/test/n_distinct_premises|int}} distinct premises and the dev split {{benchmark_baselines.json|corr2cause_description/dev/n_distinct_premises|int}}, with a median of {{benchmark_baselines.json|corr2cause_description/test/items_per_premise_median|int}} hypotheses each. The paper reports item counts per number of variables (Table 3) but not premise counts; ours are counted from the released files. An interval that treats the {{benchmark_baselines.json|corr2cause_description/test/n_items|int}} test items as independent is therefore too narrow. All intervals here resample premises.
2. **Premises shared between splits.** {{benchmark_baselines.json|corr2cause_description/premise_overlap/train&dev|int}} premises occur in both train and dev ({{benchmark_baselines.json|corr2cause_description/n_dev_items_with_premise_in_train|int}} dev items) and {{benchmark_baselines.json|corr2cause_description/premise_overlap/dev&test|int}} in both dev and test; train and test share {{benchmark_baselines.json|corr2cause_description/premise_overlap/train&test|int}}. Appendix D of the paper describes removing duplicate *items* across splits; these are the same premise with different hypotheses. The same counts (three and two) are stated in the README of an unreviewed course project (github.com/abhijeetnst/dl-aml, October 2026).
3. **Hypothesis strings and labels.** There are {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n|int}} distinct hypothesis strings. In train, {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n_never_valid|int}} of them are never valid, {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n_always_valid|int}} are always valid and {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/n_both|int}} occur with both labels. {{benchmark_baselines.json|corr2cause_description/train_hypothesis_strings/frac_of_train_items_whose_string_is_never_valid|pct1}} of train items have a hypothesis string that is never valid. The second table shows the pattern behind this: a directed claim is valid only when the cause is named with an earlier letter than the effect.

Point 3 follows from the published construction, although the paper does not state it. Graphs are
generated with nodes in topological order, so that only edges from a lower to a higher index exist
(Sec. 3.3), and the public code verbalises index *i* as the *i*-th letter. The generating graph is a
member of its own Markov equivalence class, and a hypothesis is valid only if it holds in every
member, so a directed claim can be valid only from an earlier to a later letter. The v2.0
de-duplication keeps one of the identical strings Is-Parent(i, j) / Is-Child(j, i) by forcing
i < j (Sec. 3.7 footnote, Appendix D), which leaves the `child` template with later-to-earlier
claims only. The paper's Appendix E does report that some n-grams tied to variable names are
correlated with the label, and its variable-refactorisation test (reversing the alphabet) is
designed to break such cues.

### Text-only baselines on the official splits

All baselines are fitted on train and evaluated on dev and test. "Hypothesis only" assigns each
item the fraction of train items with the same hypothesis string that are valid; since every
hypothesis-only classifier is a function of that string, this is the best such a classifier can do
on the training distribution. "Premise only" is a TF-IDF logistic regression on the premise text.
"Parsed features" is gradient boosting on shallow parsed features of premise and hypothesis (are
the two named variables correlated; are they adjacent in the skeleton, that is, never rendered
independent by any conditioning set; how many separating sets; shared neighbours; template) plus
the hypothesis-string prior. It applies no orientation rule and enumerates no equivalence class.
The two learned baselines are fitted on a seeded sample of {{benchmark_baselines.json|corr2cause_official_splits/fit/n_train_items_for_learned_baselines|int}} train items.

<!-- TABLE:benchmark_c2c_official -->

Reading:

- The hypothesis string alone ranks valid above invalid test items with AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/auroc|ci}}. Because only {{benchmark_baselines.json|corr2cause_description/test/frac_valid|pct1}} of test items are valid and few strings are valid more often than not, its accuracy ({{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/accuracy|ci}}) is close to the majority-class accuracy ({{benchmark_baselines.json|corr2cause_official_splits/test/majority/accuracy|est}}). Its F1 on the valid class is {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only/positive_f1|ci}}, or {{benchmark_baselines.json|corr2cause_official_splits/test/hypothesis_only_f1_threshold/positive_f1|ci}} with a decision threshold chosen on train to maximise F1.
- The premise alone also carries label information (AUROC {{benchmark_baselines.json|corr2cause_official_splits/test/premise_only/auroc|ci}}): premises differ in how many of their hypotheses are valid.
- Parsed features reach accuracy {{benchmark_baselines.json|corr2cause_official_splits/test/parsed_features/accuracy|ci}} and valid-class F1 {{benchmark_baselines.json|corr2cause_official_splits/test/parsed_features/positive_f1|ci}} on test, without any causal-discovery rule beyond the skeleton. Removing the dev premises that also occur in train changes the dev numbers by less than one point.
- With 21 and 22 premises, these intervals are wide. They would be several times narrower, and wrong, if items were resampled individually.

For scale, the paper reports a best F1 of 33.38 among off-the-shelf models (BART MNLI, Table 4)
and 94.74 for the best fine-tuned model (RoBERTa-Large MNLI, Table 5a), against 0.0 for the
always-majority baseline. We did not rerun those models. The paper describes the released v2.0
data as a de-duplicated version of the original and says it expects results to change little
(Sec. 3.7 footnote, App. D); we could not establish from the text which version the reported
numbers were computed on, so the comparison with our numbers on the released v2.0 files is
indicative only: the hypothesis-only baseline sits above the
off-the-shelf models and well below the fine-tuned ones on valid-class F1.

### Label-balanced pools (as used for probing)

Probing needs a label-balanced item set. The table shows the same baselines under
premise-held-out five-fold cross-validation on three pools drawn from train.

<!-- TABLE:benchmark_c2c_pools -->

- **Balanced pool** (300 premises, 4 valid and 4 invalid hypotheses each). Balancing labels within premises puts the premise-only baseline at exactly 0.50, but the hypothesis string alone gives {{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|ci}} accuracy. A probe on this pool can score in the high 0.8s by decoding which hypothesis is being asked.
- **Rejected first design.** Balancing labels within each hypothesis string over the whole pool is not enough. With premise-held-out folds, a string that is mostly valid in the test fold is mostly invalid in the training folds, so a hypothesis-only classifier falls *below* chance ({{benchmark_baselines.json|corr2cause_string_balanced_only_pool_rejected/baselines/hypothesis_only/accuracy|ci}}), and the premise-only baseline rises to {{benchmark_baselines.json|corr2cause_string_balanced_only_pool_rejected/baselines/premise_only/accuracy|ci}} because valid items come mostly from premises with many valid hypotheses.
- **Debiased pool** (`causal_probe.data.c2c_debiased_pool`; 2,400 items, 1,200 premises, the 48 hypothesis strings that occur with both labels). Each premise contributes exactly one valid and one invalid hypothesis; the pool consists of 25 blocks in which each string is used once as valid and once as invalid; folds are unions of blocks. Any classifier that sees only the hypothesis, or only the premise, scores exactly 0.50 in every fold by construction, and the measured values agree (hypothesis only {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/hypothesis_only/accuracy|est}}, premise only {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/premise_only/accuracy|est}}, whole-prompt TF-IDF {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/full_text_tfidf/accuracy|ci}}). What remains predictive is the combination of premise and hypothesis, which is the task: parsed features reach {{benchmark_baselines.json|corr2cause_debiased_pool/baselines/parsed_features/accuracy|ci}}.

## CLadder

Story-held-out five-fold cross-validation on all {{benchmark_baselines.json|cladder_full_v1.5/n|int}} items of `full_v1.5_default.csv` ({{benchmark_baselines.json|cladder_full_v1.5/n_stories|int}} stories). "Quoted numbers + question polarity" is gradient boosting on the probabilities quoted in the prompt, simple differences between them, polarity words in the question ("increase", "less likely", "not"), the query type and the graph type, and, for backdoor-adjustment questions, which of the two described methods conditions "case by case". No adjustment formula is evaluated.

<!-- TABLE:benchmark_cladder -->

Reading:

- Bag-of-words baselines are at chance ({{benchmark_baselines.json|cladder_full_v1.5/baselines/question_only/accuracy|ci}} question only; {{benchmark_baselines.json|cladder_full_v1.5/baselines/full_text/accuracy|ci}} whole prompt), consistent with the dataset's balance across stories and query types.
- The numbers-and-polarity baseline reaches {{benchmark_baselines.json|cladder_full_v1.5/baselines/numeric_polarity/accuracy|ci}} on held-out stories. It is at 1.00 for `backadj` and `collider_bias`, above 0.9 for `ate`, `ett` and `marginal`, and at chance for `det-counterfactual` ({{benchmark_baselines.json|cladder_full_v1.5/numeric_polarity_by/query_type/det-counterfactual/accuracy|ci}}).

Most of this is already documented. Clarke et al. (2026) report 86.8% accuracy for a random forest
on similar surface features (quoted probabilities, comparisons between them, keyword flags) under
ten-fold cross-validation on 9,866 items regenerated with the CLadder code, with a per-type
breakdown (their Sec. 3.2; Table 3 in App. A.4; features in App. A.8) and an account of the
generator rules responsible (App. A.7). They do not say whether folds keep stories together. Pacchiardi et al. (2024) include CLadder in a study of
how well word 2-grams predict benchmark answers, which is the counterpart of the bag-of-words rows
above. The CLadder paper
itself notes that collider-bias questions have a fixed answer in canonical form (App. A.2). What
this note adds is narrow: the estimate on the released v1.5 file with stories held out, and the
observation that with an explicit feature for which method is the case-by-case one, `backadj`
questions are fully predictable in this file (Clarke et al. report 62.2% for that type on 74
items with their feature set).

## What is and is not new here

| Observation | Status |
|---|---|
| Corr2Cause label imbalance; most items have six variables | Reported in the paper (Tables 1, 3, 4) |
| Directed hypotheses are valid only from an earlier to a later letter; the `child` template is never valid | Follows from the paper's construction (Sec. 3.3; App. D); not stated there. The course-project README cited above notes that "B directly causes A" is never valid because causes are ordered alphabetically. Quantified here |
| Performance of a hypothesis-only classifier on Corr2Cause (official splits and balanced pools); 72 of 120 strings never valid | We found no earlier report |
| Number of distinct premises per split; premises shared between splits | Not in the paper. The overlap counts match those in the course-project README cited above |
| CLadder answers are largely predictable from quoted numbers and question polarity | Clarke et al. (2026); CLadder App. A.2 for collider bias |
| Story-held-out estimate on `full_v1.5_default.csv`; `backadj` fully predictable with one extra feature | Reported here |
| A premise-paired, string-balanced Corr2Cause pool on which hypothesis-only and premise-only classifiers are at chance in every fold | Constructed here. The nearest published item we found is the hand-balanced evaluation sample of Bagheri et al. (2025, Sec. 4.2): 60 valid and 60 invalid items with equal counts per number of variables, balanced on labels and variable count but not on hypothesis strings or premises |

"We found no earlier report" means exactly that. The search behind this table is recorded in
`notes/prior_work.md`: every paper listed there was opened in the session (abstract or full text,
marked per entry), the queries are listed so that the search can be repeated, and unverified
leads are listed separately. Its limits: pages were read through a tool that summarises them, PDFs
without an HTML version could not be read, and OpenReview and Google Scholar could not be opened.
An earlier, quicker pass is in `notes/literature_check.md`.

### Related work that uses these benchmarks

- **Label imbalance on Corr2Cause** is stated in the original paper and in several follow-ups that
  therefore report F1 on the valid class (Sgouritsa et al., 2024; Bagheri et al., 2025; Sun et al.,
  2025; Kadziolka and Salehkaleybar, 2025). None of the follow-ups we opened reports a
  hypothesis-only, premise-only or other partial-input classifier.
- **Probing on CLadder.** Ding and Zhang (2026) train linear probes on last-token states for the
  causal direction of paired CLadder prompts and report probe accuracy of at least 0.969 against
  Yes/No output accuracy of 0.35 to 0.53, with a control task in the sense of Hewitt and Liang.
  Yu (2026) probes for the family of variable names. Neither reports a text-only classifier for
  the probed label. We found no probing, patching, steering or sparse-autoencoder study on
  Corr2Cause.

## Why this matters for probing studies

A linear probe on activations of a model reading a Corr2Cause prompt can reach high accuracy by
decoding the hypothesis string, which any model that reads the prompt will represent. On the
balanced pool used in the first part of our study the gold-label probes scored
{{confound_qwen0.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} (Qwen2.5-0.5B),
{{confound_qwen1.5b_c2c_probe.json|all_items/gold_probe/accuracy|est}} (Qwen2.5-1.5B) and
{{confound_gemma2b_c2c_probe.json|all_items/gold_probe/accuracy|est}} (Gemma-2-2B) on held-out premises, which
looked like evidence that the models represent the answer; on that pool the hypothesis-only baseline scores
{{benchmark_baselines.json|corr2cause_balanced_pool/baselines/hypothesis_only/accuracy|est}} and the parsed-feature baseline
{{benchmark_baselines.json|corr2cause_balanced_pool/baselines/parsed_features/accuracy|est}}. The comparison that carries weight is
therefore probe against the strongest text-only baseline on the same held-out items, on an item
set where single-input shortcuts are removed by construction. The same holds for CLadder, where
the relevant baseline is the numbers-and-polarity classifier and not bag-of-words.

This parallels hypothesis-only baselines in natural language inference (Poliak et al., 2018;
Gururangan et al., 2018) and control tasks for probes (Hewitt and Liang, 2019).

## Limits of this note

- The baselines are specific classifiers; stronger text-only models would score higher. The numbers are lower bounds on what the text gives away.
- The parsed-feature baseline uses the hypothesis-string prior as one feature and so inherits that cue on the official splits. On the debiased pool that feature is constant.
- Models evaluated zero-shot have not seen the label distribution, so these results do not explain zero-shot scores. They are relevant to fine-tuned models, few-shot prompts drawn from the same distribution, and probes or other classifiers trained on in-distribution items.
- Dev and test have 22 and 21 premises; intervals on them are wide.
- Labels are taken as released (see the note on reported label errors above).

## References

- Jin, Z., Liu, J., Lyu, Z., Poff, S., Sachan, M., Mihalcea, R., Diab, M., Schölkopf, B. (2024). Can Large Language Models Infer Causation from Correlation? ICLR 2024. arXiv:2306.05836.
- Jin, Z., Chen, Y., Leeb, F., Gresele, L., Kamal, O., Lyu, Z., Blin, K., Gonzalez Adauto, F., Kleiman-Weiner, M., Sachan, M., Schölkopf, B. (2023). CLadder: Assessing Causal Reasoning in Language Models. NeurIPS 2023. arXiv:2312.04350.
- Clarke, J., Karn, I., Murray, P., Hill, KJ, Jensen, D. (2026). Evaluating Causal Reasoning in LLMs: A Case Study in the Challenges of Procedural Generation. arXiv:2610.38100.
- Pacchiardi, Tesic, Cheke, Hernández-Orallo (2024). Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers. arXiv:2410.11672.
- Sgouritsa, Aglietti, Teh, Doucet, Gretton, Chiappa (2024). Prompting Strategies for Enabling Large Language Models to Infer Causation from Correlation. arXiv:2412.13952.
- Bagheri, Alinejad, Dehshiri, Bello, Akhondi-Asl (2025). C²P: Featuring Large Language Models with Causal Reasoning. arXiv:2407.18069.
- Sun, Nogueira, Silva (2025). Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks. arXiv:2505.18034.
- Kadziolka, Salehkaleybar (2025). Causal Reasoning in Pieces: Modular In-Context Learning for Causal Discovery. arXiv:2507.23488.
- Ding, Zhang (2026). Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express. EMNLP 2026. arXiv:2605.25891.
- Yu (2026; surnames only for the seven entries above: first names were not verified). Vernier: Probing Representational Misalignment Behind Lexical Gaps in Causal Reasoning. arXiv:2606.15733.
- Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., Van Durme, B. (2018). Hypothesis Only Baselines in Natural Language Inference. *SEM 2018.
- Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., Smith, N. A. (2018). Annotation Artifacts in Natural Language Inference Data. NAACL-HLT 2018.
- Hewitt, J., Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019.

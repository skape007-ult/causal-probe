# Handoff for the week after 15 November

Written 14 and 15 November; final at the tag `v1.0`. For the owner. Every result named here is in `results/` and the report; nothing here is new.

## (a) What is finished

- **The write-up**, tagged `v1.0`: `RESULTS.md` (three claims, one page, checked sentence by sentence by a reviewer, final since 15 Nov 08:10), `report/report.md` (Section 8 "Interpretation and what I'd do next" left empty for you; Section 9 is the plan for the next phase), `EXPLAINER.md` (start here, primer, decision log D1 to D37, figure guide, 49 professor questions), `study/` (reading order for the week, 55 graded questions, six exercises), `notes/log.md` (expectation before and comparison after every experiment), `report/benchmark_text_baselines.md`.
- **The three claims, as they stand.** (1) Probe accuracy for the gold label means nothing without text-only baselines: a hypothesis-only classifier reaches AUROC 0.895 on the official Corr2Cause test split, and probes are below the strongest text-only classifier in all eight model × benchmark cells. (2) On the debiased pool a probe on the activations of the models from 1.5B to 9B is about as accurate as a lookup of the premise statements about the queried pair, below a hand-written parse, and adds nothing demonstrable to a tuned text-only stack; prompt length still leaks within a hypothesis type, and adding it to the text references changes nothing. (3) No intervention shows that what a probe reads is what the answer is computed from; four steering comparisons of 32 exceeded their 20 controls, the two tested again by a rule fixed beforehand did not stand out from 500 controls.
- **Reviews:** eighteen independent reviews and two claims passes, raw reports in `notes/` (`audit_*_raw.md`, `claims_pass_*_raw.md`); each is answered finding by finding in `notes/log.md`.
- **Reproduction:** `scripts/run_all.sh` regenerated every number of the project on 14 to 15 November (23:11 to 07:34, at `816248f`); every value that existed before came out identical (`notes/log.md`, 07:34). Steps in `README.md`, "Reproduction".
- **GPU runs:** all in `KAGGLE.md`, each notebook pinned to the commit it ran at; all downloads ingested and audited.
- **Deliberately not done:** `YOUR_TURN.md` (two small experiments, for you); `notes/upstream_issues.md` (not filed: you verify and file); Section 8 of the report.
- **One thing on the page that is now stale and that I did not change, because you closed the page at 08:10:** "Seventeen reviews". Eighteen are on file; the eighteenth (prompt length in the probe ladder, 07:35) confirmed claim 2 as worded and is listed in the report.

## (b) Next phase (nothing of it is built)

**Why there is a next phase at all.** One model, Qwen2.5-7B (4-bit), is the first whose answers follow validity inside a hypothesis string on the debiased pool: mean AUROC within string 0.549, about 0.53 to 0.54 with prompt length removed, about 0.55 with adjacency held fixed as well. Ranked across all items at once it is at chance, and the effect is small (the valid item of a premise gets the higher answer logit in 53.6% of premises) and rests on two of the five hypothesis types. Gemma-2-9B shows the same raw statistic (0.542), and its audit (`notes/audit_gemma9b_answers_raw.md`) found that it is not distinguishable from following prompt length: with the token count removed it is about 0.52 and no longer significant after any correction. These two results are the motivation for D1 (generated data, with a power calculation before every run) and for D3 (fine-tuning): one is the first sign of a model doing some of the task, the other shows that the present pool still leaks through prompt length inside a hypothesis type, and neither can carry an intervention (`notes/power_7b_option_b.md`: reviewed power about 0.2, and the tested direction is confounded with prompt length).

**The first question to settle there.** The two models disagree by hypothesis type, beyond noise (model-by-type interaction p ≤ 0.00005, paired over the items both answered). Part of that is explained and part is not. The disagreement on direct causation (7B 0.442, 9B 0.616) disappears once prompt length and adjacency are controlled: both sit at 0.50. The disagreement on indirect causation through an ancestor (7B about 0.62 to 0.67, 9B about 0.44) survives both controls and is unexplained; with one model per family, size, precision and tokeniser it cannot be attributed to any of them. It is the first thing to settle on data where each type has enough hypothesis strings (here 6 to 14), and where prompt length does not predict the label inside a hypothesis type.

**Design requirements for D1, in order.**
1. **Control prompt length by construction, within each hypothesis type.** Inside every hypothesis string, and inside every cell of string and adjacency, the length of the prompt must not predict the label. On the present pool it does: the token count alone predicts the label within string with AUROC 0.58 for direct causation and 0.40 for common cause (`answer_bias.json`, `prompt_length_alone`), and Gemma-2-9B's within-string result is not distinguishable from following it. A pool that is balanced only for the hypothesis alone and the premise alone (as the present one is: each premise appears once valid and once invalid) still leaks through the combination of premise length and hypothesis type.
2. Balanced within hypothesis template, with enough hypothesis strings per type that a type-level effect can be tested with strings as the unit (here 6 to 14 per type, which leaves a type-level p of about 0.04 at best).
3. No one-class cells of string and adjacency.
4. Held out by graph, where the number of distinct graphs allows it.
5. Every generated item labelled as generated, and kept apart from official-benchmark results in every table and figure.

**The same requirement for D3 (fine-tuning).** Training and evaluation sets must be length-balanced within hypothesis type. Otherwise a fine-tuned model can learn prompt length, which this pool would reward, and an apparent gain on the debiased pool would say nothing about causal structure. The evaluation must include the length-removed version of the within-string statistic (`15_answer_bias.py`, `prompt_length_removed`).

**The order for the next phase (the owner's decision, 15 November).**

1. **D3, fine-tuning, first.** LoRA fine-tune a Qwen model on Corr2Cause until it does the task in distribution, then ask with the existing tools what it learned: the causal structure or the hypothesis-wording shortcut. About 16 to 20 hours of work and 15 to 25 GPU hours. The original paper's fine-tuned models reach F1 94.74 in distribution and lose up to 39.29 and 62.30 points under paraphrase and variable renaming (its Tables 5a and 5b); it has no hypothesis-only baseline and does not look inside the models. The official training split has 205,734 items but only 2,338 premises, and the debiased pool uses 1,200 of them: keep those out of fine-tuning. **Length must be controlled here too:** training and evaluation sets length-balanced within hypothesis type, and the evaluation includes the within-string statistic with prompt length removed (`15_answer_bias.py`, `prompt_length_removed`); otherwise the fine-tuned model may learn length.
2. **D1, generated data, second** (its generator is CPU-only and can be built while D3's first GPU runs go). A generator that reproduces the Corr2Cause construction and is accepted only if it regenerates a sample of official items with exactly their labels; then a difficulty ladder (2 to 6 variables), a pool at least ten times the present one, and minimal pairs that differ in causal structure only. About 20 to 25 hours of work and 25 to 35 GPU hours. Its design requirements are listed above, with prompt length first. This is where the 7B intervention gets tested properly, each test with a power calculation reviewed before it runs.
3. **D2, Kaggle automation, only if the run count justifies it.** D3 and D1 together are 15 to 25 runs, which probably does. About 2 to 3 hours and one supervised dry run; needs a Kaggle API token that you place in `~/.kaggle/` yourself; two things are unverified against the current API (choosing T4 ×2; attaching a secret without the web editor).

**Known risks, in one line each.** With two or three variables there are only 2 and 11 distinct graph classes, so "held out by graph" means little at the easy end. A text-only classifier improves with data on a deterministic synthetic task, so every "the probe adds x" must name its text model and have it tuned. The easy regime may not exist for models of this size. A fine-tuned model may simply have learned the wording shortcut, which the original paper's perturbation results already suggest. The 4-bit and two-GPU code paths have run once each.


## (c) Verify yourself before using this in any application

Do these in order; each says how. Tick them off here.

**1. The hypothesis-only result (claim 1): AUROC 0.895 [0.863, 0.919] on the official Corr2Cause test split.**
- [ ] With the project's code: `.venv/bin/python scripts/14_benchmark_baselines.py`, then read `corr2cause_official_splits/test/hypothesis_only/auroc` in `results/benchmark_baselines.json`.
- [ ] Without the project's code (a few lines of pandas; it printed 0.895, 1,162 items, 120 hypothesis strings, none unseen, on 15 Nov):

```python
# independent of the repository's code: P(valid | exact hypothesis string) from train, AUROC on test
import pandas as pd
from sklearn.metrics import roc_auc_score
tr = pd.read_csv("data/raw/corr2cause/train.csv"); te = pd.read_csv("data/raw/corr2cause/test.csv")
hyp = lambda d: d["input"].str.split("\nHypothesis: ").str[1]
prior = tr.assign(h=hyp(tr)).groupby("h")["label"].mean()
s = hyp(te).map(prior).fillna(tr["label"].mean())
print(f"hypothesis-only AUROC on the official test split: {roc_auc_score(te['label'], s):.3f}  ({len(te)} items, {hyp(te).nunique()} hypothesis strings, {s.isna().sum()} unseen)")
```

- [ ] That `data/raw/corr2cause/` is the released v2.0 data at the revision named in `report/benchmark_text_baselines.md` (Hugging Face `causal-nlp/corr2cause`, revision `42ba12c769e11ff6427c9f52d7db58e3f9bf3e53`).
- [ ] The interval rests on 21 premises; read it as rough.

**2. The papers cited, section by section.** Open each and check the statement against the page.
- [ ] **Corr2Cause**, Jin et al., ICLR 2024, https://arxiv.org/abs/2306.05836 (v3): Table 3 (207,972 items, 18.57% valid; item counts by number of variables, no premise counts); Table 4 (best off-the-shelf F1 33.38, BART MNLI; always-majority F1 0.0, accuracy 84.77%); Table 5a (fine-tuned RoBERTa-Large MNLI F1 94.74); Table 5b (losses up to 39.29 under paraphrase and 62.30 under variable refactorisation); Sec. 3.3 (graphs generated with nodes in topological order); Sec. 3.7 footnote and Appendix D (the de-duplicated v2.0 release); Appendix E (n-grams tied to variable names correlated with the label); Sec. 4.5 (the perturbation tests). Also check our statement that the paper has no hypothesis-only baseline and does not use the words "shortcut", "artifact" or "hypothesis-only".
- [ ] **CLadder**, Jin et al., NeurIPS 2023, https://arxiv.org/abs/2312.04350: Appendix A.2 (collider-bias questions have a fixed answer in canonical form). `notes/prior_work.md` flags that its Sec. 3.2 describes the fine-tuning split; check it if you cite that.
- [ ] **Clarke, Karn, Murray, Hill, Jensen (2026)**, "Evaluating Causal Reasoning in LLMs: A Case Study in the Challenges of Procedural Generation", https://arxiv.org/abs/2610.38100 (preprint, submitted 28 Oct 2026): 86.8% accuracy for a random forest on surface features under ten-fold cross-validation on 9,866 regenerated CLadder items (Sec. 3.2; Table 3 in App. A.4; features App. A.8; generator rules App. A.7); 62.2% for `backadj` on 74 items; that they do not say whether folds keep stories together. It is a two-week-old preprint: check that it is still up, the author list and the version you cite.
- [ ] **Pacchiardi et al. (2024)**, cited in `report/benchmark_text_baselines.md` for word 2-grams predicting benchmark answers including CLadder: confirm the reference and that CLadder is in it.
- [ ] The standard references (Hewitt and Liang 2019; Poliak et al. 2018; Gururangan et al. 2018; Lieberum et al. 2024, Gemma Scope; Hu et al. 2021, LoRA): bibliographic details only.
- [ ] `notes/prior_work.md` and `notes/literature_check.md` say what was opened and what was only seen in search results; anything marked "not opened" there is not checked.

**3. Every number on the results page.** Each is read from a result file when `scripts/13_build_docs.py` builds the page, so checking means re-running the script that writes the file and reading the key. The GPU ones cannot be rerun on the laptop; for those, `run_all.sh` recomputes the statistics from the saved raw outputs, and the review that recomputed them is named in the page's "Checked by".

| On the page | File → key | Produced by |
|---|---|---|
| 0.895 [0.863, 0.919] | `results/benchmark_baselines.json` → `corr2cause_official_splits/test/hypothesis_only/auroc` | `scripts/14_benchmark_baselines.py` |
| 0.765 [0.743, 0.788] | `results/probe_content_qwen3b_c2c_debiased.json` → `baselines/gold_probe/accuracy` | `scripts/05b_probe_content.py --model qwen3b` |
| 0.765 | `results/probe_content_qwen3b_c2c_debiased.json` → `baselines/B_binding_text_reference/template_x_statements_naming_the_pair/accuracy` | `scripts/05b_probe_content.py --model qwen3b` |
| 0.816 | `results/probe_content_qwen3b_c2c_debiased.json` → `baselines/B_binding_text_reference/hand_parse_all_features/accuracy` | `scripts/05b_probe_content.py --model qwen3b` |
| 0.507 [0.484, 0.530] | `results/answer_bias.json` → `qwen3b/c2c_debiased/auroc` | `scripts/15_answer_bias.py` |
| 0.784 [0.762, 0.806] | `results/probe_content_qwen7b-4bit_c2c_debiased.json` → `baselines/gold_probe/accuracy` | `scripts/05b_probe_content.py --model qwen7b-4bit` |
| 0.514 [0.490, 0.536] | `results/answer_bias.json` → `qwen7b-4bit/c2c_debiased/auroc` | `scripts/15_answer_bias.py` |
| 0.47 | `results/answer_bias.json` → `gemma2b/c2c_debiased/within_hypothesis_string/mean_auroc` | `scripts/15_answer_bias.py` |
| 0.51 | `results/answer_bias.json` → `qwen3b/c2c_debiased/within_hypothesis_string/mean_auroc` | `scripts/15_answer_bias.py` |
| 0.549 [0.530, 0.569] | `results/answer_bias.json` → `qwen7b-4bit/c2c_debiased/within_hypothesis_string/mean_auroc` | `scripts/15_answer_bias.py` |
| 0.53 | `results/answer_bias.json` → `qwen7b-4bit/c2c_debiased/within_hypothesis_string/prompt_length_removed/quadratic_in_log_token_count/mean_auroc` | `scripts/15_answer_bias.py` |
| 53.6% | `results/answer_bias.json` → `qwen7b-4bit/c2c_debiased/within_hypothesis_string/valid_item_above_invalid_item_of_the_same_premise` | `scripts/15_answer_bias.py` |
| 52.0% | `results/answer_bias.json` → `qwen3b/c2c_debiased/within_hypothesis_string/valid_item_above_invalid_item_of_the_same_premise` | `scripts/15_answer_bias.py` |
| 0.542 | `results/answer_bias.json` → `gemma9b/c2c_debiased/within_hypothesis_string/mean_auroc` | `scripts/15_answer_bias.py` |
| 0.52 | `results/answer_bias.json` → `gemma9b/c2c_debiased/within_hypothesis_string/prompt_length_removed/quadratic_in_log_token_count/mean_auroc` | `scripts/15_answer_bias.py` |
| 32 | `results/steering_controls_multiplicity.json` → `summary/all/n_comparisons` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| 4 | `results/steering_controls_multiplicity.json` → `summary/all/n_rank_1_in_absolute_slope` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| 1.52 | `results/steering_controls_multiplicity.json` → `summary/all/expected_rank_1_by_chance` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| 3 | `results/steering_controls_multiplicity.json` → `summary/debiased_pool/n_rank_1_in_absolute_slope` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| 16 | `results/steering_controls_multiplicity.json` → `summary/debiased_pool/n_comparisons` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| 0.76 | `results/steering_controls_multiplicity.json` → `summary/debiased_pool/expected_rank_1_by_chance` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| +0.738 | `results/steering_controls_qwen3b_c2c_debiased_hyp.json` → `real/gold_diffmeans/slope_per_s` | GPU run (`scripts/07c_steering_controls.py`); slope re-derived from `cache/qwen3b/…/steering_controls*.npz` by `07e` |
| -0.251 | `results/steering_controls_qwen1.5b_c2c_debiased_hyp.json` → `real/gold_probe/slope_per_s` | GPU run (`scripts/07c_steering_controls.py`); slope re-derived from `cache/qwen1.5b/…/steering_controls*.npz` by `07e` |
| 0.08 | `results/steering_settle_qwen3b_c2c_debiased_hyp.json` → `primary_test/p_one_sided` | GPU run (`scripts/07d_steering_settle.py`); raw outputs `cache/qwen3b/c2c_debiased/steering_settle_hyp.npz` |
| 0.12 | `results/steering_settle_qwen1.5b_c2c_debiased_hyp.json` → `primary_test/p_one_sided` | GPU run (`scripts/07d_steering_settle.py`); raw outputs `cache/qwen1.5b/c2c_debiased/steering_settle_hyp.npz` |
| +0.475 | `results/steering_controls_gemma2b_c2c_debiased.json` → `real/gold_probe/slope_per_s` | GPU run (`scripts/07c_steering_controls.py`); slope re-derived from `cache/gemma2b/…/steering_controls*.npz` by `07e` |
| +0.865 | `results/steering_controls_qwen0.5b_c2c_probe.json` → `real/gold_diffmeans/slope_per_s` | GPU run (`scripts/07c_steering_controls.py`); slope re-derived from `cache/qwen0.5b/…/steering_controls*.npz` by `07e` |
| 6.44 | `results/steering_controls_multiplicity.json` → `summary/all/mean_rank/gold_diffmeans/mean_abs_rank` | `scripts/07e_steering_multiplicity.py` (from the raw outputs in `cache/<model>/<tag>/steering_controls*.npz`) |
| 86% | `results/benchmark_baselines.json` → `corr2cause_balanced_pool/baselines/hypothesis_only/accuracy` | `scripts/14_benchmark_baselines.py` |


**4. Numbers on the page typed by hand** (not read from a file):
- [ ] "21 premises": `.venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/raw/corr2cause/test.csv')['input'].str.split('\\nHypothesis: ').str[0].nunique())"`
- [ ] "all eight model × benchmark cells" and "four of the eight cells" (`report/tables/audited_verdict.md`; the figure shows four models on one benchmark).
- [ ] "four models, two item pools, two tokens, two kinds of direction" and "20 matched control directions" (`results/steering_controls_multiplicity.json`, `arms`).
- [ ] "500 controls" (`results/steering_settle_*_c2c_debiased_hyp.json`, `primary_test/n_controls`).
- [ ] "first by 3%" (Gemma-2-2B: 0.475 against the largest control 0.462 in `results/steering_controls_gemma2b_c2c_debiased.json`, `families/shuffled_probe/abs_slope_per_s/max`).
- [ ] "11 expected" (the mean of ranks 1 to 21) and "three of four models" (`results/steering_slope_vs_answer_alignment.json`: r about 0.9 in three models, none in Qwen2.5-0.5B).
- [ ] "outside its controls in one of four models" (`report/tables/adjacency_contrast.md`, end-of-hypothesis best-layer rows).
- [ ] "Seventeen reviews" (stale: eighteen are on file; see (a)).

**5. Things a reader will ask about how this was done, which you should be able to show from the history.**
- [ ] The held-out steering tests were fixed before any run: the log entries of 14 Nov 00:55, 01:40 and 02:45 (`git log --format='%h %ad %s' --date=iso -- notes/log.md`) against the runs' start times in `results/round2_status_r3s_*.json`. The sentence for each outcome was recorded at 07:56, during the first run and before any output.
- [ ] What was changed after seeing results is listed in the report's Section 3.4 and in the decision log; read D18 to D37 once.
- [ ] Every reviewer was a fresh agent commissioned by the author, not a person. Say so if you cite the reviews.
- [ ] The upstream issues in `notes/upstream_issues.md` are unfiled; reproduce each before filing.

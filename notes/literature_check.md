# What is already documented about text-only shortcuts in Corr2Cause and CLadder

Written Fri 2026-11-13. Source: a literature search delegated to a separate research agent, which
read the two papers from the arXiv HTML (2306.05836v3, 2312.04350v3), the public code, GitHub
issues and Hugging Face discussions, and scanned citing papers. **Status of each line:** "read" =
the agent read the source directly; "inferred" = derived; "not found" = searched, nothing located.
Lines marked ✔ were re-checked by me against the source afterwards; unmarked lines were not and
must be checked before they are quoted to anyone. OpenReview reviews could not be opened.

## Corr2Cause (Jin et al., ICLR 2024, arXiv:2306.05836)

Construction (paper Sec. 3.3–3.7, App. D; code `data_gen.cpp`, `data_verbalize.py`):
- (read) Graphs are generated with nodes in topological order: only edges X_i → X_j with i < j.
  Isomorphic DAGs are removed. Table 1: 6,325 unique DAGs, 2,376 Markov equivalence classes for
  N = 2..6, of which 2,207 have N = 6.
- (read, code) Node index i is verbalised as the i-th letter (`chr(i + 64)`), reversed alphabet in
  the "refactorization" perturbation. The paper does not state the index-to-letter mapping in words.
- (read) Label = 1 iff the hypothesis holds in every DAG of the equivalence class. Six relations,
  3N(N−1) hypotheses per class.
- (read) v2.0 de-duplication (Sec. 3.7 footnote, App. D): Is-Parent(i, j) and Is-Child(j, i) are the
  same string; one is kept by forcing i < j. v2.0 is half of v1.0; experiments were not rerun.
- (read) Splits (Sec. 3.7, Table 3): N = 2, 3 go to dev/test only; for larger N up to 1K or 10% is
  sampled for dev and test. Sizes 205,734 / 1,076 / 1,162. In the public code the classes are
  shuffled and sliced in contiguous blocks, so splits are by block of classes.
- (not found) Number of distinct premises per split is not stated in the paper. (inferred) Our
  21 / 22 / 2,338 are consistent with Table 3 divided by 3N(N−1), and
  2,338 + 22 + 21 − overlaps = 2,376 = Table 1's class total.

Does the construction imply the hypothesis-string shortcut? (inferred) Yes: the generating DAG has
all edges from earlier to later letter and belongs to its own equivalence class, so a directed
claim can be valid only from an earlier to a later letter; after v2.0 forces i < j, the `child`
template ("j directly causes i") can never be valid. (not found) The paper never states this.

What the paper says about shortcuts:
- (read) Imbalance: 18.57% positive overall (Table 3); always-majority baseline 84.77% accuracy
  (Table 4); F1 is the main metric.
- (not found) No hypothesis-only or partial-input baseline; the words "shortcut", "artifact",
  "hypothesis-only" do not occur.
- (read) App. E, Table 11: PMI between labels and n-grams; some cues are tied to variable names.
  Paraphrase and variable-refactorization perturbations are designed to break these. Sec. 4.5 /
  Table 5(b): fine-tuned models lose up to 39.29 F1 under paraphrase and 62.30 under
  refactorization. The paper frames this as overfitting versus mastering the skill and recommends
  using the data as a test set and pairing fine-tuning with the perturbed sets. It does not
  identify the name-order cue.

Public reports (not peer-reviewed):
- (read) Hugging Face discussion #6 (Apr 2026): claims 6,943 incorrect labels (3.3%; test 18) from a
  bug in `data_gen.cpp`; open. GitHub issues #5–#9 on label correctness and reproducibility.
- (read) A GitHub issue comment (Sep 2026) states `child` has no positive instances; its supporting
  repository returns 404.
- (read) An unreviewed student project README (github.com/abhijeetnst/dl-aml, Oct 2026) states that
  dev shares premises with test (3) and train (2), and that "B directly causes A" is never positive
  because causes are ordered alphabetically. This is the only explicit prior statement of the
  name-order observation and of premise overlap that was found.

Follow-up papers: (not found) any paper with a hypothesis-only baseline, an analysis of the
name-order cue, premise counts per split, or a premise-level re-split of Corr2Cause. Papers that
note the label imbalance: Sgouritsa et al. 2024 (arXiv:2412.13952); Sun et al. 2026
(arXiv:2609.31071). No probing or mechanistic study on Corr2Cause was found.

## CLadder (Jin et al., NeurIPS 2023, arXiv:2312.04350)

- (read) Generation: sample query type, graph and Bernoulli parameters; an engine computes the
  answer; a story template verbalises it. 10,112 items in v1.5, 50% yes overall and per rung
  (Table 1). (not found) No statement that yes/no is balanced within query type.
- (read) No text-only baseline in the paper; only a random baseline.
- (read) `collider_bias`: App. A.2 says the answer is always "no" in canonical form; in the code the
  answer is (−1)^polarity, i.e. set by the "affects"/"does not affect" wording alone.
- (read, code) `backadj`: no probabilities are used; the valid adjustment set is hard-coded per
  graph and the answer depends only on graph family and which method is listed first.
  (not found) The paper does not say this.
- (not found) No changelog for v1 → v1.5 beyond App. A.4 / App. D (aggregate versus balanced set).

Prior work on CLadder surface features:
- (read) **Clarke, Karn, Murray, Hill, Jensen (2026), arXiv:2610.38100 (preprint, 28 Oct 2026).** A
  random forest on surface features (the quoted probabilities, comparisons between them, polarity
  flags) reaches 86.8% under 10-fold cross-validation on a freshly generated 9,866-item set, with
  per-type numbers (collider bias 100.0, backdoor adjustment 62.2 with n = 74). Their
  cross-validation is not story-grouped and the data is regenerated, not `full_v1.5_default.csv`.
- (read) Pacchiardi et al. 2024, arXiv:2410.11672: n-gram predictability study that includes CLadder.
- (read) Probing on CLadder: Ding & Zhang 2026, arXiv:2605.25891 (linear probes ≈ 0.97 on
  anti-commonsense items while answers are ≈ 0.5; shuffled-label control; no text-feature
  baseline seen); Yu 2026, arXiv:2606.15733.
- (read) GitHub issues #15, #16 (Sep 2026): reported ground-truth errors (83 of 10,560 by the
  reporter's count); unanswered.

## Methodological precedents (verified against ACL Anthology)

- Poliak et al., "Hypothesis Only Baselines in Natural Language Inference", *SEM 2018, pp. 180–191.
- Gururangan et al., "Annotation Artifacts in Natural Language Inference Data", NAACL-HLT 2018 (short), pp. 107–112.
- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks", EMNLP-IJCNLP 2019, pp. 2733–2743.
- Kaushik & Lipton, EMNLP 2018; McCoy, Pavlick & Linzen, ACL 2019; Tsuchiya, LREC 2018.

## What we may claim

| Observation | Status |
|---|---|
| Label imbalance; six-variable dominance | Documented in the paper (Tables 1, 3, 4). Cite, do not claim. |
| Directed claims valid only from earlier to later letter; `child` never valid | Follows from the published construction (Sec. 3.3, App. D); not stated in the paper; stated informally in one student README and one issue comment. Say "follows from the construction; we quantify it". |
| A hypothesis-only classifier and its accuracy/AUROC; 120 strings, 72 with a constant label | No prior mention found. Our measurement. |
| Distinct premises per split (21 / 22 / 2,338) | Not stated in the paper; derivable from Table 3 and the code. Our count. |
| Dev–train premise overlap | Not in any paper; in one student README. Our count, with that pointer. |
| CLadder is largely predictable from surface features; `collider_bias` deterministic | Documented by Clarke et al. 2026 (and App. A.2 of the paper for `collider_bias`). Cite; our contribution is only the story-grouped estimate on the released v1.5 file and the comparison with probes. |
| CLadder `backadj` ≈ fully predictable | Clarke et al. report 62.2% (n = 74) for it; ours is higher with an explicit "which method is case-by-case" feature. Follows from the generator code. Report as ours, with their number next to it. |
| A probing study that pairs probes with a text-only baseline on these benchmarks | None found. |

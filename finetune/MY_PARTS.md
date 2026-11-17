# Your parts (Sahil)

Three things in this study are yours. Claude has not written them and will not. The pipeline runs meanwhile on
reference implementations kept in separate files (`*_reference.py`); every output file records which
implementation produced it (`"implementation": "owner"` or `"reference"`), and `finetune/ftstudy/impl.py` switches
to yours the moment your function stops raising `NotImplementedError`.

Rule for (a) and (b): write yours **before** opening the reference file. The tests are the contract; they run on
both implementations.

```bash
../causal-probe/.venv/bin/python -m pytest finetune/tests/test_balance.py finetune/tests/test_within_string.py -v
```

Before you start, each owner test shows as `SKIPPED`; when you are done, none may be skipped and none may fail.

---

## (a) The length-and-template balancing function — `finetune/ftstudy/balance.py`

**Why it exists.** In the official data, inside a hypothesis type, a shorter premise is more often a valid item
(length alone: AUROC 0.26 to 0.30, measured 16 Nov). A model fine-tuned on that can answer from wording and length
without reading the statements. Condition B and the evaluation pool POOL-LB must not allow it.

**Signature.**

```python
def length_bin_edges(lengths: Sequence[int], n_bins: int) -> list[float]
def balance_items(items: list[dict], premise_tokens: Sequence[int], *, n_length_bins: int,
                  max_pairs_per_premise: int, seed: int) -> list[int]
```

`items[i]` has `group` (premise id), `template` (hypothesis type), `query` (hypothesis string), `label` (1 valid,
0 invalid). `premise_tokens[i]` is the token count of item i's premise (the same number for every item of a
premise). Return the indices of the items you keep.

**Length bins (fixed by this spec so that tests can recompute them).** Work one hypothesis type at a time. Consider
only items whose hypothesis string occurs with both labels somewhere in `items`. Take one length per distinct
premise of that type, sort them, and let the edges be `np.quantile(lengths, k / n_bins)` for k = 1 … n_bins − 1. An
item's bin is the number of edges ≤ its premise length (`np.searchsorted(edges, length, side="right")`).
`length_bin_edges` returns those edges (an empty list when `n_bins` is 1).

**What the returned subset must satisfy** (the test names them P1 to P5):

- P1. Inside every hypothesis string: number valid = number invalid.
- P2. Inside every premise: number valid = number invalid.
- P3. Inside every hypothesis type × length bin: number valid = number invalid.
- P4. Inside every hypothesis string × length bin: number valid = number invalid. (This is the one that removes
  the leak; P3 follows from P2 if you build pairs inside a premise, P4 does not.)
- P5. No premise contributes more than `2 × max_pairs_per_premise` items.
- Sorted indices, no repeats, subset of the input; same output for the same seed; the input is not modified; empty
  input or input with no two-label string returns `[]`.
- Not required: the largest possible subset. Required: at least half as many items as the reference keeps on the
  test's synthetic data (so "return nothing" does not pass).

**A hint, only if you want one.** Think of a kept valid/invalid pair from one premise as an arrow from the valid
item's string to the invalid item's string. P4 says: inside each (type, bin), every string has as many arrows
leaving as arriving.

**How to check your answer.**

1. The tests above (`test_balance.py`): properties on synthetic data, determinism, edge cases, and that length no
   longer predicts the label within string after balancing while it did before.
2. On the real data, after your version is in: `finetune/scripts/build_data.py --check-only` (arrives with the data
   build) prints for B and POOL-LB the hypothesis-only, premise-only, length-only and string × length AUROCs with
   intervals. Requirement in PREREG.md: each within 0.5 ± 0.02 with the interval containing 0.5.
3. Compare sizes with the reference (16 Nov, TRAIN premises, 16 bins, 8 pairs per premise: 7,766 items on 836
   premises). If yours keeps more with the same properties, yours is better and the reference is retired.
4. The core runs use the reference function (your decision, 16 Nov). Once yours passes the tests, run
   `caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/build_data.py --check-only`: it rebuilds B and
   POOL-LB with **your** function (the pipeline switches automatically) and compares the hashes with the stored sets.
   **Open point, waiting for your answer:** the reference is a seeded greedy procedure, so an independently designed
   function will pass P1 to P5 and keep a similar number of items but will not pick the same items, and this check
   will report a difference. Two honest ways to report it: (i) "the stored sets pass the owner's independent
   property checks, and the owner's function gives sets with the same properties" (independent design), or (ii) I
   add the exact procedure and order of random draws to this spec and you reproduce the sets bit for bit
   (independent typing, same design). Tell me which.

## (b) Within-hypothesis-string AUROC with length removed — `finetune/ftstudy/within_string.py`

**Why it exists.** The pooled AUROC on a debiased pool can sit at 0.5 while answers track validity inside each
string, and a within-string AUROC can be produced by answering to prompt length alone. This is the primary
statistic of E2, E4 and E5.

**Signature.**

```python
def within_string_auroc_length_removed(labels, scores, strings, n_tokens) -> dict
# {"mean_auroc": float, "per_string": {string: auroc}, "n_strings": int}
```

**Definition.** For each distinct hypothesis string that has at least one valid and one invalid item:

1. x = log(n_tokens) of its items, minus their mean.
2. Fit by ordinary least squares score ≈ a + b·x + c·x² on the items of that string. The labels are not used.
3. Residual = score − fitted value.
4. AUROC of the residual for the label: the probability that a randomly chosen valid item has a higher residual
   than a randomly chosen invalid item, ties counted one half. Write it yourself (rank-sum or pair counting); do not
   call `sklearn.metrics.roc_auc_score`.

`mean_auroc` is the unweighted mean over those strings; `per_string` has one entry per such string; strings with a
single label are skipped and not counted in `n_strings`.

**How to check your answer.**

1. `test_within_string.py`: perfect and inverted scores give 1 and 0; a score that is any quadratic in log tokens
   plus noise gives exactly what the noise alone gives; adding a different quadratic per string changes nothing;
   one-class strings are skipped; all-tied scores give 0.5; and yours equals the reference to 1e-9 on five random
   data sets (`test_owner_equals_reference`).
2. By hand, once: three valid items with residuals 2, 0, −1 and two invalid with 1, −2 → pairs won 4 of 6 → 0.667.
3. Against v1.0: for Qwen2.5-7B on the debiased pool the committed value is about 0.53
   (`results/answer_bias.json` → `qwen7b-4bit/c2c_debiased/within_hypothesis_string/prompt_length_removed/quadratic_in_log_token_count/mean_auroc/est`).
   The behaviour script prints both implementations' value for that model as an integration check.

## (c) The interpretation section of `finetune/RESULTS_FT.md`

Claude writes the claims, numbers, figures, audits, limitations and retractions, and leaves a section headed
`## Interpretation` containing only the line `(Sahil)`. Nothing Claude writes elsewhere on the page will say what
the results mean for whether the model "understands" causation.

**What a good version does** (a checklist, not a text):

- States which rung of the ladder in PREREG.md Section 0 (L0 to L4) each model reached, using only audited tests.
- Says what the result does *not* show (one model family, LoRA, one benchmark, registered effect sizes).
- Names the alternative explanation you find hardest to rule out, and what experiment would.
- Uses no number that is not already on the page.

**How to check it.** For every sentence, point to the claim or table row it rests on. Then ask Claude for an
independent review of the section against the result files (it will commission a fresh reviewer and return the raw
report without editing your text).

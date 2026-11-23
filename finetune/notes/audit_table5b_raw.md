# Audit: Table 5b analogue (`finetune/scripts/table5b.py`, `finetune/results/table5b.{json,md}`)

Independent auditor, 22 Nov 2026. I read the code and the result files only. My own scripts (scratchpad, not
committed) use sklearn `f1_score` and `accuracy_score`, my own seed lists, my own premise resampling with a different
RNG, and a separate re-read of the authors' CSV files. Python: the causal-probe `.venv`, OMP_NUM_THREADS=2.

**Verdict: PASS WITH CONDITIONS** (Section 6). The numbers are correct and match what PREREG registers. The
conditions cover presentation (bounds, outlying seeds, a wrong date, trivial baselines) and the wording beside
the authors' Table 5b.

---

## 1. Does it compute what PREREG registers?

PREREG says (Section 1, set table; Section 4, E4, last sentence): pooled F1 on ID, ID renamed, ID paraphrased and
the authors' two perturbed sets, for every condition, descriptive.

| Item | Registered | Script | Status |
|---|---|---|---|
| Sets | ID, ID renamed, ID paraphrased, authors' refactorisation (1,884 distinct), authors' paraphrasing (1,522 distinct) | the same five, plus official test (1,162) | match; official test is an addition, labelled "for continuity" |
| Conditions | every condition | base, A, B, A48, C, Bswap | match |
| Seeds | A 5, B 5, A48 3, C 3, Bswap 3 | `BH.REGISTERED_SEEDS`, seeds `s0..s{n-1}` | match. `runs/` also holds Bswap_s3 and Bswap_s4. They are correctly **excluded**. |
| Threshold | PREREG does not set a threshold for this table. E1 registers "logit difference > 0" (and a training-set F1-max threshold) | `score > 0` | a reasonable borrowing of E1's threshold. The training-set F1-max threshold is not computed because training-set scores were not stored. This is stated in the docstring and the draft's Deviations. |
| Metric | "pooled F1" | valid-class F1 headline, plus macro-F1 and accuracy | match. Pooled over items within one model, then a mean over seeds. This agrees with Section 3 ("computed per model and then combined"). |
| Interval | Section 3 form | seed term s²/n + premise term from 2,000 premise resamples of the seed-mean statistic, Welch df (= n−1 for one sample), t quantile. Base: 1.96·premise SE | match (Section 4 below) |
| Timing | — | computed after the behaviour audit | the owner says this is a listed deviation |

Deviations and omissions:
- D1. The threshold is E1's, applied by extension. No F1-max threshold exists (documented).
- D2. The official test row is an extra. This is harmless and labelled.
- D3. The authors' sets are deduplicated to distinct inputs (1,884 and 1,522 of 2,246 rows each). PREREG registers
  exactly this, but the authors almost certainly scored all 2,246 rows. Their F1 weights repeated rows; ours does not.
- D4. **Date error.** The JSON (`"computed": "23 Nov 2026 ..."`), the MD header and the docstring all say 23 Nov.
  The files' mtime is 22 Nov 2026 23:42 on this machine's clock. This is a minor inconsistency, but the date
  should match the actual run.
- Nothing registered for this table is missing.

## 2. Independent recomputation

- **Every cell recomputed.** That is 6 sets × 6 conditions × 3 metrics × (mean, min, max) = 324 cell values, plus every
  per-seed value. Max absolute difference from `table5b.json`: **0** (bit-identical; sklearn F1 and macro-F1 agree with
  `BH.f1_valid`).
- **A, ID, valid-class F1 per seed.** Mine: 0.8123, 0.8251, 0.8026, 0.8367, 0.8317. These are identical to
  `behaviour_registered.json` → `descriptive.E1.classification_id_per_seed[*].f1_valid_at_0`. Macro-F1 and accuracy
  also agree (e.g. s0 0.8890 / 0.942).
- **Intervals.** I recomputed five cells with my own premise bootstrap (1,000 resamples, a different RNG):

| Cell | Mine | Table |
|---|---|---|
| id, A | 0.822 [0.774, 0.870] | [0.773, 0.871] |
| id_renamed, B | 0.467 [0.392, 0.541] | [0.390, 0.543] |
| official_test, A48 | 0.713 [0.465, 0.960] | [0.465, 0.960] |
| authors_paraphrasing, C | 0.375 [0.031, 0.719] | [0.038, 0.712] |
| authors_refactorization, base | 0.197 [0.082, 0.313] | [0.082, 0.312] |

  They agree to Monte Carlo error.

## 3. Labels and uid alignment

- Labels come from `DS.materialise("eval_"+k)`, which checks each set's stored sha256 over (context, query, label).
  All six sets rebuilt with a matching hash.
- `__label` in every one of the 19 scores.npz files used (registered runs plus base) equals the materialised gold
  label on every item of every set. There is no mismatch.
- The uids in every npz equal the materialised uid order. The script asserts this, and I asserted it again.
- Authors' sets: I re-read both CSVs independently. Duplicate inputs never carry conflicting labels. Valid = "1"
  (refactorisation is binary; paraphrasing is 3-way, with 1 = entailment and 0 and 2 = not valid). The labels agree
  100% with `materialise`. Valid shares: 0.151 and 0.168.
- Our renaming and paraphrase of the official test reproduce 1,162 of 1,162 items of each authors' file, with
  identical labels. This confirms PREREG's claim.

## 4. Is the interval a valid reading of Section 3?

It is formally the registered form: D = seed mean, SE² = s²/n + v_premise, Welch df, and a t quantile.
`seedtest.t_and_df` gives df = n−1 here. Resampling whole premises and recomputing F1 on each resample
handles the non-additivity correctly. F1 is a ratio, so it must be recomputed per resample rather than averaged
per premise, and it is. Caveats:

1. **21 premises** (authors' sets and official test). Items per premise run from 6–10 up to 90–150. The cluster
   bootstrap variance with G = 21 is noisy, and it is slightly biased low (about (G−1)/G). On these sets the premise
   SE is 0.03–0.08, against 0.016–0.023 on ID. Read these intervals as rough.
2. **Seed term with 3 seeds** (A48, C, Bswap): t(2) ≈ 4.30. That term dominates, so the intervals are very wide.
   A48's ID interval [0.517, 0.875] comes from one seed at 0.774 against 0.652 and 0.661.
3. **Unbounded normal-t intervals on bounded statistics.** Fourteen intervals leave [0, 1]. Examples: Bswap accuracy
   [−0.29, 1.27] on ID, and A48 accuracy on the official test [0.734, 1.029]. These are artefacts of the form.
4. **Outlying seeds (Section 3 rule: > 0.10 from the condition median).** Bswap_s1 predicts "valid" on every item of
   every set: its accuracy equals the valid share and its macro-F1 is about 0.13. This makes the Bswap seeds
   bimodal, and Section 3 itself says the test's size is 0.12–0.32 in that case. A48 is outlying on F1 for ID
   (s2 0.774) and the official test (s2 0.806). Bswap is outlying on F1 for the official test (0.123). The Bswap
   intervals carry no information.
5. **Base model** has one model, so its interval is premise-only (1.96·SE). This is correct, but it is not
   comparable in kind to the fine-tuned rows.
6. Each interval is per set. Several sets share premises (ID, ID renamed and ID paraphrased share the 187 premises;
   the official test and both authors' sets share the 21). Differences between sets are paired, so overlapping or
   non-overlapping intervals are **not** a test of a drop. No drop test is registered (E4 forbids an A-against-B drop
   test, and paraphrase is descriptive only).

## 5. Wording risk beside the authors' Table 5b

Reference values I computed:
- Trivial predictors per set. "All valid" gives valid-F1 0.262–0.288. "All invalid" gives accuracy 0.83–0.85 and
  macro-F1 about 0.46.
- So **base (0.09–0.20), Bswap (0.21–0.28) and Bswap's macro-F1 are at or below the all-valid baseline.** Base
  accuracy on ID (0.789) is below the majority rate (0.846). A48, C and B accuracies (0.74–0.89) are also near or below
  it. Accuracy is close to uninformative at a 15% base rate, and macro-F1 is little better.
- Threshold 0 mixes calibration with discrimination. A's pooled AUROC is 0.980 on ID, 0.945 on ID renamed, 0.951 on
  ID paraphrased, 0.948 on authors' refactorisation, 0.887 on authors' paraphrasing and 0.984 on the official test. A's
  predicted-valid rate also moves (0.152 on ID, 0.123 on ID renamed). Part of each F1 drop is a shift of the logit
  offset. Base's predicted-valid rate swings from 0.03 to 0.21 across sets while its AUROC stays at 0.54–0.59. Base's
  F1 differences across sets are therefore offset, not skill.
- **The authors' sets are not "the perturbed official test".** Each contains all 1,162 perturbed test items plus
  extras on the same 21 premises: 722 (refactorisation) and 360 (paraphrasing). The extras matter. On authors'
  paraphrasing, A's F1 is 0.690 on the 1,162 matched items and **0.079 on the 360 extras** (B: 0.518 and 0.429). On
  refactorisation, A scores 0.749 on matched items and 0.696 on extras. Most of A's drop from 0.844 (official test) to
  0.586 (authors' paraphrasing) comes from the extra items, not from the paraphrase of the test items.
- The authors' sets share their 21 premises (graph classes) with the official test. For our models these are held
  out (PREREG: no training set shares a graph class with any evaluation set). For the authors' models the same
  premises were also unseen in training, but the authors scored all 2,246 rows, with duplicates.

**May be said:**
- On held-out premises, at threshold 0, valid-class F1 of each condition per set, with seed range and interval.
- "A's F1 is lower on ID renamed (0.694) and ID paraphrased (0.656) than on ID (0.822), in every seed". Check the
  per-seed values: A's ID minimum is 0.803 and its renamed maximum is 0.747, so this holds. The wording must be
  descriptive and must say that no test is registered.
- That the direction of the authors' Table 5b (lower F1 under both perturbations) appears for A, B, A48 and C. Add
  that A's pooled AUROC also falls (0.98 → 0.95), so the drop is not only threshold.

**May not be said:**
- That any condition is "robust" or "fragile" in a tested sense, or that one condition drops more than another. No
  test is registered, and baselines and floors differ.
- Any statement about base or Bswap beyond "at or below the all-valid baseline (≈ 0.27)".
- That an F1 on the authors' sets is "the official test perturbed". Name the extra items, or add a row restricted to
  the 1,162 matched items.
- A direct numeric comparison with the authors' Table 5b as like-for-like. The models, training data and
  threshold differ, and the authors counted duplicate rows.
- That the authors' refactorisation drop is due to "names". That is E4b's question, and the reversal of letter
  order is confounded here (PREREG Section 1 (i)).
- Any reading of the accuracy column as performance without the majority rate (≈ 0.85) beside it.

## 6. Conditions

1. **Fix the date.** "23 Nov" in the JSON `computed` field, the MD header and the docstring should match the actual
   run (files written 22 Nov 2026 23:42), or the run should be redated by re-running.
2. **Put the trivial baselines in the MD.** Per set, show all-valid F1 (≈ 0.26–0.29), all-invalid accuracy
   (≈ 0.83–0.85) and all-invalid macro-F1 (≈ 0.46). State that base and Bswap are at or below the all-valid F1.
3. **Bounds.** Clip intervals to [0, 1] when displayed, or mark them as unbounded artefacts. In either case, state
   that the 3-seed intervals use t(2) and that the 21-premise intervals are rough.
4. **Outlying seeds.** Apply Section 3's rule descriptively. Flag Bswap_s1 (it predicts all valid on every set) and
   A48_s2 (F1 on ID and the official test). Report each with and without the flagged seed, or at least name it.
   Do not interpret the Bswap intervals.
5. **Authors' sets composition.** State that each authors' file contains the 1,162 perturbed test items plus 722
   and 360 extra items on the same 21 premises, deduplicated from 2,246 rows. Preferably add a "matched 1,162"
   row for each. Without it, A's paraphrasing drop is mostly the extras (F1 0.08 there).
6. **Wording.** Keep the table explicitly descriptive and the comparison with the authors' Table 5b qualitative
   (direction only), with the "may not be said" list of Section 5. Give pooled AUROC beside F1 for any sentence about
   a drop, so that a threshold shift is not read as lost discrimination.
7. Any sentence built on this table for `finetune/RESULTS_FT.md` is wording. Queue it in
   `finetune/notes/results_ft_queue.md` for the owner (CLAUDE.md, Claims 4).

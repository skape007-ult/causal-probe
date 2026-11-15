<!-- Eighteenth review of the project. Raw report, unedited. -->
# Independent review: prompt length in the debiased-pool probe ladder (05b)

## Summary (answers to 1–5)

1. **The implementation is correct, and I reproduced it exactly for all six models, not only two.** I wrote my own driver around 05b's functions and got the same report-half accuracies for every new reference: length alone 0.5625, lookup+length one model 0.7316, lookup+length stacked 0.7716, tuned stack + length 0.8870. The probe's out-of-fold scores match the cached ones with max |diff| = 0, and every gain matches the JSON to the last digit.
   - Length (characters of premise plus hypothesis) correlates with each model's token count within every hypothesis string at r ≥ 0.99839 (median 0.99958). The value is identical for all models because Gemma's token count is exactly Qwen's minus 21.
   - Length enters through a separate boosting model per template. It uses the same outer and inner folds and seeds as the other components, and no row is scored by a model that saw that row's label.
   - The n-gram C is still chosen from {1, 10, 100} using the stack without length, and it picks 100 for every model. C=1000 does worse on the selection half (0.8698 vs 0.8724), so the top of the grid is not limiting.

2. **Replacing the first version was legitimate.** It was decided after seeing results, but the change makes the reference stronger, so it works against the probe. Against the first version, the probe would have "beaten the lookup" significantly for qwen3b, qwen7b and gemma9b.
   - The first version fell because it overfits. The lookup has only 22–35 distinct feature rows per template, and the default boosting splits those cells on a noisy continuous feature.
   - More data does not close the gap: −2.0, −1.4, −2.4 and −2.8 points at 25%, 50%, 75% and 100% of the training data. Regularised boosting removes it (0.7636 vs 0.7628).
   - None of the 11 alternative constructions I tried does better than 0.7748 on the lookup or 0.8902 on the tuned stack.

3. **Length does not change where any model's probe stands.**
   - From 1.5B to 9B the probe minus lookup+length is −1.9 to +1.2 points, and every interval includes 0. qwen0.5b is −13.9.
   - The probe stays below the hand parse for every model (−3.2 to −6.3 points, every interval excludes 0) and 10–14 points below the tuned stack + length.
   - **Gemma-2-9B's gain over the tuned stack + length is weak evidence of an increment under one point.** I reproduce the reported numbers. But the gain was already there without length (report half +1.04 [+0.32, +1.76]). No single seed exceeds +0.56 on the report half, and the all-items result fails a correction over 6 models × 3 halves.
   - Gemma-2-9B's ladder is computed exactly like the others.

4. **The probe's score does carry information that varies with length**, with signs by template that mirror the label's. The text scores show comparable patterns, so this is plausibly structure (statement counts), not a length shortcut. Length added to the text stack does not absorb the probe's increment.

5. **Verdict:** see the end.

## Files and timing
- At 07:22:47 on 15 Nov I copied the `probe_content_*` JSONs (06:17–07:20), the `confound_*` JSONs, 05b (unchanged since 14 Nov 22:52) and the two tables (14 Nov 23:11). I re-copied both tables at 07:34 after the pipeline regenerated them at 07:33.
- Everything is in `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit19/`:
  - scripts: `common.py`, `text_components.py`, `probe_nest.py`, `analyze.py`, `q4.py`, `lc.py`
  - logs: `text_analysis.log`, `analysis.log`, `probe_nest.log`
- Peak memory was ≤ 618 MB, all runs used one thread, and nothing was written to the repository.
- **Stale table, now fixed:** the 23:11 version of `debiased_ladder.md` labelled the first version (0.732) as "Lookup stacked with prompt length" for five models. The 07:33 version is correct. The incremental table did not change.

## 1. Implementation details
- **Nesting and leakage:** outer 5 block-wise folds × inner 4 folds (seed 100+k), 3 seeds. The stacking logistic regression is fitted on inner out-of-fold scores and applied to outer test rows. I found no leakage.
- **C selection:** the n-gram C uses selection-half accuracy only. Those selection-half scores were trained partly on report-half labels from other folds, but report-half rows are never part of the choice.
- **Accuracy by n-gram C, stack without length:**

| C | Selection half | Report half |
|---|---|---|
| 1 | 0.8576 | 0.8654 |
| 10 | 0.8655 | 0.8806 |
| 100 (chosen) | 0.8724 | 0.8870 |
| 1000 | 0.8698 | 0.891 |

- **Tuned stack vs tuned stack + length:** their predictions differ on 6 of 1,248 report-half items, so both score 0.8870.
- **Gemma-2-9B's probe selection** works like every other model's: the best of 43 layers × 3 values of C on the selection half (layer 23, C=1, selection accuracy 0.7685). The only difference is that it has no random-init cache, which also holds for qwen7b.

## 2. Why the first version fell, and alternative constructions
Per template on the report half, the first version loses 4–7 points in the four templates other than "parent":

| Template | Lookup | First version (one model) | Stacked version |
|---|---|---|---|
| non-parent ancestor | 0.714 | 0.645 | 0.714 |
| has_collider | 0.723 | 0.677 | 0.727 |

The first version's |logits| are larger, and its pooled AUROC is higher (0.847 vs 0.834). That pattern is overconfident cell splitting, not a lack of signal.

Text-only alternatives, report-half accuracy (selection half in brackets):

| Construction | Accuracy |
|---|---|
| Lookup, default boosting | 0.7652 |
| Lookup, regularised boosting | 0.7628 |
| Lookup + length, one regularised model | 0.7636 |
| Lookup + string-centred length, one regularised model | 0.7668 |
| Lookup + length, stacked (adopted) | 0.7716 (0.7561) |
| Lookup + per-string length logistic, stacked | 0.7716 |
| Lookup + centred-length logistic, stacked | 0.7644 |
| Lookup + three length models, stacked (best found) | 0.7748 (0.7526) |
| Length alone: per template / per-string logistic / centred per template | 0.5625 / 0.5737 / 0.535–0.548 |
| Tuned stack + length (adopted) | 0.8870 |
| Tuned stack + three length models | 0.8902 (0.8715) |
| Tuned stack (C=1000) + length | 0.8902 |

- **No construction makes length matter much:** at most about +1 point on the lookup and +0.3 points on the tuned stack.
- **The stacked version is a fair reference.** It is within 0.3 points of the best alternative, and the probe's position is the same under every alternative.

## 3. Probe minus each reference (report half, points, 95% premise bootstrap)

| Model | Probe acc | − lookup | − lookup+len (stacked) | − hand parse | − tuned | − tuned+len |
|---|---|---|---|---|---|---|
| qwen0.5b | 0.633 | −13.2 [−16.8, −9.8] | −13.9 [−17.3, −10.5] | −18.3 [−21.6, −15.1] | −25.4 | −25.4 [−28.1, −22.4] |
| qwen1.5b | 0.752 | −1.3 [−4.5, +1.6] | −1.9 [−5.2, +0.9] | −6.3 [−9.5, −3.6] | −13.5 | −13.5 [−16.2, −11.0] |
| gemma2b | 0.763 | −0.2 [−3.5, +3.0] | −0.9 [−3.9, +2.2] | −5.3 [−8.2, −2.5] | −12.4 | −12.4 [−14.9, −9.7] |
| qwen3b | 0.765 | 0.0 [−3.0, +2.9] | −0.6 [−3.5, +2.2] | −5.1 [−8.1, −2.2] | −12.2 | −12.2 [−14.8, −9.7] |
| qwen7b-4bit | 0.784 | +1.8 [−1.1, +4.9] | +1.2 [−1.7, +4.2] | −3.2 [−6.3, −0.5] | −10.3 | −10.3 [−12.8, −7.9] |
| gemma9b | 0.771 | +0.6 [−2.6, +3.7] | −0.1 [−3.0, +3.0] | −4.5 [−7.4, −1.6] | −11.6 | −11.6 [−14.0, −9.1] |

Gain from adding the probe (points). "Report" and "selection" are halves; seeds are the report-half gain of each single seed; p is one-sided bootstrap with 10,000 resamples.

| Model | Base | Report | Selection | All items | Report, single seeds | p report / all |
|---|---|---|---|---|---|---|
| qwen0.5b | tuned | 0.00 [−0.40, +0.48] | 0.00 | 0.00 | — | — |
| qwen0.5b | tuned+len | −0.08 [−0.56, +0.48] | 0.00 [−0.35, +0.35] | −0.04 [−0.33, +0.25] | — | — |
| qwen1.5b | tuned | +0.40 [−0.48, +1.28] | −0.52 | −0.04 | — | — |
| qwen1.5b | tuned+len | +0.48 [−0.32, +1.36] | −0.17 [−1.13, +0.69] | +0.17 [−0.46, +0.75] | — | — |
| gemma2b | tuned | +0.80 [+0.16, +1.52] | −0.17 | +0.33 | 0.00 / +0.24 / −0.08 | .020 / .12 |
| gemma2b | tuned+len | +0.80 [0.00, +1.60] | +0.35 [−0.43, +1.13] | +0.58 [0.00, +1.17] | — | .028 / .022 |
| qwen3b | tuned | +0.24 [−0.56, +1.12] | −0.52 | −0.12 | — | — |
| qwen3b | tuned+len | 0.00 [−0.80, +0.88] | −0.26 [−1.04, +0.43] | −0.13 [−0.67, +0.42] | — | — |
| qwen7b-4bit | tuned | +0.80 [0.00, +1.76] | −0.43 | +0.21 | +0.88 / +1.04 / +0.40 | .048 / .29 |
| qwen7b-4bit | tuned+len | +0.72 [−0.16, +1.68] | +0.26 [−0.61, +1.04] | +0.50 [−0.12, +1.12] | — | .076 / .065 |
| gemma9b | tuned | +1.04 [+0.32, +1.76] | +0.43 [−0.35, +1.48] | +0.75 [+0.21, +1.33] | +0.24 / +0.56 / +0.08 | .0036 / .0054 |
| gemma9b | tuned+len | +0.80 [−0.08, +1.68] | +0.95 [+0.09, +1.91] | +0.87 [+0.25, +1.50] | +0.24 / 0.00 / +0.24 | .047 / .0029 |

**Gemma-2-9B in detail:**
- **The reported numbers reproduce exactly.** Length does not create the gain: without length, the report-half gain is larger (+1.04) and the selection-half gain smaller (+0.43).
- **The estimate moves with the base specification.** Across five reasonable bases the gain spans:
  - report half: +0.48 to +1.04
  - selection half: +0.43 to +1.13
  - all items: +0.67 to +0.88
- **What the gain is not:** a pure-noise fourth stacking component gains 0.06 ± 0.15 points (max 0.32), so this is not a stacking artifact.
- **The 3-seed average exceeds every single seed's gain.** So the effect depends on averaging scores across seeds, and the premise bootstrap ignores that training variability.
- **Multiplicity:**
  - Bonferroni over 6 models: the all-items gain over tuned+len is [+0.04, +1.75]; the report-half gain over tuned is [+0.08, +2.08].
  - Over 18 statistics (6 models × 3 halves): [−0.04, +1.83].
  - The report-half gain over tuned+len (p = .047) fails any correction.
- **The selection-half and all-items figures are optimistic for the probe.** They include the half on which the probe's layer and C were the best of 129 configurations.
- **The other models:** gemma2b and qwen7b have isolated intervals that touch or just exclude 0, and none hold across halves.

## 4. Length inside the activations
Within string, by template:

| Score | parent | non-parent ancestor | has_confounder |
|---|---|---|---|
| Label vs length (r) | +0.16 | −0.10 | −0.12 |
| Probe score vs length (r), across models | +0.20 to +0.36 | −0.18 to −0.30 | −0.12 to −0.29 |
| Hand parse, partial r controlling for label | +0.09 | −0.26 | −0.24 |
| Lookup, partial r controlling for label | — | — | −0.31 |
| 4-grams, partial r controlling for label (parent) | +0.36 | — | — |

- The probe's correlations persist after partialling out the label.
- Regressing length out of the score within string costs the probe 1.3 to 4.0 points, depending on the model. The same operation costs the lookup 2.4, the 4-grams 1.8 and the hand parse 0.6 points.
- So the probe's length dependence is comparable to that of the text scores, consistent with premise structure that varies with length.
- Gains over the stack with length match or exceed those without, so length is not the channel of any probe increment.

## 5. Verdict
After adding prompt length, the claim still holds as worded:
- From 1.5B to 9B (7B included), the probe is statistically level with the lookup, with or without length (every interval includes 0, point estimates from −1.9 to +1.8).
- It is less accurate than the hand-written parse for every model, by 3–6 points.
- It is 10–14 points below the tuned text stack.

For Gemma-2-9B, it is fair to say that adding its probe to the tuned text stack (with or without length) gives a small increment of about +0.8 to +1.0 points, likely under 2 points, at about +0.9 on all items. That evidence is fragile: it is not present in single seeds, it falls out of significance under a correction over models and halves, and the all-items figure includes the probe's selection half. It should not be called a demonstrated gain.

## Could not check
- **Old JSONs:** I could not see the first-version (21:50) JSONs. I judged the replacement from the code comment and from recomputing both versions.
- **Token counts:** I checked n_tokens only as stored in the cache metadata; I did not re-tokenise.
- **Block-level resampling:** I did not run a bootstrap or permutation over the 25 blocks. The premise bootstrap ignores block structure and training variability, so the intervals are probably too narrow.
- **Mechanism of the length dependence:** I did not test whether the probe's length dependence comes from position-dependent directions in the residual stream, which would need activations at other positions.


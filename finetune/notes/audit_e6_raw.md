# Audit of E6 (PREREG Amendment 2): raw report

Auditor: independent reviewer. I was given the code and result files, not conclusions. I edited no file except this one.
Scratch scripts (not part of the repo) are in the session scratchpad: recompute.py, logit.py, extra.py, boot.py, selcurve.py.

Audited: `finetune/scripts/e6_analysis.py` (committed in 0310332 at 21:46 -0500, before the activations were downloaded at 22:48 and unchanged since), `e6_extract.py` and `make_e6_notebook.py` (as run: commit a6a2f65), the imported code, `finetune/e6/data/*`, `finetune/runs/*/scores.npz`, `finetune/results/e6_descriptive.{json,_tables.md}`, `finetune/notes/e6_analysis.log`, and the Kaggle STATUS and log.

## Verdict: PASS WITH CONDITIONS

The numbers are computed as specified, with no leakage, and reproduce independently. The data are sound, and the FAILED status comes from a bug in the notebook's packing cell. The conditions concern (i) one item missing against Amendment 2 and (ii) wording: as computed, "probe − own answer" does not measure "information present and not used". In every condition the probe is below a text classifier trained on the same selection-half labels.

---

## 1. Implementation against Amendment 2

| Amendment 2 item | Implemented | Status |
|---|---|---|
| 1 Models: base, A_s0–4, B_s0–4 | `RUNS` | as written |
| 2 Activations, integrity: largest diff recorded; > 0.05 → not used, reason reported | `load_run` computes the max diff against `scores.npz`; > 0.05 **raises and aborts the whole analysis** | deviation (moot: every run has diff 0.0) |
| 3 Halves: `data.selection_mask` on POOL-LB; layers and C chosen on the selection half | yes (POOL-LB items have no `block`, so the split is by `group` = premise) | as written |
| 4 Probe: `probes.fit_predict` logic, C ∈ (0.01, 0.1, 1) by premise-grouped inner CV on the selection half, trained on the selection half, scored on the report half | `probe_cell` calls `probes._fit` directly, with a scaler fitted on the selection half (same as `fit_predict`; done this way so that C can be read). C is chosen by **accuracy** (v1.0's `scoring="accuracy"`). | as written (v1.0 behaviour) |
| 4 Statistic: S on the report half, plus v1.0 accuracy | `BH.Plan.S` (asserted equal to `stat.S` for the references), accuracy at decision > 0 | as written |
| 5 Reference rows: own answer; E5c registered (on the training set); the same families refitted on the selection half, with the stack tuned inside it; E5b memorisation | all present; the selection-half stack is tuned by 5-fold premise-grouped CV inside the selection half (C = 100 chosen) | as written |
| 6a S by layer: seed mean, range, premise-bootstrap 95% CI, seeds pooled | yes; the **curves use 500 resamples**, the summaries 2,000 (Section 3 says 2,000) | minor deviation, disclosed in the docstring |
| 6b first layer with CI > 0.5; first layer reaching 90% of max seed-mean S | yes; an extra, unregistered criterion is added ("90% of max excess over 0.5") | addition; it must be labelled as unregistered |
| 6c at the layer chosen on the selection half, S(probe) − S(answer), paired interval, **per model and condition** | per condition only. Per-model D is stored (`runs.*.S_minus_answer_at_chosen`), but **with no per-model interval** (only base, which is a condition, has one) | **deviation** |
| 6c the layer is the "largest selection-half S" | per run: S of 3-fold premise-grouped out-of-fold scores inside the selection half, at the C the inner CV chose. The inner-CV folds and these OOF folds are **the same split** (StratifiedGroupKFold(3, shuffle, seed 0) on the same y and groups), so the C choice and the layer choice share their folds: a slight optimism, confined to the selection half | implementation detail, disclosed; no leakage |
| 6d S_cells at the chosen layer; correlation with own answer within string | `Plan.S_cells` (= `stat.S_cells`); the correlation is **Spearman** on raw scores (not length-removed), over strings with ≥ 5 items, weighted by string size | unregistered detail; it should be stated |
| Multiplicity / descriptive | no test, no p | as written |

Other remarks
- The probe grid tops out at C = 1. C = 1 is chosen for every base layer and for A's early layers (grid edge). With `max_iter=300` lbfgs does **not converge** at C = 1 (base final L25: 300 of 300 iterations; it converges at 739). The stored value depends on LogisticRegressionCV's warm-start path. My from-zero refit gives 0.752 at 300 iterations and 0.763 at convergence, against 0.762 in the JSON. Cells with C = 1 therefore reproduce to about ±0.01; cells with C = 0.1 or 0.01 reproduce exactly. This is v1.0's registered setting, so it is a caveat, not a deviation.
- Per-seed chosen layers vary (A final 18–21, B final 20–27). B_s4's layer 27 is the top of a plateau that is flat to 0.001 (selection-half S 0.9005, 0.9000, 0.8998, ... over layers 27, 28, 25, ...). This is harmless for (c), but "the chosen layer" is not a meaningful location.

## 2. Leakage

None found.
- **Halves:** 596 selection premises and 597 report premises; **0 premises and 0 premise texts in both halves**. Of 48 strings, 47 occur in both halves (expected when the split is by premise). Labels are 0.50/0.50 in each half.
- **Probe:** the scaler and the logistic model are fitted on `X[sel]` only. Inner-CV groups are `groups[sel]` (premise).
- **C:** LogisticRegressionCV runs on the selection half only.
- **Layer:** `argmax(Ssel)`, where Ssel is computed with `e_sel` (a Plan on selection items) on out-of-fold selection scores. No report label is used. I recomputed the selection-half OOF curve independently for A_s0 final and B_s4 final: it matches the JSON to 0.0 and gives the same argmax (18 and 27).
- **sel_folds:** `grouped_folds(sel_items, 3, 0)` uses StratifiedGroupKFold on premises, with 0 shared premises in every fold (checked).
- **Selection-half references:** `TR.Side(sel_items)` is used for training and the report half only for scoring. The stack's C is chosen by OOF inside the selection half.
- **Registered references:** fitted on training sets, with vectors stored in POOL-LB order (the order is asserted in text_references.py; I also checked my recomputed S against them).
- **Length removal:** token counts come from the base scores file. No labels are used.

## 3. Independent recomputation

I wrote my own S (within-string quadratic log-length residual, pairwise AUROC with ties 0.5, strings with ≥ 3 per class) and S_cells, plus my own probe (StandardScaler, then LogisticRegression; C picked by my own grouped 3-fold accuracy loop).

| Quantity | Mine | JSON |
|---|---|---|
| answer S, base / A_s0 / B_s0 | 0.5082 / 0.7771 / 0.8989 | 0.5082 / 0.7771 / 0.8989 |
| probe S, A_s0 final L18 (C 0.1) | 0.8600 | 0.8600 |
| probe S, A_s0 content_end L18 | 0.8633 | 0.8633 |
| probe S, B_s0 final L21 (C 0.01) | 0.9178 | 0.9177 |
| probe S, B_s0 content_end L18 | 0.9136 | 0.9136 |
| probe S, A_s0 final L10 (C 1) | 0.6590 | 0.6658 |
| probe S, base final L25 (C 1) | 0.7522 (0.7629 converged) | 0.7619 |
| probe S, base content_end L16 (C 1) | 0.8354 | 0.8337 |
| chosen C, all cells checked | identical | — |
| S_cells, A_s0 final L18 / B_s0 final L21 | 0.8650 / 0.9003 | 0.8650 / 0.9003 |
| ρ(probe, answer), A_s0 final / B_s0 final | 0.611 / 0.872 | 0.610 / 0.872 |
| ref E5c_registered\|A\|tuned_stack | 0.8435 | 0.8435 |
| ref E5c_registered\|B\|hand_parse | 0.8897 | 0.8897 |
| ref E5c_registered\|A\|premise_counts_boosting | 0.5561 | 0.5561 |
| ref selection_half_fit\|lookup (my own refit) | 0.7627 | 0.7627 |
| ref selection_half_fit\|hand_parse (my refit, probability scale, unclipped) | 0.8781 | 0.8807 (the difference is the logit clip's ties) |
| integrity: max \|meta answer − stored score\|, all 11 runs | 0.0 (bit-identical, float32) | 0.0 |

Further integrity checks, all passed:
- I recomputed the answer from the stored activations: RMSNorm(resid[:, 28, final]) · tied unembedding rows of the Yes/No variants, using the local Qwen2.5-1.5B-Instruct weights. Against the meta answers this gives correlation 0.99999 and max abs error 0.02–0.03 (float16 storage) for base, A_s0, A_s3, B_s0 and B_s4. So the activations come from the same forward pass that produced the answers.
- The adapter sha256 in each meta matches `finetune/e6/adapter_sha256.json`.
- Runs differ from one another (A_s0 vs A_s1 answers differ by up to 10.1).
- meta.json inside the Kaggle zips is byte-identical to `e6/data` (checked for A_s1 and B_s3).
- uids and token counts match POOL-LB order.

## 4. The interval (`ci`)

- With ≥ 2 seeds: D ± t₀.₉₇₅(df) · √(s²/n + v). Here v is the variance, over premise resamples (models fixed), of the seed mean (or of the seed-mean paired difference), and df is the Welch–Satterthwaite df of the seed term alone (= n − 1 = 4). This is exactly `seedtest.test`'s interval, so it is a valid reading of "seeds pooled as in Section 3". The paired differences in (c) and in "probe minus reference" use per-seed differences for s² and resampled differences for v (correct pairing; the reference is fixed across seeds).
- With one model (base, references): D ± 1.96 √v. This is reasonable; Section 3 has no one-model case.
- Centring on the full-sample point rather than taking percentiles is justified. Resampling premises duplicates items and biases S downward: my bootstrap mean of S(base probe, content_end) is 0.827, against the point value 0.835. The percentile interval for the A tuned-stack reference is [0.804, 0.855], against [0.818, 0.869] from `ci`.
- Internal consistency:
  - A final: S_probe_at_chosen 0.8617 is the mean of the five per-run values. Probe − answer 0.0990 = 0.8617 − 0.7628. The per-seed differences (0.083 to 0.114, SD 0.0136) and the CI half-width 0.029 imply a premise SE of 0.009. That is plausible.
  - My independent bootstrap of base content_end probe − answer (400 resamples, my probe vector) gives a premise SE of 0.0186, half-width 0.036, against 0.039 in the JSON.
  - A tuned_stack reference: my half-width 0.026, against 0.025.
- The layer-0 cells are exactly 0.5 with CI [0.5, 0.5], because the embedding at the final and content_end tokens is the same token for every item, so the probe is constant. This also produces the logged divide-by-zero warning (√0 in `t_and_df`). The result is harmless (half-width 0).
- The 500-resample curves are a minor departure from Section 3's 2,000 (Monte Carlo error on the half-width is about 3%).

## 5. STATUS.txt = FAILED

**This reflects the notebook's final check only, not the data.** Evidence:
- The run is commit a6a2f65 (first line of the Kaggle log). In that commit the PACK cell sets only `not_finished = ""`. `FINAL_CHECK` (`scripts/notebook_cells.py`, l.175–178) reads `skipped`, which gives `NameError: name 'skipped' is not defined`, the exact text in STATUS.txt. Commit c6a59cd changes only that line (`not_finished, skipped = "", ""`); e6_extract.py is unchanged.
- The log shows all 11 extractions "done", then "failed: none", "all models finished", and 11 archives of 838.8 MB each (9.2 GB).
- Section 3's checks above all pass:
  - every resid has shape [4706, 29, 2, 1536] and the zip metas match;
  - the adapter hashes match;
  - the answers are bit-identical to the stored scores;
  - the final-layer activations reproduce the answers through the unembedding.
- Remaining procedural point: `finetune/kaggle/HALT` is still present (untracked). The run's "the ingest will refuse this run" was bypassed by a manual ingest. That should be recorded as an owner-approved exception.

## 6. What would make the headline sentences misleading

1. **The probe and the answer are not trained on the same thing.** The probe is a linear classifier fitted on 2,346 *POOL-LB* items with gold labels: same distribution, same 47 strings, balanced. The model's answer comes from fine-tuning on a different training set (A), or from no task training at all (base). "Probe − own answer" therefore mixes "information in the representation the model does not use" with "a supervised readout fitted in distribution beats a model trained out of distribution". The base row shows this clearly: probe 0.762 / 0.834 against an answer of 0.508, a gap of +0.25 / +0.33. Base has no task training. Its gap is what 2,346 labels buy from a representation of the text.
   **The fair comparator is the selection-half text references, fitted on the same labels, and the probe is below them everywhere:**

   | Condition, position | Probe | Selection-half hand_parse | Selection-half tuned_stack |
   |---|---|---|---|
   | base, final / content_end | 0.762 / 0.834 | 0.881 | 0.938 |
   | A, final | 0.862 | 0.881 | 0.938 (D −0.076 [−0.104, −0.048]) |
   | A, content_end | 0.880 | 0.881 | 0.938 (D −0.058 [−0.089, −0.026]) |
   | B, final | 0.917 | 0.881 | 0.938 (D −0.020 [−0.048, +0.007]) |
   | B, content_end | 0.915 | 0.881 | 0.938 (D −0.023 [−0.049, +0.003]) |

   A sentence like "the probe exceeds the model's own answers by 0.10 (A)" is true as arithmetic. It does not show unused *task* information beyond what a text parser fitted on the same 2,346 labels extracts.
2. **"Decodable from layer X".**
   - The registered criterion (first CI > 0.5) gives **layer 1** for every condition and both positions. Layer 0 is 0.5 by construction (a constant token).
   - At layer 1, S is about 0.54 at the final position and 0.62 at content_end. This is an early, nearly lexical representation. A global TF-IDF 1–2-gram logistic model of premise plus hypothesis, fitted on the selection half, scores S = 0.51–0.55 on the report half.
   - So "gold label decodable from layer 1" only means that a linear readout of early text features beats chance after length removal at this sample size. It says nothing about the model computing the label.
   - "90% of max S" is weak on a scale whose floor is 0.5: 0.9 × 0.86 = 0.77, which is reached at content_end by layer 8–9. The unregistered "90% of excess" variant is more meaningful, but it must be labelled as unregistered.
3. **Loading on the model's own answer.**
   - In B the probe largely *is* the answer: ρ = 0.86–0.91 within string. Partialling B_s0's answer out of its probe within string (linear) drops the probe's S from 0.918 to 0.590 (final) and from 0.914 to 0.607 (content_end). Probe − answer is +0.013 [+0.002, +0.024] (final) and +0.010 [−0.005, +0.025] (content_end).
   - In A, ρ is about 0.5. The probe keeps S = 0.73–0.74 after partialling the answer, and the answer partialled on the probe falls to about 0.49–0.50.
   - Base: ρ is about 0, because base's answer carries nothing.
   - Any B sentence should say "indistinguishable from / at most 0.01–0.02 above its own answers, and mostly the same signal".
4. **Prompt length.**
   - S removes the quadratic in log length within string, for the probe and for the answers alike.
   - The probe scores barely track length within string (Spearman −0.05 to +0.01).
   - The answers do: +0.29 for base, −0.24 for A_s0, about 0 for B_s0.
   - This is handled by S, but the v1.0 accuracy column beside it is not length-removed and should not be compared with S.
5. **Adjacency.** S_cells (0.83–0.91) is close to S. But adjacency alone scores S = 0.501 on the report half (0.498 on the full pool), so in this balanced pool the cell control cannot reveal adjacency loading. It is not evidence that the probe "does not use adjacency".
6. **Grid edge and convergence.** Base and early-layer probes sit at C = 1, the top of the grid, and are not converged at 300 iterations. Base's S values, and hence base's +0.25/+0.33, are reproducible to about 0.01. A larger C might give a stronger base probe, so the base curve is a lower bound for "what a linear probe can read".

## Conditions

1. Either add per-model (each A_s*, B_s*) paired 95% intervals for item (c), or state in the write-up that only per-condition intervals were computed (deviation from 6c).
2. Never phrase "probe − own answer" as "information present and not used" without, in the same sentence or table, the selection-half text references and the probe-minus-selection-half-tuned-stack D: A −0.076 / −0.058, B −0.020 / −0.023. Say that the probe is a readout trained on 2,346 in-distribution gold labels. Draw no such conclusion from the base row.
3. For (b), report "first CI > 0.5 at layer 1, layer 0 is constant by construction". Do not say the label "becomes decodable" at layer 1 in any computational sense. Label the "90% of excess" criterion as unregistered, and note that the registered "90% of max" is reached early because S has a 0.5 floor.
4. For B, report ρ(probe, answer) of about 0.9 next to D, and say the content_end interval includes 0. Claim nothing for B beyond "the probe is at most about 0.01–0.02 above its answers".
5. Disclose the analysis details that Amendment 2 left open, as the code fixed them:
   - Spearman, raw scores, strings with ≥ 5 items, size-weighted, for 6d;
   - the OOF-based layer choice, sharing folds with the C choice;
   - 500 resamples for the curves;
   - accuracy as the C criterion;
   - the C = 1 non-convergence (cells with C = 1 reproduce to about ±0.01).
6. Record in the run log that STATUS FAILED came from the packing-cell `NameError` (commit a6a2f65), together with the data checks above. Also record that the ingest past the FAILED status and HALT was an owner-approved exception.
7. Do not cite S_cells as evidence about adjacency (adjacency alone is 0.50 on this pool).

<!-- Seventeenth review of the project (the reviewer numbered its scratch folder 18). Raw report, unedited. -->
# Audit 18: Qwen2.5-3B settle run (`r3s_qwen3b`) and all-pairs patching (`r3p_qwen3b`)

## Summary (≤300 words)

Both runs are clean. I recomputed every number in the JSONs and both table rows from the raw npz files and got the same values (max |diff| 0 over 352 settle numbers and all patching statistics). Nothing in the ingest is abnormal.

**Findings by severity**
- **F1 (major, wording): settle.** The registered outcome is **"p > 0.05"** (p = 41/501 = 0.0818). It is firm against item resampling (0 of 5000 premise bootstraps reach 0.05) and against the control draw (0.05 is 2.6 Monte Carlo SE away). Its evidential weight at the observed value is weak, though. A fixed linear response predicts +0.619 ± 0.216 from the cosine alone (0.839 × 0.738), and the observed value is +0.569. The likelihood ratio at that value is about 1.3–2 against an effect centred on the first slope, about 1 against a half-size effect, and only about 5 under the tenth audit's larger effect model. Split 0 against its own 100 premise-balanced controls is already only rank 7/101 (p 0.069).
- **F2 (major, interpretation): patching.** The all-pairs interval contains the 60 discovery pairs. On the 150/171 **new** pairs alone, the hyp difference is **−0.050 [−0.585, +0.464]**. The earlier +0.970 did not replicate, which is what selection predicts. |D| responds to a label shift only to second order, so the lower bound −0.22 excludes only label shifts of about ≥1.0 logit (hyp) or ≥1.2 (prem). That is about half of mean |D|, not 10%.
- **F3–F7 (minor):** see below. They cover how strongly label-preserving |D| depends on the shared label, the independent bootstrap over pair sets that share premises, the isotropic column, the unregistered within-string p = 0.0495, and the gold direction being largely an adjacency direction (cos 0.81).

**Sentences a results page could carry**
- Settle: "In the held-out test fixed beforehand, the Qwen2.5-3B gold difference-of-means direction again had a positive slope (+0.569 per s, against +0.738 first). That is what the overlap of the two fitted directions predicts by itself. It did not stand out from 500 matched controls (one-sided p = 0.082; z +1.34)."
- Patching: "On all available minimal pairs, label-changing and label-preserving pairs differ by the same amount within hypothesis type (+0.24 [−0.22, +0.74] logits for hypothesis edits; −0.05 on the pairs not in the earlier 60-pair run). This excludes only label effects of about one logit or more."

---

## A. Ingest and logs

**Recomputed and found correct.**
- **Byte-identical to the downloads (`cmp`).** This holds for all 17 ingested `results/` files of both runs: the settle JSON, both patching JSONs, the status, environment and output_status files, and all eight stage logs. It also holds for `steering_settle_hyp.npz`, `steering_settle_hyp_plan.npz`, `patching_n240.npz` and `patching_same_n240.npz`.
  - `meta_kaggle.json` and `readout_kaggle.npz` equal the download's `meta.json` and `readout.npz`.
  - The laptop's `meta.json` and `readout.npz` (dated 00:11) were not overwritten. The laptop `readout.npz` is byte-identical to the Kaggle one anyway, and `logit_diff` and `labels` are identical. The two meta files differ only in `positions`, `shape`, `pos` and `resid_exported` (export without activations).
- **`git diff --name-status b8114df~1 b8114df`.** 20 files were added. Four were modified: `notes/log.md`, `report/report.md` and two tables. **No pre-existing result file changed.**
- **Download `results/` versus the repository.** Six files differ: `answer_bias.json`, `steering_controls_multiplicity.json` and four `probe_content_*` files. All six are the 988941b versions; the repository holds later laptop updates. None was copied.
  - Cosmetic: the ingest report lists only three of them under "aggregates not taken". The four `probe_content_*` files are not mentioned.
- **Code.** Both runs ran at 988941b, and there is no difference between 988941b and HEAD in `scripts/07d`, `08`, `08b`, `gpu_round2.py` or `src/`. The 20-control result file is identical in both downloads and the repository.
- **Settle log.** No warnings. All probe fits converged (shares 1.0). `configurations_dropped_for_time` is empty, nothing was shortened, and all six configurations ran (54.8 of 100 minutes). No NaN or infinity appears in any output array.
  - Baseline versus cache: maximum |difference| 0.106–0.153, mean 0.03, correlation 0.99998, on answers with SD 7.2.
  - Split 0's baseline column is bit-identical to the 20-control run, whose value was also 0.1379.
- **Patching logs.** No warnings. Stages took 62.5 and 53.3 of 175 minutes.
  - n = 210 and 231 fall below the 240 cap.
  - The NaNs in the grids are exactly the empty `premise_rest` region for hyp. Donor-control arrays are all NaN, as intended (`--no-donor-control`, and "same" mode).
- **The laptop activations stand in for the GPU run's almost exactly.** I rebuilt all directions of all six configurations with the script's own functions, run on `resid.npy`. Every saved direction (stored as float16) matches its rebuild with cosine ≥ 0.99976; all but the model-answer probes are at 0.99999 or above. The match is one-to-one: the largest cosine with any *other* control is 0.66. `s_unit` agrees to 1e-6.

## B. Settle run

**All six configurations match the JSON and the log.** Column pairing checks out: column 0 is the baseline, 1+2e is +a and 2+2e is −a.

| Config | Gold dm slope | Premise-balanced controls n, mean ± SD | z | Signed rank | One-sided p | vs within-string p |
|---|---|---|---|---|---|---|
| 1, 25 (primary) | +0.5693 | 500, +0.0374 ± 0.3969, −1.112 to +1.281 | +1.340 | 41/501 | **0.0818** (40 at least as large) | 0.0495 (4/100), z +1.70 |
| 0, 25 | +0.7381 | 100, +0.016 ± 0.456 | +1.58 | 7/101 | 0.069 | 0.069 |
| 0, 23 | +3.296 | 20 | +1.62 | 2/21 | 0.095 | 0.143 |
| 0, 27 | +0.039 | 20 | +1.22 | 3/21 | 0.143 | 0.143 |
| 2, 25 | +0.529 | 20 | +0.92 | 6/21 | 0.286 | 0.190 |
| 3, 25 | +0.550 | 20 | +1.08 | 4/21 | 0.190 | 0.095 |

- Other primary-test numbers: absolute rank 74/501; rank per natural SD 91.
- Second question: +0.2458, 27 of 100 controls at least as large, p = 0.2772, z +0.64.
- The table row matches in every cell, including the isotropic range −0.385 to +0.295.

**Sign convention.** It is the registered one. `sign` = +1 from the 20-control slope; the p counts controls ≥ the real slope; p = (1 + count)/501. The gold dm points toward "valid" (class gap on the steered half +0.025 s).

**Split 0 is the 20-control run.** Same 150 indices, same `s_unit` (19.259335), the same slope (+0.7380565) in both raw files. Per-item outputs and baselines are bit-identical (difference 0.0) for the three real directions and the first 20 isotropic, within-string dm and within-string probe controls.
- Of the 80 additional within-string controls, 6 exceed the gold slope. The "first of 21" was a property of the first 20 draws.

**Split 1.**
- Fitted on the complementary 13 blocks: `split1 == ~split0`, 1248 items, 624 premises.
- Its 150 steered items: all 12 selection-half blocks, 138 premises, 75 valid.
- No item or premise of the fitting half is steered; no premise lies in two blocks. The hypothesis strings are shared by design.
- The saved item indices of all six configurations are reproduced from the seeds.

**Premise-balanced labellings, regenerated from the seeds.**
- Split 1 (500): every labelling has one valid and one invalid item in all 624 premises, and exactly half valid in all 48 strings.
- Agreement with gold is 0.50088 (SD 0.0198; binomial 0.0200); successive agreement 0.50247. Both equal the JSON values.
- Lag-1 autocorrelation is −0.02. Per-premise frequency of "first item valid" is 0.499 ± 0.0225 (binomial 0.0224).
- All 500 are distinct, and none equals gold or its complement.
- Splits 0, 2 and 3 behave the same way.
- The within-string shuffles keep the string balance and, as intended, not the premise balance (50%).

**Calibration of the control family (primary configuration).**
- Premise-balanced dm controls: skew −0.03, excess kurtosis −0.28, Shapiro p 0.52, 54% positive. The normal-theory p (0.090) is close to the empirical 0.082.
- The 95th percentile is +0.666, and a slope of +0.675 would have been needed for p ≤ 0.05.
- Within-string versus premise-balanced: KS p 0.059, means −0.041 versus +0.037 (Welch p 0.053). The families are close but not identical.
- **Anisotropy is large.** The root-mean-square slope of the premise-balanced controls is 2.10 times that of the 20 isotropic directions (F-test p 0.0002; split 0: 1.89). See F5.
- The pairwise cosine between premise-balanced dm controls is 0.000 ± 0.164 (range ±0.66).

**Item-level robustness (directions fixed).**
- Gold slope positive in 150/150 items and all 12 blocks; valid items +0.665, invalid +0.473.
- Premise cluster bootstrap (5000 resamples): slope 95% [+0.489, +0.655]; p [0.074, 0.096], median 0.084.
  - **p ≤ 0.05 in 0 of 5000; beyond all 500 in 0.** The second question reaches p ≤ 0.05 in 0.
- Leave one block out: p 0.078–0.088.
- By hypothesis type, z ranges from +0.95 to +1.47 and p from 0.068 to 0.170. No type stands out.
- Monte Carlo SE of p with 500 controls: 0.012.

**Geometry and prediction.**
- cos(gold dm split 0, split 1) = **0.839**. Splits 2 and 3 have cosine 0.87–0.94 with both.
- **A fixed linear response predicts the split-1 slope from the cosine: 0.839 × 0.738 = +0.619 ± 0.216.** The observed +0.569 is −0.23 SD from that. Splits 2 and 3 predict +0.670 and +0.695; observed +0.529 and +0.550.
- Linearity holds well, which justifies this prediction. A kernel-ridge linear response fitted to the 900 control directions of split 1 predicts held-out control slopes with LOO R² 0.98, and predicts the gold dm at +0.638 (observed +0.569; split 0: +0.816 versus +0.738).
  - So the slope is fixed by geometry, and item noise is irrelevant. This cannot distinguish a label-related from a label-free response; it only shows the prediction logic is sound.
- The gold dm is largely an adjacency direction: cos +0.81 with the "parent"-items adjacency dm and +0.58 with within-string adjacency (split 1).
- At natural scale, moving across the gold class gap changes the answer by +0.014 logits, on answers with SD 7.2.
- Positive control: the model-answer probe ranks first of 21 in both splits (z +2.55 / +3.78). The setup can detect a direction the model uses.

**Category, how firmly, and the likelihood ratio.**
- **"p > 0.05", firm.** As a category, the registered figures give 46% / 9% ≈ 5 against an effect of the first size.
- At the observed value, normal approximations give the following density ratios (null over effect):

| Null model | Effect model | Ratio favouring the null |
|---|---|---|
| Fixed label-free response given the first slope, N(0.619, 0.216) | Effect of the first size, N(0.738, 0.397) | 2.0 |
| Same | Effect of the first size, N(0.738, 0.216) | 1.3 |
| Same | Half-size effect | about 1.1 |
| Gold exchangeable with its controls, N(0.037, 0.397), ignoring the first slope | Effect of the first size, N(0.738, 0.397) | 0.45 (favours the effect) |
| Same conditional null | Effect sized like the tenth audit's model (split-1 z centred about 2.6–3; inferred from its 9%) | about 4.6–6 |

- The last row is rough: I did not rerun that simulation.
- **May be concluded:** the slope did not stand out from its controls. It sits at about the 92nd percentile in both halves, which for one of 24 comparisons is unremarkable. There is no evidence for a label-specific effect at this site.
- **May not be concluded:**
  - "No effect", or that the direction "did not reproduce": the sign and 77% of the size came back, as the geometry guarantees.
  - Strong evidence against a smaller effect.
  - Anything from the second question, which is uninformative when null by registration.

**The within-string comparison (p = 0.0495).**
- It is **not registered.** It was the 00:55 test, replaced at 01:40 because, under the selection, it would have called a label-free response "replicated" 57% of the time.
- It sits on the boundary: 4 of 100 controls at least as large, Monte Carlo SE about 0.022, p ≤ 0.05 in 88% of item bootstraps.
- Even if it counted, the registered reading of anything between the threshold and 0.05 is "evidence for neither account." **It does not matter.**
- Against the 20 isotropic directions p is 0.048 (also unregistered, and against a family whose responses are known to be uneven across directions).

## C. Patching

**All match the JSONs and the table rows.**

| | n (premises) | Mean D | Mean \|D\| | First layer ≥ 0.5 final-token recovery |
|---|---|---|---|---|
| hyp, label changes | 210 (105) | +0.645 [+0.244, +1.061] | 2.100 | 26 (layer 24: 0.12; layer 26: 0.77) |
| hyp, label preserved | 231 (142; 89 Yes/Yes, 142 No/No) | +0.349 [−0.002, +0.697] | 2.133 (2.774 / 1.731) | 26 (0.79) |
| prem, label changes | 210 | +0.356 [−0.057, +0.773] | 1.963 | 26 (0.79) |
| prem, label preserved | 231 | +0.125 [−0.234, +0.442] | 1.835 (2.174 / 1.622) | 26 (0.81) |

**Within-template difference (label preserved minus label changes).**
- An exact replica of 08b gives hyp **+0.2365 [−0.2213, +0.7349]** and prem **+0.0545 [−0.3612, +0.4925]**, matching.
- Weights (min of the two counts per type): collider 37, confounder 15, non-child descendant 15, non-parent ancestor 14, parent 69. "child" occurs only among label-preserving pairs and is excluded.

**Robustness (hyp; prem in brackets).**
- Joint premise bootstrap [−0.226, +0.719]; independent premises [−0.243, +0.696]; pairs [−0.245, +0.705]. Prem: [−0.405, +0.502], [−0.403, +0.480], [−0.399, +0.507].
- Leave one type out: +0.026 (without parent) to +0.352 (without collider); every interval contains 0. Prem: −0.095 to +0.292.
- Other weightings: weights by label-changing counts +0.296, by label-preserving counts +0.177, equal weights +0.171. Label-preserving pairs reweighted 50/50 Yes/Yes : No/No within each type: +0.230.

**Pair sets.**
- These are *all* candidates under the construction rules. Rebuilding `build_pairs` without a tokenizer gives exactly 210 and 231 candidates, and none was dropped for token alignment.
  - The rules: at most 5 variables, at most 2 pairs per premise; in "same" mode, at most one Yes/Yes and one No/No pair per premise.
- No duplicates: all (clean, corrupt) pairs are unique. 190 of 210 clean items are unique, because a clean item can pair with two different edits. Each unordered label-preserving pair appears once (later-letter edits only).
- Template mixes differ sharply: parent is 59% of label-changing pairs and 30% of label-preserving pairs; child is 0% versus 16%. That is why the comparison is within type.
- All 105 label-changing premises also occur among the label-preserving pairs (F4).

**The 60-pair runs are a subset of the all-pairs runs, but not a simple random one.**
- All 60 of each list are in the all-pairs list. Their LD and grid values are **bit-identical** to the all-pairs run (difference 0).
- Label-changing: the first 60 in premise order of a random 120, i.e. positions 1–95 of 210.
- Label-preserving: balanced 31/29 by shared label by design, against 39% Yes/Yes in all pairs.

**What the new interval excludes.**
- In |D| units, a label-changing excess above 0.22 (10% of the standardised mean |D| of 2.19). For prem, above 0.36 (18%).
- But |D| responds only to second order to a signed label shift β. Adding β to the symmetrised label-preserving D raises mean |D| by 0.06 at β = 0.5, by 0.22 at β = 1.0, and by 0.72 at β = 2.0.
- **The bound therefore excludes label shifts of about ≥ 1.0 logit (hyp) and ≥ 1.2 (prem), roughly half of mean |D|.** Smaller label effects are not excluded.

**Replication of +0.970.**
- **It did not replicate.**
  - New pairs only (150 label-changing, 171 label-preserving): **−0.050 [−0.585, +0.464]**; prem −0.003 [−0.534, +0.505].
  - The old 60 with the all-pairs weights: +1.016.
- The reported +0.24 mixes the two, and 26% of its pairs are the discovery sample.
- This is expected. +0.970 was the only positive interval excluding 0 among the 8 within-type comparisons in the table. With 8 comparisons, at least one positive exclusion occurs by chance about 18% of the time.
- Its small cells carried much of it: non-parent ancestor, label-preserving n = 6, |D| 5.2; parent No/No, n = 2, |D| 6.0.

## D. Other things a careful reader should know

- **F3 (minor; patching).** Label-preserving |D| depends strongly on the shared label: Yes/Yes 2.77 versus No/No 1.73 (hyp).
  - Using only one kind of label-preserving pair, the within-type difference is +0.557 (Yes/Yes) or −0.155 (No/No).
  - A comparison whose comparator's composition moves the result this much is weak evidence about the label. The 60-pair and all-pairs runs also differ in that composition.
- **F4 (minor; patching).** 08b bootstraps the two kinds as independent samples ("the two kinds of pair are independent samples"), but they share all 105 label-changing premises. A joint premise bootstrap gives nearly the same interval, so this is harmless here.
- **F5 (minor; settle table).** The isotropic column (−0.385 to +0.295) sits next to a slope of +0.569. Against isotropic directions the slope looks exceptional (p 0.048), but the controls respond 2.1 times more strongly. The column is a measure of that unevenness, not a reference for the slope; one note would prevent misreading.
- **F6 (minor; settle).** The registered weakness applies. In split 1 the steered half is the selection half, on which the layer was chosen.
- **F7 (info).**
  - The 20-control "first of 21" is not robust to more controls even on the original split (rank 7/101).
  - Layer 23 responds about six times more strongly (gold +3.30, controls SD about 2), and layer 27 hardly at all.
  - Smaller steps on split 0 give +0.933 (step 0.1) and +0.915 (step 0.3), against +0.738 at step 1: the response saturates.
  - In the 08b file, label-changing later-letter pairs have mean D −0.08 and earlier-letter pairs +1.38. The mean D of label-changing pairs is dominated by the direction of the letter change, not by the label.

## Could not check

- The forward passes themselves (no model load). Indirect support: split 0 and the 60 shared patching pairs reproduce earlier GPU sessions bit for bit.
- That the injected vectors were exactly the saved ones (only through the code path and the reproduction above).
- The tenth audit's power table (not rerun). The likelihood ratio under its effect model is inferred from its quoted frequencies.
- The 50% `premise_balanced` gold agreement is checked; the Markov chain's uniformity is not re-proven.
- The table note figures "3% / 51% / 46%" and "25% / 67% / 9%" for Qwen 3B.
- Log entries after 03:00 and the report text.

Scripts (read-only use of the repository):
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/lib07d.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s1_recompute.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s2_split0.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s3_rebuild.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s4_patch.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s5_template.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s6_pairs.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s7_settle_more.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s8_field.py
- /private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit18/s9_beta.py

All processes were single-threaded, one at a time, peak RSS 339 MB.

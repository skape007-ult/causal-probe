# Fifteenth review: the settle run for Qwen2.5-1.5B (raw report)

## Summary

The run is clean and the registered outcome is **"p > 0.05"**. Every number in the JSON, log and table row reproduces from the raw npz exactly.

**Findings by severity**
- **F1 (major, wording):** the slope came back in the registered sign (98% of items, 12/12 blocks) at 45% of its first size. "Not reproducible" does not describe that; "does not stand out from its controls" does.
- **F2 (major, descriptive, post hoc, mine):** the 100 within-cell probes are negative 100/100 in both splits (−0.151 ± 0.061; −0.217 ± 0.075). Their shared component has an implied slope of −0.246, beyond all 500 controls, and predicts the gold probe's slope (−0.145 against −0.112). This is a hypothesis, not a result.
- **F3–F7 (minor):** split 0 is the old data; two unregistered statistics change the category; table notes are literals; the docstring cites the 07:56 entry, written during the run.

**Q2.** All match: slope −0.1119; controls −0.0038 ± 0.0854, −0.2263 to +0.2283; p = 60/501 = 0.1198; z −1.266; rank 442/501; within-string p 0.1782; second question −0.1331, p 0.1089. Column pairing and sign convention are the registered ones.

**Q3.** Yes. Split 0 is bit-identical to the 20-control run (items, outputs, −0.25096). Split 1 is fitted on the complementary 13 blocks; its 150 items share no block or premise with the fitting half.

**Q7.** "p > 0.05", firm against item resampling (0 of 5000 bootstraps reach p ≤ 0.05). "Moderate evidence against an effect of the size first observed" is legitimate. The ratio of 9 belongs to the category; at the observed value I get about 4, and about 1 against a half-size effect. Not licensed: "no effect", "chance", "nothing label-related", or anything about validity beyond adjacency.

**Q8.** Fitting halves 2 and 3 share 7 of 12 blocks with split 0 and 5 with split 1. Gold-probe cosines: 0–1 0.531; 0–2 0.795; 0–3 0.798; 2–3 0.785; 1–2 0.700; 1–3 0.703. Cosine times −0.251 predicts −0.199 and −0.200 (observed −0.186, −0.240). "Not counted" is right.

**Sentence for a results page:** "In the held-out test fixed beforehand, the Qwen2.5-1.5B gold-probe direction again had a negative slope (−0.112 per s, under half the first −0.251) but lay inside its 500 matched controls (one-sided p = 0.12): moderate evidence against an effect as large as first observed, none against a smaller one."

---

# Audit 15: the settle run for Qwen2.5-1.5B (`r3s_qwen15b`, arm `gold_probe`)

Independent recomputation from the raw outputs. Code and result files only; RESULTS.md, README.md, report/ (except the one table named in the brief), EXPLAINER.md, notes/progress.md and log entries after 03:00 on 14 November were not read. Nothing in the repository was created or changed. No model was loaded.

## Verdict

The run is clean and the arithmetic is right: every number in the JSON, the stage log and the table row reproduces from `steering_settle_hyp.npz` exactly (maximum absolute difference over about 700 compared numbers: 0). The registered outcome is **"p > 0.05"** (p = 60/501 = 0.1198). No finding changes that. Two findings concern what the outcome may be said to mean.

## Findings

| # | Severity | Finding |
|---|---|---|
| F1 | major (wording, not the run) | "p > 0.05" is correct, but the slope did come out again in the registered sign (98% of items, 12 of 12 blocks, 5 of 5 hypothesis types) at 45% of its first size. That is what a fixed label-free response predicts from the cosine alone (0.531 × −0.251 = −0.133; observed −0.112), and equally what a real effect of half the first size predicts. The "likelihood ratio about 9" is the pre-computed figure for the whole category; at the observed value (p = 0.12, 0.38 control SDs from the 0.05 boundary) my normal approximation gives about 4. Against an effect of half the size the ratio is about 1. |
| F2 | major (descriptive, unregistered, mine) | A family already in the file contradicts "nothing there". Probes fitted to labels shuffled within (hypothesis string, adjacency) cells have a negative slope in 100 of 100 cases in both configurations (split 1: −0.151 ± 0.061; split 0: −0.217 ± 0.075). They share one component (pairwise cosine 0.37) which the gold probe also has (cosine 0.36 with each, 0.59 with their mean). Under linearity that shared unit direction has slope −0.246 per s in the primary configuration: more negative than all 500 premise-balanced controls (z −2.84). The gold probe's own slope is what its projection on that component predicts (−0.145 predicted, −0.112 observed; split 0: −0.200 and −0.251). Post hoc; see the caveats in section 7. |
| F3 | minor | Split 0 at layer 19 is not a new measurement: its baseline column, the three real directions and the first 20 controls of each family are bit-identical to the 20-control run's raw file. "Again far out" is the same observation against more controls, and against the 100 premise-balanced controls it is no longer first (rank 2 of 101; one control at −0.355). |
| F4 | minor | Two unregistered statistics would have given another category: the slope per natural SD (rank 1 of 501, by a margin of 3%; declared "not a test" before the run) and the slope divided by its between-item standard error (one-sided p 0.044). Everything based on the registered statistic stays above 0.05. |
| F5 | minor | Table notes: all figures in the notes are string literals in `t_settle()`, not read from result files. Those I could check are right (0.53; 32 comparisons; 0.0016; 0.0020). The Qwen 3B figures (0.84; 3% / 51% / 46%; 25% / 67% / 9%) appear under a table that has no Qwen 3B row and cannot be checked from this run. "1 to 12%" has no referent in its sentence (it is the rate of "beyond all 500"). |
| F6 | minor | `t_settle()`'s docstring cites the log entry of 07:56 among what was "fixed before the run". The run started at about 07:54 local time (finished 13:36:48Z, 42.6 minutes) and that commit is stamped 07:56:49: during the run, about 20 minutes before the first configuration finished and 40 before any file existed. The decision rule itself (00:55, 01:40, 02:45; code e336431 committed 02:46:28) predates the run. I did not read the 07:56 entry. |
| F7 | info | The second question is reported in the table although the registered rule asks it only if the primary test passes. The note says a null means nothing, which is the registered reading. |

## 1. Ingest and logs

**Recomputed and found correct.**

- Byte-identical to the download (`cmp`): the result JSON, both npz files, `round2_status`, `round2_environment`, `round2_output_status`, all four stage logs. `meta_kaggle.json` and `readout_kaggle.npz` equal the download's `meta.json` and `readout.npz`; the laptop's `meta.json` was not overwritten.
- `git diff --name-status bee0c13~1 bee0c13`: ten files added, one modified (`notes/log.md`). No pre-existing result file changed.
- Of 267 files in the download's `results/`, two differ from the repository: `steering_controls_multiplicity.json` and `answer_bias.json`. Both are the older versions of commit e336431 (verified against `git show`), and the ingest report lists them as "aggregates not taken from the run". Correct handling.
- Code: the run's environment file gives commit e336431; `scripts/07d_steering_settle.py`, `scripts/gpu_round2.py` and `src/` have no difference between e336431 and HEAD. The 20-control result file the script read for layer, C and sign is identical in the download and the repository.
- Logs: no warnings; all probe fits converged (shares 1.0; my laptop refits needed at most 270 of 2000 iterations); `configurations_dropped_for_time` is empty; no configuration shortened; all six planned configurations ran (37.9 of 100 minutes); no NaN or infinity in any output array.
- Baseline against the cache: maximum 0.053 to 0.056, mean 0.014, correlation 0.99997, on answers with SD 2.14. The 20-control run had the identical value (0.0534) and split 0's baseline column equals that run's bit for bit. Slopes are differences within one forward path, so this does not bias them. Not abnormal.
- Cosmetic: file times in the download are UTC read as local time.

## 2. Slopes, families, primary test

**Recomputed and found correct**, for all six configurations.

Column pairing: column 0 is the baseline, column 1+2e is +a and 2+2e is −a for entry e of `alphas`; the first n rows of `alphas` are (j, 1.0) in order, and split 0 has 84 further rows for steps 0.1 and 0.3. Three independent confirmations: the positive control (`model_out`) is positive in every configuration; split 0 reproduces the 20-control run item by item although that script lays its columns out differently (all + then all −); the smaller steps give −0.267 (0.1) and −0.261 (0.3) against −0.251 at 1.

| Split, layer | Gold probe slope | Premise-balanced controls: n, mean ± SD, range | z | Signed rank | One-sided p | Against within-string: p, z |
|---|---|---|---|---|---|---|
| 1, 19 (primary) | −0.1119 | 500, −0.0038 ± 0.0854, −0.2263 to +0.2283 | −1.266 | 442/501 | 0.1198 (59 as extreme) | 0.1782, −1.196 (rank 84/101) |
| 0, 19 | −0.2510 | 100, +0.0051 ± 0.0983, −0.3550 to +0.2085 | −2.605 | 100/101 | 0.0198 | 0.0099, −2.673 |
| 0, 17 | +0.3420 | 20, −0.110 ± 0.424 | +1.068 | 4/21 | 0.857 | 0.762 |
| 0, 21 | −0.0280 | 20, −0.001 ± 0.027 | −0.990 | 18/21 | 0.190 | 0.286 |
| 2, 19 | −0.1862 | 20, −0.007 ± 0.082 | −2.194 | 21/21 | 0.048 | 0.048 |
| 3, 19 | −0.2404 | 20, −0.017 ± 0.130 | −1.721 | 21/21 | 0.048 | 0.048 |

Other real directions in the primary configuration: gold difference of means +0.128, model answer +0.390 (z +4.61, first of 501), adjacency within string +0.079, adjacency ("parent" items) +0.184, label beyond adjacency −0.061.

Second question: slope −0.1331, 10 of 100 controls at least as negative, p = 11/101 = 0.1089, z −1.223. Matches.

Table row: −0.251, 150, −0.112, 500, −0.004 ± 0.085, −0.226 to +0.228, −1.27, 0.1198, 0.0021, isotropic −0.176 to +0.102, −0.133 and 0.109. All match.

Sign convention: `sign` = sign of the 20-control slope = −1; the count is of controls with slope ≤ the real slope. That is the registered test (one-sided in the sign of the 20-control run, p = (1 + count)/501). The gold probe points towards "valid" (class gap along it on the steered half +0.0088 s).

## 3. Splits and items

**Recomputed and found correct.**

- The pool: 2400 items, 25 blocks of 96, 1200 premises of two items each, 48 hypothesis strings. No premise lies in two blocks. Every string occurs in every block.
- Split 0 = `data.selection_mask` (12 blocks, 1152 items), split 1 = its complement (13 blocks, 1248 items). No block and no premise is in both halves of any split.
- The saved item indices of all six configurations are reproduced by the script's rule from the seed. Split 1's 150 items all lie in the selection half (75 valid, 138 premises, 12 blocks); none is in its fitting half and no premise is shared with it. The hypothesis strings are shared between halves by design (47 of 48).
- Split 0 at layer 19 is the 20-control configuration exactly: same 150 indices, same `s_unit` (13.172062), slope −0.2509640947977702 in both raw files, per-item outputs identical for the three real directions and for the first 20 isotropic, within-string probe and within-string difference-of-means controls (maximum difference 0.0).

## 4. The controls

**Recomputed and found correct.** The labellings do not depend on the activations, so I regenerated them from the seeds.

Premise-balanced labellings, split 1 (500): one valid and one invalid item per premise in all 500; exactly half valid in each of the 48 strings (26 items each on the fitting half) in all 500; agreement with gold 0.5009 (SD 0.0198, binomial 0.0200; the JSON says 0.50088); successive agreement 0.5025 (JSON 0.50247); all pairs 0.5000 ± 0.021; lag-1 autocorrelation of the agreement with gold −0.02; per-premise frequency of "first item valid" 0.499 with SD 0.0225 (binomial 0.0224); no duplicates; none equal to gold or its complement. Split 0 (100) and splits 2 and 3 (20 each): the same. The within-string shuffles keep the string balance and, as intended, not the premise balance (50% of premises have one of each).

Saved directions:

- Norms 0.99995 to 1.00007 (stored as float16); all finite.
- The 20 isotropic directions equal the regenerated ones (cosine > 0.9999999), which fixes the generator's position for the within-string shuffles.
- Refitted on the laptop's activations: gold probe cosine 0.9992 with the saved one (both splits); **all 500 premise-balanced probe controls of the primary configuration 0.9970 to 0.9987** (median 0.9981), each matching only its own labelling (largest cosine with any other saved control about 0.2); all 100 within-string probes 0.9975 or more; five within-cell probes 0.998; all difference-of-means, residual and adjacency directions 0.9992 to 1.0000. `s_unit` agrees to 0.0005 and the natural SDs to 0.3%. The laptop activations are close enough to tell.
- Model-answer probe: 0.9994 with the Kaggle logit differences, 0.988 (split 1) and 0.960 (split 0) with the laptop's (8 items change side of the median). Irrelevant to the primary test.

Geometry (saved directions, layer 19):

| Quantity | Split 1 | Split 0 |
|---|---|---|
| cos(gold probe, gold difference of means) | +0.252 | +0.329 |
| cos(gold probe, model-answer probe) | −0.008 | +0.047 |
| cos(gold probe, adjacency within string) | +0.021 | +0.013 |
| cos(gold probe, adjacency in "parent" items) | +0.088 | +0.085 |
| cos(gold probe, label beyond adjacency; with adjacency projected out) | +0.165; +0.189 | +0.184; +0.191 |
| Premise-balanced probes, pairwise cosine | +0.000 ± 0.055 (−0.25 to +0.23) | +0.001 ± 0.057 |
| Premise-balanced probes, cosine with gold probe | +0.001 ± 0.050 | −0.013 ± 0.061 |
| Within-cell probes, pairwise cosine | **+0.372 ± 0.034** | **+0.379 ± 0.035** |
| Within-cell probes, cosine with gold probe | **+0.362 ± 0.029** | **+0.356 ± 0.037** |

cos(gold probe of split 1, gold probe of split 0) = **0.531**.

**Could not check:** a bit-exact refit (the GPU run's activations were not exported).

## 5. Calibration of this run

- Premise-balanced probe controls (500): mean −0.0038 ± 0.0038, median −0.0026, skewness −0.01, excess kurtosis −0.27, Shapiro p 0.61, 51% negative; 5% quantile −0.144, minimum −0.226. Symmetric and normal. The normal-theory p of the gold probe is 0.103 against the empirical 0.120.
- Within-string probes (100): −0.0045 ± 0.090; same distribution (KS p 0.79).
- Isotropic (20): −0.012 ± 0.070, range −0.176 to +0.102. Ratio of root-mean-square slopes, probe controls to isotropic: 1.23 (variance ratio test p 0.31 with 20 directions: not distinguishable from 1). The unevenness caveat concerns false positives and is moot for this outcome.
- Natural SD along the direction: gold probe 0.00829 s; controls 0.00362 to 0.00457 (median 0.00404). The gold probe has 2.05 times the median and 1.81 times the largest. Per natural SD it is −0.000927 against a control range of −0.000902 to +0.000921: rank 1 of 501 by 3%, entirely from that factor. The label "a size, not a test" is right, and against the within-cell probes (natural SD 0.0070) the same statistic gives p 0.60.
- Size at natural scale: one unit of s is 120 natural SDs of the gold probe direction; moving across the gold class gap changes the answer by −0.001 logits.

Alternative statistics in the primary configuration (one-sided unless stated):

| Statistic | p against 500 premise-balanced | Category changes? |
|---|---|---|
| Mean slope (registered) | 0.120 | |
| Two-sided | 0.216 to 0.222 | no |
| Median over items; 10% trimmed mean | 0.116; 0.112 | no |
| Share of items with a negative slope (98%) | 0.110 | no |
| Against within-string controls | 0.178 | no |
| Against within-cell probes | 0.733 | no |
| Slope / between-item standard error (−26.8) | 0.044 | yes, to "in between"; unregistered |
| Per natural SD | 0.002 | yes, to "beyond all"; ruled out before the run |

Smaller steps exist only on split 0 (rank 21 of 21 among the first 20 within-string controls at 0.1 and 0.3, as at 1).

## 6. Item-level robustness

- Cluster bootstrap over the 138 premises, 5000 resamples, directions fixed: gold slope −0.120 to −0.104 (95%); p 0.102 to 0.130; **p ≤ 0.05 in 0 of 5000**; second question p ≤ 0.05 in 0 of 5000. Leaving out one block at a time: p 0.110 to 0.122. The spread is between directions, not items; item resampling says nothing about the direction draw.
- Monte Carlo error of p from 500 controls: ± 0.015; 0.05 is 4.8 standard errors away.
- Sign: negative in 98% of items (147 of 150), in all 12 blocks (−0.085 to −0.138), in 45 of 47 strings, for valid (−0.114) and invalid (−0.110) items.

| Hypothesis type | n | Slope | z against controls | p |
|---|---|---|---|---|
| has_collider | 29 | −0.109 | −1.45 | 0.082 |
| has_confounder | 16 | −0.100 | −1.00 | 0.152 |
| non-child descendant | 24 | −0.038 | −0.44 | 0.329 |
| non-parent ancestor | 29 | −0.120 | −1.03 | 0.168 |
| parent | 52 | −0.147 | −1.27 | 0.106 |

## 7. Which outcome, how firmly, and what it licenses

**"p > 0.05"**, firmly as a category: 59 of 500 controls are at least as negative; the slope would have had to be below −0.144 for p ≤ 0.05 and below −0.226 for "beyond all 500"; no resampling of items moves it.

Is "moderate evidence against an effect of the size first observed (likelihood ratio about 9)" legitimate?

- As the registered reading of the category, yes: 72% / 8% = 9 is the table written before the run. I did not rerun the tenth audit's simulation. A normal approximation reproduces its two figures roughly (8 to 10% under the effect; 56% to 75% under nothing, depending on whether one conditions on the observed first slope or only on its rank).
- For this particular result the figure is too high. At the observed slope the same approximation gives a density ratio of about 4.3 (nothing label-specific, given the first slope: centre −0.133, SD 0.072; effect of the first size: centre −0.251, SD 0.085). I would write "roughly 4 to 9" or drop the number.
- It is evidence against that one size only. For an effect of half the first size the registered table gives 72% / 45% = 1.6 and my approximation 1.1. The observed slope is 45% of the first one.

What may NOT be concluded:

1. That there is no effect, or that the first observation was chance. The sign and about half the size came back; a smaller real effect is as compatible with this outcome as none.
2. That the direction "does not reproduce". The 01:40 gloss "not even reproducible on the other half" does not fit a slope that is negative in 98% of items; what failed is standing out from the controls.
3. That nothing label-related is there (F2). The within-cell probe family lies wholly on the negative side in both halves, and its shared component, which is close to a probe of the label's majority value in its (string, adjacency) cell (cosine 0.85 with that probe refitted on the laptop; 0.13 and 0.01 with a probe of adjacency itself), stands outside all 500 premise-balanced controls in the primary configuration. The tenth audit predicted z about 2.2 for the gold probe against within-cell probes under a response along the gold probe direction, and about 0 under an adjacency-only response; observed in the arm's sign: −0.65 (split 1) and +0.46 (split 0). Caveats, all of which apply: I found this after seeing the data; it is one of about ten families; the shared component is not independent between halves (cosine 0.55) and has cosine 0.40 with the selected split-0 probe, so selection alone predicts −0.100 ± 0.078 for it in split 1 (observed −0.246, z −1.9 beyond that); the unit-direction slope assumes linearity. It is a hypothesis for a registered test, not a result.
4. Anything about validity beyond adjacency from the second question (p 0.109): registered as uninformative when null.
5. Anything about other layers (layer 17: +0.342, opposite sign, among widely spread controls; layer 21: −0.028), other models or the difference-of-means arm.

## 8. Descriptive rows

Fitting halves (blocks shared; share of items shared) and gold probe cosines:

| Pair | Blocks shared | Items of the fitting half shared | Cosine of gold probes | Steered items shared |
|---|---|---|---|---|
| 0 and 1 | 0 | 0% | 0.531 | 0 |
| 0 and 2 | 7 of 12 | 58% | 0.795 | 9 |
| 0 and 3 | 7 of 12 | 58% | 0.798 | 9 |
| 2 and 3 | 7 of 12 | 58% | 0.785 | 11 |
| 1 and 2 | 5 | 42% of split 2's, 38% of split 1's | 0.700 | 4 |
| 1 and 3 | 5 | 42% / 38% | 0.703 | 8 |

For comparison, two control probes have cosine 0.00 ± 0.055. The ninth audit's prediction holds: the gold probes of splits 0, 2 and 3 are largely one vector.

A fixed label-free linear response predicts each later slope from split 0's slope and the cosine alone:

| Split | Predicted | Observed | Deviation |
|---|---|---|---|
| 1 | −0.133 ± 0.072 | −0.112 | +0.30 SD |
| 2 | −0.199 ± 0.050 | −0.186 | +0.27 SD |
| 3 | −0.200 ± 0.078 | −0.240 | −0.51 SD |

So "most negative of 21" on splits 2 and 3 is what the overlap produces by itself, and split 0 is the original measurement (F3). **"Not counted" is the right treatment** for all three rows. If they are shown, the split 0 row should say it is the 20-control data against more controls, and that against premise-balanced controls it ranks second of 101.

## 9. `t_settle()` and the table

The row is correct and the outcome logic is right (beyond all: p at the minimum and at or below the threshold; else p > 0.05; else in between). Nothing in the notes misstates this run. Points: F5 (literal figures, Qwen 3B figures without a row, "1 to 12%" without its referent), F6 (docstring), F7 (second question shown although the primary test did not pass). The table shows neither the split-1 sign consistency nor the within-cell probe family; a reader gets "p > 0.05" without the two facts that bound its meaning (F1, F2).

## Could not check

- The forward passes themselves (no model load). Indirect support: split 0 reproduces an earlier GPU session bit for bit.
- That the vectors injected were the saved ones (only through the code path and the reproduction above).
- The power table of the tenth audit (not rerun; roughly consistent with a normal approximation).
- The Qwen 3B figures quoted in the table notes.
- Log entries after 03:00, including the 07:56 mapping of outcomes to wording.

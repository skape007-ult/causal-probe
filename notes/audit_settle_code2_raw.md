# Audit of the settle run's code, second version (tenth audit, raw report)

**Can it run as it is:** yes. Nothing I found is expected to lose the session, and nothing attaches a number to the wrong direction; every reported number was recomputed independently for both arms at the three sites, and resume, dropping, tiny mode and the launch path work.
**What must change first:** the written test, not the plumbing. (1) The "beyond adjacency" comparison is answered "yes" by a response to the adjacency direction alone (F3) and the pre-registered second question has no numeric criterion and, in the probe arm, 20 controls instead of the 100 it names (F4): project the adjacency direction out and write the criterion down. (2) State in the log that the error rate of "beyond all 500" holds for an isotropic label-free response only (F1), and that the rank per natural SD must not be read as a rank test in the probe arm (F5).
**What the owner must know:** the test can reach its threshold, but a real effect of exactly the size that was observed passes in about 1 run in 4 for Qwen 3B and 1 in 2 for Qwen 1.5B (given how the arm was chosen; 13% and 39% without that conditioning), against 3% (isotropic response) or 1 to 12% (anisotropy matched to the first run) when nothing label-specific exists. For Qwen 3B the middle outcome ("reproducible, no more than selection explains") is the most likely one under both accounts (67% with a real effect of the observed size, 51% without) and therefore says nothing; only "beyond all 500" (moderate evidence for) and "p > 0.05" (moderate evidence against) are informative.

Sat 14 Nov 2026. Reviewed at `6a0c1e3` (`git diff df7313e HEAD -- scripts src` is empty; `df7313e` is on `origin/main` by the local tracking refs). No model was loaded, nothing was fetched. "Ran" means: the stand-in (answer = item base + item-dependent gain x a fixed linear function of the added vector) on the cached activations of Qwen 0.5B (layer 13), Qwen 1.5B (layer 19, laptop rebuild) and Qwen 3B (layer 25, the GPU run's own cache), last token of the hypothesis; or an exact computation for a linear response (slope of a direction = constant x direction . w) with the script's own `build()` at the real control counts (500 premise-balanced controls of the arm's kind in split 1, including 500 logistic fits for the probe arm; no reduction was needed). Scratch scripts and outputs:
`/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit10/`
(`pb_a.py`, `pb_b.py`, `pb_c.py` chain; `buildcache.py`, `calib2.py`, `calib3.py`, `calib4.py`, `anchor.py` calibration; `power.py`, `outcomes.py` power; `q5.py`; `harness4.py`, `recompute.py`, `timing.py` robustness; `*.out`, `*.log`).

**Mistakes of my own.** One launch line relied on word splitting that zsh does not do; two stand-in runs stopped at argument parsing before reading or writing anything, and left two misnamed log files in the scratch folder, which I removed. Nothing was written into the repository except this file (`git status --short` empty before it; no file under `cache/`, `results/`, `scripts/`, `src/`, `notebooks/` changed during the session; the files under `kaggle_out/` with modification times later than now are archive timestamps from an earlier extraction).

## Summary

| | Severity | What |
|---|---|---|
| F1 | major | "Beyond all 500" has the implied error rate (about 5% for the selected arm) only if the label-free part of the response is isotropic. Under anisotropic label-free responses the gold direction is not exchangeable with its premise-balanced controls: up to 20% (Qwen 3B) and 12 to 82% (Qwen 1.5B probe) in pure families, 1 to 12% when the anisotropy is matched to what the 20-control run measured. |
| F2 | major | Power. At the observed size the test passes in 13% (3B) / 39% (1.5B) of runs, 25% / 48% given the selection. The middle outcome does not discriminate. |
| F3 | major | `gold_beyond_adjacency` has cosine +0.57 with the adjacency direction (controls 0.00 +- 0.24). A response to adjacency alone puts it at z +1.1 to +2.0 against its controls (one-sided p <= 0.05 in 22 to 76% of runs). It is also miscalibrated for premise-level responses (SD of z 1.5). |
| F4 | major (probe arm) | The pre-registered second question names 100 residual controls and gives no criterion or sign; the probe arm builds 20 (smallest p 0.048) and the script prints another comparison for it. |
| F5 | major (probe arm, descriptive number) | `abs_rank_per_natural_sd`: the gold probe's natural SD is twice that of every one of the 500 controls; with no effect it is first of 501 in 12% (isotropic) to 41% of runs. |
| F6 | minor | Nothing is saved inside a configuration; "the primary alone does not fit" exits with no output. |
| F7 | minor | No isotropic directions in the primary configuration, so the anisotropy that decides F1 cannot be measured there. |
| F8 | minor | Checksum is blind to order and sign of the vectors. |
| F9 | minor | `model_out` has no matched controls (its label is a premise-level one). |
| F10 | minor | Wording in the log and docstring (numbers carried over from the first version; "most extreme of 24"). |
| F11 | minor | The cache stage still builds `c2c_probe`, which this job does not use. |

## Findings

### F1 (major). The error rate of "beyond all 500 = survives the correction" depends on an isotropy assumption nobody has stated

`scripts/07d_steering_settle.py` lines 195-203 (controls), 460-476 (test); `notes/log.md`, last entry, "Amended decision rule" and "So the strict threshold has about its nominal size".

What happens. The gold direction is `unit(mean within-premise difference)`; a premise-balanced control is `unit(randomly signed sum of the same differences)`. The first contains the systematic part, the second only noise. If the model's response to a label-free vector is larger along some activation directions than others, the two have different slope variances, whatever the label. For the probe arm the difference is larger: a probe that fits something real loads on higher-variance directions than a probe that fits a random labelling (natural SD 0.0083 against 0.0036 to 0.0046 for all 500 controls).

How established (ran; `calib2.py`, `calib3.py`; 100,000 response draws per family, exact). Split 1, real control counts, primary comparison. SD of z; P(one-sided p <= 0.05) per side (nominal 0.05); P(real beyond all 500) per side (nominal 0.002); then, given the selection that happened (the arm first of 21 in absolute slope in split 0 against the first 20 within-string controls), P(beyond all 500) in the split-0 sign (the yardstick is about 0.05: what a Bonferroni test on one of 24 gives its most extreme arm).

| label-free response | Qwen 3B diffmeans: SD z / P(p<=.05) / P(all 500) | given selection: P(p<=.05) / P(all 500) | Qwen 1.5B probe: SD z / P(p<=.05) / P(all 500) | given selection: P(p<=.05) / P(all 500) |
|---|---|---|---|---|
| isotropic | 0.98 / 0.046 / 0.0011 | 0.54 / **0.029** | 1.00 / 0.050 / 0.0019 | 0.29 / **0.024** |
| premise-level directions | 1.26 / 0.108 / 0.0050 | 0.89 / 0.174 | 1.56 / 0.145 / 0.029 | 0.66 / 0.249 |
| within-premise directions | 1.36 / 0.134 / 0.011 | 0.83 / 0.161 | 2.16 / 0.223 / 0.084 | 0.77 / 0.394 |
| the same, class-mean difference removed | 1.27 / 0.112 / 0.0046 | 0.79 / 0.101 | 1.40 / 0.120 / 0.017 | 0.51 / 0.121 |
| string-centred activations | 1.30 / 0.120 / 0.0072 | 0.88 / 0.205 | 1.95 / 0.200 / 0.062 | 0.77 / 0.382 |
| all activation variance | 0.91 / 0.033 / 0.0004 | 0.33 / 0.008 | 1.93 / 0.198 / 0.062 | 0.75 / 0.365 |
| top 20 principal components | 1.24 / 0.102 / 0.0064 | 0.74 / 0.106 | 4.23 / 0.359 / 0.246 | 0.93 / 0.819 |
| top 100 principal components | 1.08 / 0.065 / 0.0026 | 0.65 / 0.064 | 2.26 / 0.236 / 0.095 | 0.80 / 0.462 |

The other comparisons under the same families, split 1 (SD of z; P(one-sided p <= 0.05)): `gold_beyond_adjacency` vs `within_cell_residual` 0.95 / 0.04 isotropic, 1.51 / 0.21 premise-level, 1.47 / 0.19 string-centred (3B, n = 100); `adjacency_within_string` vs its shuffles 0.98 isotropic, 1.75 / 0.18 within-premise, 1.34 to 1.43 premise-level and string-centred; `gold_probe` vs `within_cell_probe` (1.5B, n = 100) 1.03 isotropic, 1.36 to 1.75 within-premise, 2.0 to 2.35 top components; `gold_diffmeans` vs `within_string_diffmeans` 1.03 isotropic, 0.88 premise-level, 1.5 to 1.6 within-premise. Only the isotropic column is calibrated, which is the only one the author checked.

Which family is real cannot be known without the model, but the 20-control run constrains it: rms slope of shuffled-difference directions over isotropic ones was 1.81 (3B) and 1.23 (1.5B), of shuffled probes over isotropic 1.28 and 1.07. A pure anisotropic family predicts ratios of 4 to 14 for the first and 0.1 to 0.7 for the second, so none of them is the whole response. Mixing each family with an isotropic part so that the first ratio is reproduced (`calib3.py`, 60,000 draws) gives, for P(beyond all 500 | selection): Qwen 3B 0.008 (components 21 to 300), 0.020 (all variance), 0.048 (top 100), 0.071 (premise-level), 0.080 (within-premise without the class-mean difference), 0.081 (top 20), 0.088 (string-centred), 0.118 (within-premise); Qwen 1.5B 0.022 to 0.035 for those families (the measured ratio leaves them 0.3 to 7% of the response energy), and 0.112 for a mid-spectrum family (components 21 to 300), the only one that comes close to both measured ratios (matched at 1.23, it predicts 1.09 for the probes; measured 1.07). So the plain statement is: the rule has its implied 5% only under isotropy; with what is known about the response the rate is between 1% and about 12% for either model; it could be 20% (3B) or far higher (1.5B probe) under responses that the first run makes unlikely but does not exclude.

A read-out of the model's own answer (ridge regression of the cached logit difference on the activations) as the response vector is not a stable account: for Qwen 3B it puts the gold direction at z +0.1 to +0.9 for every penalty; for the Qwen 1.5B probe between z -3.4 and +2.3 depending on the penalty and on whether the answer is first centred within string and class. The measured slope of `model_out` in the 20-control run bounds what such a component can contribute to the arm's slope at +0.05 per s (3B, measured +0.74) and +0.02 (1.5B, measured -0.25) (`anchor.py`).

Smallest fix. In the log, before the run: "The 5% reading of 'beyond all 500' assumes that the response to label-free directions is isotropic. Under label-free anisotropic linear responses compatible with the 20-control run the chance is 1 to 12%." Add F7 so that the anisotropy is measured in the primary configuration. No change to the test is needed.

### F2 (major). Power: what the three outcomes can and cannot show

How established (ran; `power.py`, `outcomes.py`; response = isotropic background + kappa x a planted unit direction estimated from all items, so both halves estimate it; kappa set so that the arm's median z in split 0 against its first 20 within-string controls has the stated value; 20,000 to 60,000 draws; analytic check: z ~ N(2,1) beats the maximum of 500 standard normals with probability 0.165, with an infinite reference set 0.193).

Probabilities of the three outcomes given that the arm was first of 21 in split 0 in the observed sign:

| response | Qwen 3B: all 500 / middle / p > 0.05 | Qwen 1.5B: all 500 / middle / p > 0.05 |
|---|---|---|
| label-free, isotropic | 0.027 / 0.511 / 0.463 | 0.023 / 0.259 / 0.718 |
| validity effect, median split-0 z = 1 | 0.089 / 0.649 / 0.262 | 0.089 / 0.461 / 0.450 |
| z = 1.5 | 0.147 / 0.688 / 0.166 | 0.159 / 0.526 / 0.315 |
| z = 2 (observed for 3B: +2.06) | **0.245** / 0.668 / 0.088 | 0.265 / 0.534 / 0.201 |
| z = 2.75 (observed for 1.5B: -2.75) | 0.500 / 0.480 / 0.020 | **0.476** / 0.444 / 0.079 |
| z = 3.5 | 0.850 / 0.149 / 0.001 | 0.703 / 0.277 / 0.020 |

Without conditioning on the selection, at the observed size: Qwen 3B median p 0.024, P(p <= 0.05) 0.67, P(beyond all 500) 0.13; Qwen 1.5B median p 0.004, 0.86, 0.39. With a mildly anisotropic background (matched as in F1) the 3B figures are 0.62 and 0.17. The threshold is reached almost surely only from a split-0 z of about 4 (0.98).

Consequences.
- A real effect of the size that was seen would pass in a quarter (3B) or half (1.5B) of runs. The arm was picked as an extreme of 24, so its true size is more likely below than at the observed one; at z = 1 to 1.5 it passes in 9 to 16%.
- "Beyond all 500" is 9 times (3B) or 20 times (1.5B) more likely under an effect of the observed size than under an isotropic label-free response; with F1's upper range, 2 and 4 times.
- "0.002 < p <= 0.05" is as likely with a real effect of the observed size as without (ratio 1.3 for 3B, 1.7 for 1.5B). The wording "reproducible, and no more than selection explains" is accurate as a description but must not enter claim 3 as evidence for the selection account: it is evidence for neither.
- "p > 0.05" is informative against an effect of the observed size (ratio 0.19 and 0.11), less so against a smaller one (0.6 at z = 1).
- More steered items do not help: the spread is between directions, not between items (bootstrap interval of the 3B slope in the first run +0.64 to +0.83, control SD 0.38). More controls help little (5,000 controls: 0.19 instead of 0.165 at z = 2).

Smallest fix. None in code. Put the table's two bold rows and the three likelihood ratios into the log before the run, so that the middle outcome is not later read as a result.

### F3 (major). `gold_beyond_adjacency` is not orthogonal to adjacency where it matters, and a response to adjacency alone makes it stand out

Lines 204-215, 221-230; decision rule ("if not, the finding is an effect of adjacency"); expectations ("a clearly positive result for the label beyond adjacency").

What happens. The label minus its cell mean is orthogonal to adjacency as a vector over items. Its image in activation space, `(y - ybar) @ Xc`, is not orthogonal to the image of adjacency, `(t - tbar) @ Xc`: cosine +0.571 (Qwen 3B) and +0.592 (Qwen 1.5B) in split 1, +0.40 and +0.33 in split 0, while the 100 controls have cosine -0.00 +- 0.24 with it (`q5.py`). Both vectors are covariance directions and are dominated by a few high-variance components; inside (string, adjacency) cells the label is still related to graph structure that the activations encode along the same components (the residual label correlates +0.19 to +0.26 with the number of common neighbours and with "the pair is marginally correlated", R squared 0.07 to 0.09 on the parser's other features). The real direction also carries more premise-level variance than its controls (squared projection on the top five premise-level components 0.66 against 0.32), which is why it is miscalibrated for premise-level responses (F1).

How established (ran, `power.py`, Qwen 3B, 20,000 draws). Response to the within-string adjacency direction only, at the strength at which a validity response gives a split-0 z of 2 / 2.75 / 4: `adjacency_within_string` z +1.9 / +2.4 / +3.0 (as it should), and `gold_beyond_adjacency` against `within_cell_residual` median z +1.12 / +1.47 / +1.96, one-sided p <= 0.05 in 22% / 37% / 76% of runs (nominal 4%). For comparison a response to validity beyond adjacency only gives z +1.74 / +2.11 / +2.47 (56% / 86% / 100%), and a response to the full validity direction of the observed size gives z +0.78 (16%). So at realistic sizes the comparison separates "adjacency only" from "validity beyond adjacency" by about half an SD, and reads the first as the second. For Qwen 1.5B the same response moves the residual direction by 1.0 to 1.5 SDs; `gold_probe` against `within_cell_probe` stays at z 0.0 (5%) under an adjacency-only response, so that family behaves, but see F1 for its calibration.

With the adjacency direction of the same fitting half projected out of the real vector and of every control (computed from the same slopes, exact for a linear response): adjacency-only response z -0.12 / -0.17 / -0.32 (3% / 3% / 2%), beyond-only response z +1.57 / +2.02 / +2.59 (44% / 73% / 99%), no effect 4%.

What a result licenses, as built. Positive: the label minus its (string x binary adjacency) cell mean points along something the model responds to; this includes a graded encoding of adjacency or of correlation structure, and any premise-level response. Null: nothing; at the observed size the comparison has 16% power under a full validity response, and 21 to 22% of the fitting items lie in single-class cells and do not enter it at all (in the "X directly causes Y" template 59 to 67%; that template supplies 7 to 12% of the residual label's energy, so the comparison is mostly about the other four templates). "If not, the finding is an effect of adjacency" does not follow from a null.

Smallest fix. In `build()`, after `tbar` is computed: `a_dir = steer.unit((tf - tbar) @ Xc)`; `perp = lambda v: steer.unit(v - (v @ a_dir) * a_dir)`; add `gold_beyond_adjacency_orth = perp((yf - ybar) @ Xc)` and `within_cell_residual_orth = perp((lab - ybar) @ Xc)` for the same labellings, with a `MATCH` entry. Cost 101 directions (202 vectors per item) in splits 0 and 1: about +13% and +21% of those configurations, +7 minutes for Qwen 3B. Without touching the run it can be done afterwards from the saved directions and slopes, `(slope_d - c * slope_a) / sqrt(1 - c^2)` with `c = d . a_dir`, on the assumption that the response is linear in the step. Either way the claim is "beyond the one linear adjacency direction", not "beyond graph structure".

### F4 (major for the probe arm). The pre-registered second question is not what the probe arm computes, and has no criterion

`notes/log.md` ("also stands out against 100 controls built the same way from labels shuffled within those cells"); lines 213-220, 503-508.

What happens. For `gold_probe` the script builds `within_cell_residual` from `wc[:n_other]`: 20 controls in split 1 and in split 0 (ran: counts 20 at Qwen 1.5B layer 19). The smallest one-sided p is 1/21 = 0.048. The script's final print gives, for the probe arm, `gold_probe` against 100 `within_cell_probe` next to it, which is a different comparison (gold probe weights against probes of within-cell labellings) from the one the log defines. Neither the log nor the code says what "stands out" means (z above what, which p, one- or two-sided) or in which sign; for Qwen 1.5B the arm's sign is negative and the residual direction points towards "valid", so both signs can be argued after the fact. The first pre-registration had a number ("z below 1"); the amendment dropped it.

Smallest fix. Line 214: use `wc` for both kinds (+80 directions, +160 vectors per item in splits 0 and 1 of the probe arm, +10% and +16% of those configurations). In the log: "stands out = one-sided p <= 0.05 in the arm's sign against the 100 controls, for the direction with the adjacency direction projected out (F3)", and name which of the two comparisons counts for the probe arm.

### F5 (major for the probe arm; a descriptive number that the log presents as a safeguard). The rank per natural SD is not a rank among comparable directions

Lines 251-253; log: "the rank per natural SD is reported next to the rank per unit of norm".

What happens. `abs_rank_per_natural_sd` ranks |slope x natural SD|. For Qwen 1.5B in split 1 the gold probe's natural SD is 0.0083 s, the 500 premise-balanced probes span 0.0036 to 0.0046: the real direction's statistic is doubled relative to every control before any slope is measured. Ran (`calib4.py`, 30,000 label-free draws): first of 501 in 11.9% of draws for an isotropic response, 25 to 41% for the anisotropic families (nominal 0.2%). For the Qwen 3B difference of means the natural SDs agree (0.083 against a median of 0.086) and the statistic is conservative (0.0%). The number itself is computed as labelled (verified).

Smallest fix. No code. Do not report this rank for a probe arm as a place among controls; report the two natural SDs and the effect per natural SD as a size.

### F6 (minor). Nothing is saved inside a configuration; the "does not fit" branch ends with no output

Lines 403-425. The primary configuration is 150 x 1,573 vector-items (about 28 minutes for Qwen 3B at the earlier rate, 15 for Qwen 1.5B); the partial file is written only when a configuration is complete. Ran: with a slow stand-in and `--max-minutes 1.0` the script dropped all five trailing configurations at item 5 and then exited with code 1 ("the primary configuration alone does not fit"); on disk: the plan file only. At the planned sizes this needs the GPU to be more than three times slower than in the earlier runs, so it is not expected. Fix: write the rows done so far every 10 items (`rows_<key>`, `n_rows_<key>`) and resume from them; in the "does not fit" branch stop after the last item that fits instead of exiting (the test can be computed on fewer items).

### F7 (minor). The primary configuration has no isotropic directions

Lines 188-190 (`role == "original"` only). F1 turns on how anisotropic the response is; the ratio of rms slopes of control and isotropic directions measures it, and only split 0 will have it. Fix: add the 20 isotropic directions for `role == "primary"` too (+40 vectors per item, 2.5%).

### F8 (minor). The resume checksum ignores order and sign

Line 370: `check = sum(|vecs|)`. Ran: a partial file of the other arm with the same shape (diffmeans arm with 500 controls and probe arm with `--n-primary 480` both give 150 x 1,573) is rejected; a partial file made with the same vectors by a different model response is accepted (unavoidable: the check is on the inputs). A reordering of families or a sign flip of any direction would pass. Not reachable in a fresh session. Fix: a hash of `cols.tobytes()` (the vectors are rebuilt bit for bit on a restart: the plan file was identical across my runs), or a sum weighted by row and column index; a plain projection would cancel, because every direction enters with both signs.

### F9 (minor). `model_out` has no matched control family

`MATCH` line 265. The model's answer is a property of the premise as much as of the item; its probe direction is compared with probes of labellings that are balanced within string (and within premise). Ran: SD of z 1.08 for an isotropic response, 5 to 33 for the anisotropic families. It is a positive control and should be labelled as one; its z and rank are not interpretable.

### F10 (minor). Wording

- Log, "beyond all 500 controls in 5.4% of cases (Qwen 3B) and 2.4% (Qwen 1.5B)": those figures were computed by the first review with within-string controls. With the premise-balanced controls now used: 2.9% (+- 0.2) and 2.4% (+- 0.2), isotropic.
- Log, "every real-against-control comparison ... SD 0.91 to 1.13 ... 3 to 7%": true for isotropic responses only, and the check used 200 primary controls (`calib_settle.py`), not 500. I reproduce it at 500 (table in F1, first row).
- Log and docstring, "the most extreme of 24": three arms ranked first of 21 (`results/steering_controls_multiplicity.json`: Qwen 0.5B probe-pool final token z +2.49, Qwen 1.5B -2.75, Qwen 3B +2.06); the Qwen 3B arm is the third of them by |z|. The threshold 0.05/24 is valid for any of the 24, so only the phrase is loose.
- Log, "drops configurations from the end": random splits go before the neighbouring layers (ran). Say which you would rather keep.
- JSON `note` in `primary_test` and the final print do not mention the isotropy assumption.

### F11 (minor). Unused work in the cache stage
`gpu_round2.py` line 227 still builds `c2c_probe` (first review, F12). Minutes, no risk.

## Answers

**Q1. First review's findings.**

| | Fixed as proposed? | Where | What remains |
|---|---|---|---|
| F1 | yes, as proposed; the proposal itself is insufficient | lines 204-230; old families removed; `within_cell_probe` kept for the probe arm | Calibrated for isotropic responses only; a planted adjacency response makes the residual direction stand out (F3); 20 controls in the probe arm (F4). |
| F2 | yes | `--n-primary` separate from `--n-shuf` (lines 315, 196); threshold 0.05/24 (line 468-469); log amended | The stated size holds under isotropy only (F1); power (F2). |
| F3 | yes, and more (the family is now the primary reference) | class `PremiseBalanced` lines 89-139; lines 195-203; rank per natural SD lines 251-253 | Remaining non-exchangeability (F1); the natural-SD rank in the probe arm (F5). |
| F4 | yes | lines 392-420; `--max-minutes 100`, stage limit 115 (`gpu_round2.py` 239-241) | Exit when the primary alone does not fit; no save inside a configuration (F6). |
| F5 | yes | lines 371-378; exported by the `steering*` rule | Fitting notes are not in the file (they are in the log of the stage). |
| F6 | yes | line 351 | - |
| F7 | partly | lines 370, 397-398 | Order- and sign-blind (F8). |
| F9 | yes | line 324 | - |
| F11 | yes | lines 283-284, 363 (`gold_class_gap_...`) | - |

**Q2. `PremiseBalanced`.**
(a) Correct (ran, `pb_a.py`). Splits 1, 0 and 2 (624, 576, 576 premises; 48 strings; out-degree 13, 12, 12 for every string): 30,000 reversals each, both balances and `first[]` checked after every reversal, out-lists compared with a rebuild from the labels after each of the first 200 and every tenth after that: no violation. Out-degree equals in-degree at every string and never changes, so an out-list cannot be empty; the walk stops at the first revisited string, at most 48 steps. Cycle length mean 5.4 (2 to 27), 24% of length 2, 41% odd (no parity trap).
(b) Uniform over all labellings with the two balances. Reasoned: start string uniform (out-degrees equal), every step has probability 1/degree, so a cycle and its reversal are proposed with equal probability; any two states differ by edge-disjoint directed cycles, so the chain is irreducible. Ran (`pb_b.py`): five synthetic instances built like the pool (4 to 6 strings, 2 to 4 blocks; 42 to 1,664 admissible labellings by enumeration): every labelling visited; chi-square against uniform p = 0.16 to 0.78 for the four instances run at thinning 6 to 8 (120,000 to 250,000 samples), and p = 0.70 for every fifth sample of the one run at thinning 1. The family is slightly wider than the gold label's own structure: gold has one valid and one invalid item per string in every block, the controls only over the fitting half (48% of (block, string) cells differ); premise balance already cancels block-level variance, so this does not matter.
(c) Mixing with burn 8,000 and thin 2,500 (ran, `pb_c.py`, split 1, 500 samples): each premise is flipped 21.5 times between two samples (minimum 15.6); agreement between successive samples 0.5025 (SD 0.020), at lag 5 0.4992, at lag 50 0.4999; per-premise autocorrelation at lag 1 +0.003 with SD 0.045 (independence gives 0.045), largest 0.13; 400 random linear statistics of the labelling: lag-1 autocorrelation +0.0002. No frozen structure: the graph is one strongly connected component, no premise has its ends in different components, and every premise flipped at least 187 times in 30,000 reversals.
(d) Valid. Agreement with gold: first sample 0.505, mean of 500 0.5009, SD 0.0198 (fair coins: 0.0200); correlation of the per-premise mean label with gold +0.04 (noise level 0.045). The controls have no association with adjacency (gold +0.061, controls -0.001 +- 0.030).
(e) 26.8 s on this laptop for the six chains (1.71 million reversals, 15 to 16 microseconds each); one to two minutes on Kaggle's CPU.

**Q3.** F1 (table, conditional figures, plain statement) and F2 (the rule's three outcomes). In one sentence each: "beyond all 500" has its implied rate under an isotropic label-free response (2.9% and 2.4% given the selection) and up to several times that otherwise; "0.002 < p <= 0.05" occurs in 51% (3B) and 26% (1.5B) of label-free runs given the selection, and about as often or more often with a real effect of the observed size; "p > 0.05" occurs in 46% and 72% of label-free runs and in 9% and 8% with a real effect of the observed size.

**Q4.** F2. Further rows (Qwen 3B, isotropic background, at the strengths that give a validity response a split-0 z of 2 and of 4): validity direction: primary p median 0.024 / 0.002, beyond all 500 in 13% / 98%, `gold_beyond_adjacency` z +0.78 / +1.89, `adjacency_within_string` z +1.0 / +2.0. Adjacency only: primary P(p <= 0.05) 25% / 73%, beyond all 500 1.0% / 3.4%, `gold_beyond_adjacency` z +1.12 / +1.96, adjacency z +1.9 / +3.0. Validity beyond adjacency only: primary P(p <= 0.05) 19% / 65%, beyond all 500 0.8% / 9.7%, `gold_beyond_adjacency` z +1.74 / +2.47, adjacency z +1.1 / +1.6. So a pure beyond-adjacency effect almost never passes the primary gate, and the z of the residual direction saturates near 2.5 whatever the strength (its controls have cosine +- 0.24 with any high-variance direction). Qwen 1.5B, response along the gold probe direction of all items at the observed size (z 2.75): primary p median 0.004, beyond all 500 in 39% (48% given selection), `gold_probe` vs `within_cell_probe` z 2.2 (p <= 0.05 in 67%), residual direction z 0.45, adjacency z 0.0.

**Q5.** F3 and F4.

**Q6. Robustness.**
- Projection: right. `work()` is tokens x vectors, the same quantity `w_done` accumulates. With the primary first, its first six prompts have 418, 412, 954, 246, 311, 400 tokens (mean 457; all 150: 438), so the first projection is not taken from long prompts any more: projection / true cost 1.00 to 1.02 at item 5 and 1.00 to 1.13 at the midpoint, also under a cost model with a fixed part per vector fitted to the earlier logs (`timing.py`). It is checked at item 5 and at the midpoint of every configuration with the cumulative rate.
- Dropping (ran, slow stand-in): trailing configurations are popped until the rest fits 0.9 of the time left; the running one is never popped (`len(plan) > ci + 1`); the index stays valid; the summary, the JSON (`configurations_dropped_for_time`), the raw file and the final print work with three remaining configurations and were verified number by number. The primary cannot be dropped; if it alone does not fit, the script exits (F6).
- Plan file (ran, `recompute.py`): from `steering_settle_hyp_plan.npz` plus `out_1_13` of a run killed in its second configuration I recomputed the primary slope, p, z and the rank per natural SD; all equal to the uninterrupted run. The stored check value is reproduced from the plan's float16 directions to 3e-8. Not in the file: fitting notes, role names, the labellings (these can be regenerated on the laptop, they depend on items and seeds only).
- Resume (ran): killed in the second configuration and restarted: the first is taken from the partial file, result JSON and raw file identical to an uninterrupted run, plan file identical, partial file removed.
- Threads (read, joblib 1.6.0 `_prepare_worker_env`): with `OMP_NUM_THREADS` unset each worker gets `cpu_count // n_jobs` threads, 1 on four cores with four jobs. No oversubscription. The fits finish before the model is loaded. Each worker imports torch through `steer` (a few seconds, some hundred MB).
- Sizes and time. Vectors per item (printed by the script): Qwen 3B 1,573 / 981 / 4 x 87, 435,300 vector-items, 92 logistic fits; Qwen 1.5B 1,613 / 1,021 / 4 x 87, 447,300 vector-items, 1,172 fits. At the first review's rates: 51 and 28 minutes of steering; fits 2 to 5 minutes (3B) and 8 to 29 (1.5B, by whether four jobs on two physical cores gain anything); chains 1 to 2 minutes. Both fit `--max-minutes 100` with more than 30 minutes to spare, and the stage limit of 115 leaves 15 minutes plus the 10% margin beyond the script's own limit. The longest steered prompt has 1,648 tokens, the same item as in the 20-control run. Memory of the plan: under 60 MB.
- Tiny mode (ran, the runner's exact options, both arms, in a results folder without the 20-control file): runs end to end, verification passes.
- Launch (read): notebooks pinned to `df7313e`, `--settle-arm gold_diffmeans --no-resid` and `--settle-arm gold_probe --no-resid`, jobs `cache,settle,export`; the two 20-control result files the script requires are tracked at that commit; the plan, partial and raw files are exported with `--no-resid` (name starts with `steering`).

**Q7.** Ran `harness4.py` (gain varying by template and item): Qwen 0.5B both arms at full size, Qwen 3B diffmeans arm, Qwen 1.5B probe arm, the tiny configuration for both arms, and a run with dropped configurations. Checked against an independent recomputation: counts of every family by role; order of configurations; every difference-of-means and residual direction, the first four probes of every probe family, the isotropic directions; both balances of every premise-balanced labelling; `s_unit`, natural SDs, gold class gap; every raw column including the smaller steps and their layout; all `versus` entries including `abs_rank_per_natural_sd`; `by_template`; smaller-step ranks; `families`; `primary_test` (slope, p, n, survives flag, p against within-string controls, smallest p); `arm_across_splits_and_layers_descriptive` for both families; plan and raw files. All checks passed in every run. Split 0 reproduces the 20-control run: natural SDs of the three real directions, the 20 isotropic and the first 20 of both shuffled families agree to 6e-5 or better for Qwen 3B (2e-3 for Qwen 1.5B, whose local activations are a rebuild).

**Q8.** F4, F5, F9, F10. Nothing else.

## Verified as correct

- `PremiseBalanced`: implementation, uniform stationary distribution, mixing at the script's settings, independence from the starting point (Q2).
- Every reported number is the number of the direction it is labelled with, both arms, all roles, with and without dropped configurations (Q7).
- The primary p-value is uniform for an isotropic label-free response at 500 controls (0.046 / 0.050; beyond all 500: 0.0011 / 0.0019).
- Time projection, dropping, plan file, resume, tiny mode, thread limits, the launch path as far as it can be read (Q6).
- The first review's F2, F3, F4, F5, F6, F9, F11 are fixed as proposed.

## Not checked

- Anything on a real model or on CUDA: time per vector at 1,573 vectors per item, half precision, GPU memory (same call and same longest prompt as the 20-control run).
- Fit times on Kaggle's CPU; joblib's behaviour on the four-core Linux machine was read, not run.
- Where the real response lies between isotropic and anisotropic: F1's range rests on two rms ratios of the first run, each from 20 directions (sampling SD about 16%).
- Non-linear responses: the stand-in is linear; the post-hoc projection of F3 and all calibration figures assume linearity at one step.
- Qwen 1.5B figures use laptop-rebuilt activations; the GPU run's will differ slightly.
- The notebooks were read, not executed; `KAGGLE.md` was not reviewed.

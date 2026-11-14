# Audit 11: runs r2_qwen7b (Qwen2.5-7B, 4-bit) and r2_gemma (Gemma-2-2B), ingest commit 8b8ca08

Independent review from code and result files only. Every number below was recomputed from raw outputs
(`cache/<model>/<tag>/meta.json`, `resid.npy` via mmap, `steering_controls*.npz`, `steering_adjacency_*.npz`,
`patching*.npz`) with my own scripts in this folder (`a_ingest.py`, `b1.py` … `b5.py`, `c1.py` … `c7.py`, `d1.py`, `e1.py`).
No model was loaded; the Gemma tokenizer was loaded offline once (to rebuild the patching pair lists).
Nothing in the repository was written. `results/` and `report/tables/` are unchanged between 8b8ca08 and the current HEAD (be56dec).

Severity: **critical** = a reported number is wrong; **major** = numbers right, but a reading they invite is not supported
or is contradicted by the same data; **minor** = caveat, fragility or bookkeeping.

No critical finding: every number I recomputed matches the JSONs and the tables.

---

## Findings

### F1 (major). "Qwen 7B is at chance on the debiased pool" holds only for the pooled AUROC; within hypothesis string the 7B answer logit tracks validity

Evidence: `cache/qwen7b-4bit/c2c_debiased/meta.json` (logit_diff, labels), items from `data.get_dataset("c2c_debiased")`; scripts `b2.py`, `b4.py`, `b5.py`.

* Pooled, as in `scripts/15_answer_bias.py`: AUROC 0.5135 [0.4899, 0.5363] → "at chance (AUROC CI includes 0.5)" by the script's rule (lines 73-75). Reproduced exactly.
* The pooled AUROC is an insensitive statistic on this pool: the answer logit differs between hypothesis types by far more than within them
  (7B type means −23.9, −20.9, −19.1, −3.8, −0.1; within-type SD 0.9–2.2), and the label is balanced inside every string, so between-string variance is pure noise for the label.
* Within hypothesis string (the comparison the pool is built for), 7B:

| statistic (7B, debiased pool) | value | interval / p |
|---|---|---|
| mean within-string AUROC over the 48 strings | 0.549 | [0.530, 0.570] premise-cluster bootstrap; permutation p ≤ 0.0005 (labels swapped within premise, 2000 draws; null mean 0.500, SD 0.010) |
| strings with AUROC > 0.5 | 33 / 48 | |
| non-parent ancestor (9 strings) | 0.673 (pooled within type 0.659 [0.608, 0.705]) | perm p ≤ 0.0005; 9/9 strings > 0.5 |
| non-child descendant (9) | 0.649 (0.629 [0.580, 0.680]) | p ≤ 0.0005; 9/9 |
| has_confounder (6) | 0.567 (0.551 [0.487, 0.616]) | p 0.04 |
| has_collider (10) | 0.487 (0.484 [0.435, 0.536]) | p 0.64 |
| parent, "X directly causes Y" (14) | 0.442 (0.458 [0.413, 0.496]) | p 0.008, **below** chance |
| within string × pair-adjacency cells (pair-weighted) | 0.591 | [0.559, 0.621]; above 0.5 in all five types (0.537, 0.556, 0.636, 0.642, 0.555) |
| within-premise pairs: P(LD valid > LD invalid) | 0.536 | n = 1200 |

* None of this moves the answers: the yes rate is 0.000 in three of the five types, so accuracy at threshold 0 is 0.5062 [0.4958, 0.5167].
* The same insensitivity hides structure in every model. Omnibus permutation test over the five types (sum of squared z): p ≤ 0.0005 for all five models.
  Mean within-string AUROC: qwen0.5b 0.506 (p 0.54), qwen1.5b 0.507 (0.41), qwen3b 0.512 (0.26), **gemma2b 0.473 (p 0.0035, below chance)**, **qwen7b 0.549 (p ≤ 0.0005)**.
  Of the 25 model × type within-type intervals, 11 exclude 0.5 (1.25 expected).
* The 7B pattern does not resemble the smaller Qwen models; it is close to their mirror image. Type-level mean within-string AUROC:

| type | 0.5B | 1.5B | 3B | 7B | Gemma |
|---|---|---|---|---|---|
| parent | 0.541 | 0.628 | 0.561 | **0.442** | 0.387 |
| ancestor | 0.445 | 0.449 | 0.506 | **0.673** | 0.480 |
| descendant | 0.449 | 0.430 | 0.444 | **0.649** | 0.482 |
| collider | 0.586 | 0.500 | 0.565 | 0.487 | 0.560 |
| confounder | 0.465 | 0.442 | 0.415 | 0.567 | 0.507 |

  After stratifying on pair adjacency the smaller models are at or below 0.5 (0.5B 0.487, 1.5B 0.435 [0.405, 0.464], 3B 0.513, Gemma 0.467 [0.433, 0.497]); 7B is the only one above (0.591).
* Caveats on my side: the within-string statistics are post hoc (chosen by me after seeing the pooled ones), adjacency is one shallow control among many possible, and the effect is small in logits (valid − invalid gap +0.5 to +0.8 in the two strongest types). The permutation p is at the floor of 2000 draws, so it survives any plausible count of statistics and models.

Reading: "at chance by the AUROC criterion" is literally correct and should not be shortened to "the 7B answers carry no label information on the debiased pool".

### F2 (major). The Gemma arm that "exceeds all 20 controls" (debiased pool, final token, layer 17, gold probe) is first by a 3% margin and is not robust

Evidence: `cache/gemma2b/c2c_debiased/steering_controls.npz`, `results/steering_controls_gemma2b_c2c_debiased.json`; `c1.py`, `c2.py`, `c3.py`.

* Slope +0.4754 [+0.4558, +0.4935]; largest control |slope| 0.4622 (a control with slope −0.462); second largest 0.369. Margin 0.013, ratio 1.03.
* Item bootstrap with directions fixed (premise clusters, 4000 draws): P(rank 1) = 0.81; interval of the margin over the largest control [−0.016, +0.042].
* z against the 20 controls +1.94 (normal-theory two-sided p 0.073); 96th percentile of the 100 isotropic directions (4 isotropic directions are larger; isotropic max 0.679).
* Per-item sign: 150/150 positive, all 13 blocks (0.44–0.53), all 48 strings (0.29–0.68), all five types (0.35 parent … 0.64 descendant). This is not evidence: 40% of all 143 directions have ≥ 95% same-sign items; shuffled probes have median 90%.
* All 21 probe fits it is ranked among (gold + 20 shuffles) stopped at `max_iter=300` without converging (`src/causal_probe/steer.py:117`; 21 ConvergenceWarnings in `r2.gemma2b.controls.c2c_debiased.log`; reproduced locally: 21 of 22 fits non-converged, the model-answer probe converges in 20 iterations). The converged gold probe (555 iterations) has cosine 0.996 with the 300-iteration one, i.e. a 9% orthogonal component; with control slopes of SD 0.24 that is enough to move the slope by more than the 0.013 margin. I could not measure the converged direction's slope (needs the model).
* Natural scale: the direction's natural SD is 0.0029 s, so the step s = 1 is 349 natural SDs. Slope × natural SD = +0.0014 logits; crossing the whole valid–invalid class gap along the direction (0.0030 s on the report half) moves the answer by +0.0014 logits, against an answer logit difference with mean −2.85 and SD 3.51. The direction decodes the label (report-half AUROC 0.78) and has no natural-scale effect on the answer.
* No replication: same pool and token in the three Qwen models −0.229, −0.142, −0.229 (ranks 6, 12, 17 of 21); Gemma on the original pool +0.276 (rank 6/21, z +0.86); Gemma at the other token −0.003 (rank 15/21, dead site, F3).
* This one arm moves the debiased-pool multiplicity line from 2 first ranks of 12 (P 0.109, parent commit) to 3 of 16 (P 0.038).

### F3 (major). Gemma's end-of-hypothesis sites on the debiased pool are causally disconnected from the answer; their rows are not null results

Evidence: `steering_controls_hyp.npz` (debiased), `steering_adjacency_hyp_{best,first50}.npz`, `patching*.npz`; `c6.py`, `d1.py`, `e1.py`.

* Steering controls, debiased pool, content_end, layer 21/26: every direction is ~30× weaker than at any other site of any model. Median |ΔLD| at +1 s over the 100 isotropic directions: 0.0068 logits, against 0.06–0.60 at the other 15 sites (Gemma original pool, content_end, layer 15: 0.179). Model-answer direction +0.0107 = 1.9 × the isotropic median (3.3–14 × elsewhere); its rank among shuffled probes is 7/21.
* Not a bug, as far as can be checked without the model:
  - same script, same run, same hook code at content_end layer 15 on the original pool gives slopes of order 1 (model-answer +1.09);
  - the unsteered pass reproduces the cache (max |diff| 0.035) and the adjacency self-check (shortcut vs full forward) is 0.008;
  - the hook fires: no exact zeros, model-answer slope has t = 27.5 over items (149/150 positive), 91% of the 143 directions have a mean slope with |t| > 3;
  - s_unit is right (121.65 recomputed from `resid.npy`, layer 21, position index 1); content_end is always 5 tokens before the final token; activations finite, max |x| 195;
  - independent confirmation from patching (different items, two runs): pooled recovery from patching the hypothesis region is 0.33–0.52 at layers 12–14, 0.05–0.07 at 16 and 0.00 at layers 18–26 (label-changing and label-preserving, hyp and prem kinds). After layer ~16 nothing at the hypothesis tokens reaches the answer.
* Why the site was chosen: `steer_select_gemma2b_c2c_debiased_content_end.json` picks the layer by decodability on a stride-2 grid; layer 21 wins with CV accuracy 0.747 against 0.735 (layer 15) and 0.734 (17), a difference of 14 items of 1152. Decodability is flat from 15 to 25; causal connectivity is not.
* Same for the adjacency test: hyp_best is layer 19/26 (positive control +0.008 per s, larger than 70% of controls; largest cell of the transfer matrix 0.003 logits; 79% of all item × vector cells change the answer by < 0.005), hyp_first50 is layer 5 (positive control −0.016 per s, 15%).
  The Qwen hyp_best sites sit earlier relative to their hand-over and are alive (positive control 100% in all three; +0.71, +1.40, +0.86 per s).
* Consequences:
  - The two Gemma debiased/content_end arms in `steering_controls_multiplicity.json` cannot show anything; honest counts are 30 comparisons (14 on the debiased pool), not 32 (16). With 30 and 14: P(≥ 4 first ranks) 0.053, P(≥ 3) 0.026.
  - `report/tables/adjacency_contrast.md` prints "95" (controls smaller) for a translation contrast of +0.001 logits at Gemma hyp_best, and its footnote says the positive control fails at "the earlier-layer sites"; for Gemma it also fails at the best-layer end-of-hypothesis site (70%), and is weak at final_first50 (90%, 2 controls larger).
  - Only one of the four Gemma adjacency sites (final_best) has a working positive control.

### F4 (major). The count of first ranks is the wrong summary of the 32 comparisons; the difference-of-means arms are systematically non-null, and the reason is behavioural

Evidence: 16 `steering_controls_*.json` + npz; `c4.py`, `c7.py`.

* Recomputed table (all 32 arms agree with `steering_controls_multiplicity.json`): 4 first ranks in |slope| of 32 (expected 1.52; binomial P(≥ 4) = 0.0640); debiased pool 3 of 16 (0.76; 0.0380); original pool 1 of 16 (0.542). First ranks: qwen0.5b original/final/diff-means (+0.865), qwen1.5b debiased/hyp/probe (**−0.251**, signed rank 21/21), gemma debiased/final/probe (+0.475), qwen3b debiased/hyp/diff-means (+0.738). Three different arm types, two signs.
* Dependence: within a (model, pool) the four arms share the steered items and, for the two tokens, the same 20 shuffled labellings. Spearman of |slope| across the 20 shared shuffle indices: probe vs diff-means at one site +0.05 (SD 0.22 over 16 pairs), final vs content_end +0.12 (SD 0.31). A permutation null that keeps this dependence (each of the 21 labellings in turn designated "real", jointly for the 4 arms of a cluster; clusters independent) gives P(≥ 4 of 32) = 0.063 and P(≥ 3 of 16) = 0.037, the same as the binomial. Bound for perfectly dependent arms inside a cluster: P(≥ 3 of 8 clusters hold a first rank) = 0.18. Dependence between models (same gold labels, same items) cannot be estimated from the controls, which differ between models.
* No arm survives a correction and none could have: with 20 controls the smallest attainable p is 1/21 = 0.048; Bonferroni over 32 needs 0.0016, i.e. rank 1 among ≥ 639 controls (319 for 16). Normal-theory p of the best arms: qwen1.5b debiased/hyp/probe z −2.75, p 0.015 (× 32 = 0.47); gemma debiased/final/probe z +1.94, p 0.073; smallest BH q over 32 arms 0.40.
* But "just another draw from the control family" is rejected in aggregate, which the count of first ranks hides. Mean |rank| over the 32 arms 7.4 (null 11; P ≈ 0.0003 under independent uniform ranks). By direction: difference of means 6.4 over 16 arms (P 0.001), 4.9 over the 8 debiased-pool arms (P 0.002), 15/16 slopes positive, mean signed rank 4.75; gold probe 8.3 (P 0.04), 10.0 on the debiased pool (P 0.33), 12/16 slopes negative.
* The difference-of-means slopes are the behavioural correlation between label and answer, read back through the answer direction. At the final token, over the 20 shuffled difference-of-means controls, slope ≈ b × cos(direction, answer difference of means) with r = +0.90 (qwen1.5b), +0.93 (gemma), +0.96 (qwen3b); the gold direction's slope is predicted from its cosine alone:

| model (debiased, final) | cos(gold diff-means, answer diff-means) | rank of that cosine among 21 | predicted slope | observed |
|---|---|---|---|---|
| qwen1.5b L22 | +0.492 | 2 | +1.60 | +1.89 |
| gemma2b L17 | −0.416 | 21 (most negative) | −1.79 | −1.80 |
| qwen3b L25 | +0.316 | 4 | +3.22 | +4.02 |
| qwen0.5b L14 | −0.067 | 13 | +0.00 | +0.15 (relation absent, r −0.15) |

  So a gold difference of means out-ranking its controls says that the answers are not label-independent within string (F1), not that a validity direction is used. Within-string shuffled labels are not a sufficient null for that claim.

### F5 (minor). Non-converged probe fits throughout the Gemma run

`ConvergenceWarning: lbfgs failed to converge after 300 iteration(s)` counts in `results/round2_logs_r2_gemma/`: controls.c2c_debiased 21, controls.c2c_probe 21, controls_hyp.c2c_debiased 21, controls_hyp.c2c_probe 4 (C = 0.1), adjacency.final.best 21, adjacency.final.first50 22, adjacency.content_end.first50 22, adjacency.content_end.best 0. These are the visible fits of `steer.probe_direction` (gold, adjacency, shuffles); the fits inside layer selection and the fresh-probe AUROCs go through `probes._fit`, which suppresses warnings (`probes.py:35`), so the count is a lower bound. Consequence measured for one case: cosine 0.996 (gold) and 0.993 (a shuffle) between 300-iteration and converged directions. Difference-of-means directions, isotropic directions, the translation, erasure and transfer vectors are unaffected.

### F6 (minor). The 7B forward pass ran in bfloat16, not float16; its logits are noisier than the other models'

`meta.json` says `model_dtype: torch.bfloat16`; every cached activation I sampled has ≤ 8 significant bits (100% at layers 0 and 14; 24% for qwen3b, which is float16). `models.py:118` comments "float16 compute"; the non-quantised modules follow the checkpoint dtype. The top-1 logit (lm_head, bfloat16) differs from the float32 answer logit by up to 0.125. For the 51 items that occur in both c2c_debiased and c2c_probe (different batches) the two logit differences differ with SD 0.17, max 0.45 (qwen3b: SD 0.034; Gemma: 0.013). Label-independent, so it can only attenuate; relevant to any later probe work on `cache/qwen7b-4bit/c2c_debiased/resid.npy`.

### F7 (minor). The ingest change is correct; three small gaps

`git diff 0451340 8b8ca08 -- scripts/kaggle_ingest.py`.
* `AGGREGATES` skip (lines 133-134, 339-341): correct. The downloaded `results/answer_bias.json` of run r2_qwen7b holds one model; taking it would replace the five-model file. The skip sits before the byte comparison and cannot mask another file.
* Gaps: (a) the list is hand-maintained; the other model-independent files in `results/` (`benchmark_baselines.json`, `sae_*.json`) are not produced by round-2 jobs, so nothing leaks today. (b) Step 7 regenerates `answer_bias.json` but not `cross_model.json` or `steering_controls_multiplicity.json`; those stay stale until someone runs 09/07e, and the skipped names are printed, not recorded in `round2_ingest_report_<run>.json`. (c) The "code changed since then" message diffs `src`, 07c, 17, 08, 02 only; it printed "none" although `scripts/gpu_round2.py` (the driver the run executed) changed after 08d99b1. I read that diff: additions for the settle job and a `--cache-tags` option, nothing that affects these runs. `01_build_cache.py` and `15_answer_bias.py` are also outside the list.
* `results/round2_ingest_report_r2_qwen7b.json` still lists `results/answer_bias.json` under `results_written` (the overwrite happened before the fix). The file was restored: the four older models are identical to the parent commit, and the 7B block is identical to the run's own copy.

### F8 (minor). Gemma patching: the two kinds of pair are not the same draw, and the within-template comparison has no power

`e1.py`; `cache/gemma2b/patching.npz` (earlier run, 13 Nov), `patching_same.npz` (this run).
* Label-changing: n = 40 pairs from 28 premises, the `--n-pairs 40` draw (rebuilt pair list matches the saved premise ids). It shares 38 of 40 pairs with the 60-pair draw used for the other models but is not its first 40. The footnote of `report/tables/patching_label_preserving.md` names only Qwen 1.5B as "a different draw".
* Label-preserving: n = 60 from 50 premises (31 Yes/Yes, 29 No/No), rebuilt list matches saved uids. 13 premises and 1 original item are shared between the two sets.
* Same layers (0, 2, …, 26). Template mix differs: parent 24/40 vs 21/60, has_collider 7 vs 17, "child" 0 vs 8, confounder 4 vs 2, descendant 3 vs 6, ancestor 2 vs 6.
* Within-template difference of mean |D| (preserving − changing): hyp −0.318 [−0.84, +0.17] (JSON −0.318 [−0.871, +0.192]), prem −0.320 [−0.69, +0.03] (JSON [−0.735, +0.027]). The weights are 21 of 35 on "parent"; three cells have n = 2–4; 13–18% of bootstrap draws are discarded because a template is absent; without has_collider the hyp difference is −0.10. The interval allows anything from "label change adds 70%" to "nothing".
* Direction of the letter change: label-preserving pairs are all later-letter edits; label-changing 18 of 40. Mean D in label-changing pairs: later letter −0.99, earlier +0.72 (hyp). The like-for-like comparison in `patching_by_template_gemma2b.json` (later-letter changing minus preserving) is −0.70 [−1.66, +0.14] for hyp and −0.68 [−1.41, −0.07] for prem. The prem interval excludes zero, in the direction "more No-like for the valid original"; it is one of eight such comparisons across models and resamples pairs individually although premises are shared. It is not in the table.

### F9 (minor). The model-answer "positive control" is not matched to its controls on the debiased pool

`07c_steering_controls.py`: `mcal = (ld0 > np.median(ld0))` over all items. On the debiased pool the answer logit is mostly a function of the hypothesis type (Gemma type means −7.0 … +1.5), so this probe mostly encodes which hypothesis is asked; its controls are probes for the gold label shuffled within string, which carry no string information. It is valid as a check that the site reaches the answer (which is how F3 uses it); "× 9.4 the control median" is not a matched comparison.

### F10 (minor). Bookkeeping in the run records

* `round2_status_r2_qwen7b.json`: `r2.qwen7b-4bit.behaviour.cache_part0` has `ok: false, exit code 1`. That is the designed stop at the 50-minute budget ("time budget reached; re-run to resume"); part 1 resumed c2c_probe at chunk 6 of 10 after a second model load. Consistency across the reload: the 51 items shared with c2c_debiased differ by at most 0.39 logits when cached in the first session and 0.45 in the second, i.e. the same batch noise (F6).
* Gemma environment: `bitsandbytes: not available` (not needed).
* `cache/gemma2b/c2c_debiased/{resid.npy,meta.json}` were replaced by this run's two-position cache. I compared with the earlier download in `kaggle_in/cache/gemma2b/c2c_debiased/`: final-position activations bit-identical at all 27 layers, logit differences identical. The existing `probe_oof_*.npz` there still correspond to the activations on disk.
* `baseline_max_abs_diff_vs_cache` in the two Gemma c2c_probe JSONs (0.047, 0.032) refers to the run's own cache (`meta_kaggle.json`); against the local MPS-built `meta.json` it is 0.046 in both.
* The difference-of-means controls of qwen1.5b's debiased runs reproduce the saved natural SDs only to 1% (other models: 1e-7); its local cache was built on MPS and the run rebuilt its own. Not part of these two runs.

---

## Answers to the questions

### A. Ingest
1. All files the runs created are present and byte-identical in `results/` (and in the 8b8ca08 tree): 8 for r2_qwen7b, 28 for r2_gemma. The only changed-by-run file that differs is `results/answer_bias.json` (intended). The 196 and 197 other files in the downloads are unchanged copies of the run commit's tree. Caches: all 9 (7B) and 14 (Gemma) files byte-identical (sha256), with `gemma2b/c2c_probe/meta.json` stored as `meta_kaggle.json` beside the local one, as designed.
2. Modified pre-existing files in the commit: `notes/log.md`, eight `report/tables/*.md`, `results/answer_bias.json`, `results/steering_controls_multiplicity.json`, `scripts/kaggle_ingest.py`. Nothing else. `answer_bias.json`: four older models identical to the parent. Multiplicity: the 24 earlier arms unchanged, 8 Gemma arms added; summary 3/24 (P 0.104) → 4/32 (0.064), debiased 2/12 (0.109) → 3/16 (0.038). Tables: 72 added Gemma/7B rows, 7 source/footnote lines replaced, no other row touched.
3. See F7.
4. See F5, F10. No NaN, no truncated stage (controls 150 items each, adjacency 500 items each with `shortened = False`, patching 60 pairs per kind), no fallback model, self-checks 0.008–0.014 logits, STATUS OK for both.

### B. Qwen 7B behaviour (recomputed; identical to `results/answer_bias.json` to 4 decimals)

| tag | n (clusters) | yes rate | accuracy | AUROC | cross-fitted best-threshold accuracy | mean LD |
|---|---|---|---|---|---|---|
| c2c_test | 1162 (21) | 0.080 | 0.786 [0.741, 0.839] | 0.581 [0.545, 0.611] | 0.844 [0.802, 0.897] (majority 0.845) | −15.9 |
| c2c_probe | 2400 (300) | 0.085 | 0.496 [0.485, 0.508] | 0.594 [0.572, 0.618] | 0.601 [0.582, 0.620] | −15.4 |
| c2c_debiased | 2400 (1200) | 0.072 | 0.506 [0.496, 0.517] | 0.5135 [0.4899, 0.5363] | 0.5246 [0.5054, 0.5433] | −15.3 |
| cladder | 3000 (47) | 0.489 | 0.544 [0.520, 0.568] | 0.570 [0.541, 0.601] | 0.534 [0.511, 0.555] | −0.6 |

Clusters are premises (stories for CLadder). c2c_test has 21 clusters only, so its percentile intervals are rough.

1. Yes, exact match.
2. c2c_debiased: at chance (interval includes 0.5). c2c_probe and c2c_test: informative (interval above 0.5).
3. Yes: the cross-fitted best-threshold accuracy, 0.5246 [0.5054, 0.5433].
   - Cross-fit: sound. Threshold from one half of the premises, scored on the other; both items of a premise stay together. A premise split mirrors each string's label rate between halves, which biases against, not for.
   - Other models, same statistic: 0.498, 0.511, 0.497, 0.506 (0.5B, 1.5B, Gemma, 3B), all intervals include 0.5.
   - Split seed: over 200 premise splits mean 0.5223, SD 0.0049, min 0.5012; over 200 block splits mean 0.5245, min 0.5146. Always above 0.5.
   - Permutation (labels swapped within premise): p 0.0075 one-sided, 0.0095 two-sided; AUROC p 0.24, accuracy p 0.28.
   - Multiplicity: max |z| over the three pooled statistics, p = 0.023 for this model; × 5 models ≈ 0.11. On its own it does not survive counting models. It is, however, the pooled trace of the within-string effect in F1, which does (p ≤ 0.0005).
   - Mechanism: the thresholds (−21.4, −19.9) fall between the ancestor (mean −19.1) and descendant (−20.9) types, the two types where valid items score higher.
4. See F1 tables.
5. Nothing wrong found. Answer-token ids identical to qwen3b's; logit difference = logsumexp of the stored answer logits (max diff 2e-6); top-1 token is "No"/"Yes" for 100% of items and agrees with the sign of LD for 99.5–99.7%; LD values all distinct on c2c_debiased and c2c_test, one tie on c2c_probe (range −26.3 … +5.3 on the debiased pool, no saturation); mean LD −15 is the same scale as qwen3b (−15.3). CLadder has 2827 distinct LDs because the sample has 2807 distinct prompts (also in qwen3b). Item counts 2400/2400/1162/3000, uids in `data.get_dataset` order for all four tags, labels match. Cache complete: activations exported for c2c_debiased only (2400 × 29 × 2 × 3584, by design); finite at layers 0, 1, 14, 27, 28, both positions; final-layer activations through the saved readout reproduce the stored answer logits (max diff 0.054, r = 0.9999998). Precision caveat: F6.

### C. Gemma steering
Recomputed slopes, intervals, ranks, z and isotropic percentiles for the 12 Gemma rows match the JSONs and `report/tables/steering_controls.md` (all digits shown).

| run | direction | slope / s | rank of |slope| | signed rank | z | isotropic pct |
|---|---|---|---|---|---|---|
| debiased, final, L17 | gold probe | +0.475 [+0.456, +0.494] | 1/21 | 1/21 | +1.94 | 96 |
| | gold diff-means | −1.799 [−1.873, −1.728] | 2/21 | 20/21 | −1.69 | 100 |
| | model answer | +1.890 [+1.814, +1.970] | 1/21 | 1/21 | +7.80 | 100 |
| debiased, hyp, L21 | gold probe | −0.003 [−0.003, −0.002] | 15/21 | 13/21 | −0.14 | 19 |
| | gold diff-means | +0.008 [+0.006, +0.009] | 7/21 | 4/21 | +0.78 | 66 |
| | model answer | +0.011 [+0.010, +0.011] | 7/21 | 4/21 | +1.35 | 81 |
| original, final, L18 | gold probe | +0.276 [+0.244, +0.309] | 6/21 | 4/21 | +0.86 | 67 |
| | gold diff-means | +2.250 [+2.084, +2.407] | 12/21 | 5/21 | +1.27 | 100 |
| | model answer | +1.832 [+1.779, +1.890] | 1/21 | 1/21 | +6.93 | 100 |
| original, hyp, L15 | gold probe | −0.136 [−0.150, −0.124] | 5/21 | 18/21 | −0.98 | 55 |
| | gold diff-means | +0.938 [+0.871, +1.012] | 4/21 | 3/21 | +1.55 | 100 |
| | model answer | +1.092 [+1.030, +1.157] | 1/21 | 1/21 | +8.54 | 100 |

1. Real as computed, fragile as a ranking: F2.
2. F4. 4 first ranks of 32 (expected 1.52), P 0.064 independent, 0.063 with within-cluster dependence, ≤ 0.18 in the worst case; debiased 3 of 16, 0.038 / 0.037. No arm survives; none could with 20 controls.
3. No: see the last-but-one bullet of F2.
4. Yes. I regenerated all 143 directions of the debiased/final run from `resid.npy` with the script's seeds. Natural SDs of the 100 isotropic and 21 difference-of-means directions match the saved `own_sd_over_s` to 1e-7 (so the shuffled labellings are exactly those of the run); probe directions match to 0.3% (gold) and ≤ 5.5% (shuffles), the residue of non-convergence. Every shuffle keeps the label count inside each of the 48 hypothesis strings; agreement of shuffled with gold labels 0.47–0.55. Matching: unit norm, same fit, same C; natural SD gold probe 0.0029 vs control median 0.0021, gold diff-means 0.300 vs 0.261. For the content_end run the difference-of-means controls also reproduce to 1e-7.
5. Geometry at layer 17, final token (selection half; `c3.py`):
   - cos(gold probe, gold diff-means) = +0.13: the two "gold" directions are nearly orthogonal and measure different things.
   - Gold diff-means: cos −0.42 with the answer difference of means (most negative of 21, controls mean −0.05, SD 0.20), −0.54 with the within-string answer covariance direction, +0.68 with the adjacency class-mean difference of "X directly causes Y" items, +0.46 with the within-string adjacency difference overall, +0.20 with that of the other four types. 54% of it lies in the span of the two adjacency differences (parent, non-parent), 17% along the answer direction alone.
     Cause: within "directly causes" items the label is almost the adjacency (r = +0.88; −0.35 to −0.37 in three other types), and Gemma answers lower for adjacent pairs and for valid items there (within-type AUROC of the answer for the label 0.387). The gold difference of means is therefore mostly "adjacent pair in a directly-causes question", which points against Yes. Its slope is predicted by its cosine with the answer direction: −1.79 from the 20 controls' regression (r 0.93), −1.77 from a two-direction fit on the 141 non-gold directions; observed −1.80.
   - Gold probe: cosine 0.0000 with the answer difference of means (as for every probe: shuffled probes ±0.0001), +0.06 with the model-answer probe, +0.04 with the parent adjacency difference; 3.3% of it lies in the span of {gold diff-means, the two adjacency differences, model-answer probe}. The answer-direction model predicts +0.04; observed +0.475, a residual of 1.8 SDs of the shuffled probes' residuals (0.24). Its slope comes from pushing 349 natural SDs along a low-variance direction, where controls give ±0.24.
   - Natural scale: gold probe +0.0014 logits per natural SD; gold diff-means −0.54 (crossing its class gap: −0.07); model answer +0.45. Answer logit difference: mean −2.85, SD 3.51.
6. Property of the site, not a bug as far as the saved outputs can show: F3.

### D. Gemma adjacency test
All recomputed values equal the JSONs (translation effects, contrasts, control shares, erasure reductions, transfer matrices to 0, held-out R², positive control) and the Gemma rows of the five `adjacency_*.md` tables.

Observed gaps (adjacent − not, within string, all 2400 items): parent −0.199 [−0.290, −0.102], ancestor −0.069 [−0.163, +0.026], descendant +0.201 [+0.058, +0.321], collider −0.488 [−0.640, −0.330], confounder −0.659 [−0.929, −0.392]; benchmark 0.386 over the four types whose interval excludes zero. (Report half alone: −0.19, −0.00, +0.11, −0.40, −0.58.)

| site | positive control per s (controls smaller) | translation contrast (rank of absolute value among 21; z) | erasure: mean absolute gap before → after (fair controls smaller) | rank-one share | held-out R² |
|---|---|---|---|---|---|
| final, L16 | +2.01 (100%) | +0.267 = 69% of benchmark; controls −0.294 … +0.274; rank 3; z +1.54 | 0.285 → 0.111, +0.174 (95%, rank 2) | 97% | 0.957 |
| hyp, L19 | +0.008 (70%) | +0.0012; rank 2; z +2.21 | 0.285 → 0.284 (80%) | 76% | 0.785 |
| final, L8 | +0.073 (90%) | −0.023; rank 5; z −1.65 | 0.285 → 0.294 (90%) | 93% | 0.418 |
| hyp, L5 | −0.016 (15%) | +0.005; rank 8 | 0.285 → 0.282 (40%) | 75% | 0.563 |

A 5 × 5 matrix of pure noise has a rank-one share of 55% on average (95th percentile 71%), so 75–76% at the two hyp sites is close to noise; 97% at final/L16 is not.

What distinguishes Gemma: (i) the end-of-hypothesis sites are dead (F3); (ii) at the one working site the translation contrast is 69% of full mediation yet two of 20 shuffled-adjacency controls are larger in absolute value, and the erasure is second of 21: inside the control range, unlike qwen3b (100%/100%) and qwen0.5b's translation (100%); (iii) as in the Qwen models the effect follows the vector's template, not the item's: off-diagonal contrast with the source sign +0.310 [+0.304, +0.316], with the target sign +0.002 [−0.003, +0.007]; rank-one 97%; within-string answer linearly predictable from the site at R² 0.96; erasure effect = −coefficient × translation effect at r = 0.999. The class-mean difference acts as an answer shift.
Probe fits not converged: F5.

### E. Gemma patching
Recomputed and equal to JSON and table: label-changing (n = 40, 28 premises) mean D −0.054 [−0.433, +0.392], mean |D| 1.208 (hyp); −0.062 [−0.346, +0.262], 0.928 (prem). Label-preserving (n = 60, 50 premises) −0.298 [−0.655, +0.041], 1.026 (hyp; Yes/Yes 1.071, No/No 0.978); −0.218 [−0.460, +0.026], 0.724 (prem; 0.735 / 0.712). First layer with ≥ 0.5 recovery at the final token: 16 for both kinds of pair and both kinds of edit (0.06–0.09 at layer 14, 0.67–0.69 at 16, grid of stride 2), relative depth 0.62. Recovery curves of the two kinds agree within 0.03 at every layer. Comparability and the within-template difference: F8.

### F. Other things a reader should know
* Steering sites at the final token come from the earlier "main steering run" JSONs (layers 17 and 18), at content_end from a stride-2 quick selection in this run.
* The steered items are 75 valid + 75 invalid from the report half (13 of 25 blocks); intervals on slopes are over items only.
* The three `steering_controls` table columns "per natural SD" compare a gold probe with controls of smaller natural SD (footnote ‡ says so).
* The original-pool control labels are shuffled over the whole set (footnote §), so those 16 arms are not tests of validity.
* HEAD has moved to be56dec; `scripts/12_tables.py` changed after 8b8ca08 but `report/tables/` did not.

---

## Recomputed and found correct

* Byte identity of every run-created result file and every cache file with the downloads; no stray overwrite in the ingest commit; `answer_bias.json` for the four older models unchanged; the 24 earlier multiplicity arms unchanged.
* `results/answer_bias.json`, qwen7b-4bit, all four tags: n, clusters, yes rate, mean LD, accuracy, AUROC, cross-fitted and in-sample best-threshold accuracy, with intervals.
* 7B cache integrity: uid order, labels, logit difference vs stored answer logits, readout from cached activations, finiteness, answer-token ids.
* Gemma final-position activations and logits of the replaced debiased cache: bit-identical to the earlier download.
* All 12 Gemma rows of `steering_controls.md` and the four `steering_controls_gemma2b_*.json` (slopes, intervals, percentiles, ranks, natural SDs, s_unit, baseline check).
* `steering_controls_multiplicity.json`: all 32 arms (slope, z, rank, signed rank) and the three summaries; the footnote numbers of the table (1.52 / 4 / 0.064; 0.76 / 3 / 0.038).
* Control construction of the Gemma debiased runs: within-string shuffles regenerated exactly from the seeds.
* Four `adjacency_gemma2b_c2c_debiased_*.json` and the Gemma rows of the five adjacency tables (see D).
* `patching_gemma2b_same.json`, `patching_gemma2b.json`, `patching_by_template_gemma2b.json` (point estimates exact; my intervals for the within-template difference are slightly narrower with 4000 draws) and the Gemma rows of `patching_label_preserving.md`; pair lists rebuilt and matched.

## Could not check

* Anything that needs a forward pass: the slope of the converged gold probe (F2); that the hook adds the vector at the intended token and layer (supported indirectly by F3's evidence, not proven); the key-value shortcut beyond the saved self-check values; reproducibility of the 7B logits.
* The adjacency test's directions (probe, class-mean differences, LEACE read-outs) were not regenerated; the LEACE AUROCs and the per-item erasure coefficients are taken from the JSON/npz.
* The Kaggle notebooks themselves and whether the downloaded folders are complete copies of the archives named in STATUS.txt (the zips are not on disk; only the unzipped folders).
* Dependence between models in the multiplicity count (shared gold labels and items).
* Files being written by the running job (`probes_qwen7b*`, `confound_qwen7b*`, `probe_content_qwen7b*`, `logs/q7b_*`) were not opened.
* The excluded documents (RESULTS.md, README.md, report, EXPLAINER, progress notes, 14 Nov log entries) were not read, so I cannot say whether they over-read any of the above. I did see the subject line of commit 8b8ca08 and the footnotes of the report tables.

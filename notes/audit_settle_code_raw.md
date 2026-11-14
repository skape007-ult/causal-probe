# Audit of the settle run's code (ninth audit, raw report)

Sat 14 Nov 2026. Reviewed at `082a6c4`: `scripts/07d_steering_settle.py`, `scripts/07c_steering_controls.py`, the `settle`
job of `scripts/gpu_round2.py`, the two `kaggle_r3_settle_*` definitions in `scripts/make_round2_notebooks.py`, and the helpers
in `src/causal_probe/{steer,probes,data,cache,models,prompts}.py`. HEAD moved to `85fef4a` while I worked; `git diff 082a6c4 HEAD`
is empty for all reviewed files. No model was loaded. Everything marked "ran" used the stand-in (an answer that is linear in the
added vector) on the cached activations of Qwen 0.5B (layer 13), Qwen 1.5B (layer 19) and Qwen 3B (layer 25) at the last token of
the hypothesis. Scratch scripts are in
`/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit9/`
(`harness2.py`, `c_structure.py`, `d_exch.py`, `e_probe.py`, `f_calib.py`, `g_fix.py`, `h_select.py`, `i_bonf.py`).

**One mistake of my own.** Five edge-option runs at 01:21 wrote a stand-in output to
`results/steering_settle_qwen0.5b_c2c_debiased_hyp.json` in the repository (I prefixed the environment variables to `eval`, which
did not apply them; the cache variable was exported, so nothing went to `cache/`). The file was untracked, was mine alone, and I
deleted it. `git status` shows nothing of mine now. If anything read `results/` between 01:21 and 01:29 it may have seen it.

## Summary

The plumbing is right: every reported number is the number of the direction it is labelled with, split 0 reproduces the
20-control run, resume works, the runner and notebook definitions launch it correctly. The defects are in what three of the
comparisons mean.

| | Severity | What |
|---|---|---|
| F1 | critical | The "controls that keep adjacency" (difference of means) and the pooled adjacency controls share a common component the real direction has less of. With no effect at all the real direction lies outside them in a third to a half of runs. |
| F2 | major | The mirror split is not an independent draw of the direction (cosine 0.84 / 0.53 with split 0). Given how the arm was chosen, a response with nothing label-specific gives p <= 0.05 in 57% (Qwen 3B) and 28% (Qwen 1.5B) of cases. |
| F3 | major | Gold is balanced within premise, within-string shuffles are not; 69 to 82% of the string-centred variance at these sites is premise-level. Direction of the resulting bias depends on the model. |
| F4 | major | The time projection is taken from six items that are 2.1 times longer than average; it errs high by about 1.5 (measured in the earlier GPU logs). It passes at the present size by my estimate, but a failed projection ends the stage with nothing. |
| F5 | major | Direction vectors are not saved, and the partial file holds neither families nor step sizes. |
| F6-F12 | minor | Order of configurations, stale partial file, command-line trap, small-family crash, labels. |

## Findings

### F1 (critical for two labelled comparisons). `within_cell_diffmeans` and `adjacency_pooled_shuffled` are not calibrated

`scripts/07d_steering_settle.py` lines 120-128 and 132-138; the reading in the docstring lines 16-22 and in the decision rule
("z below 1 against them: an effect of adjacency"; "exceeds them: validity beyond adjacency").

What happens. A within-cell labelling keeps every cell's label rate, so every control direction is
`unit(common + noise_i)`, where `common` is the class-mean difference under the cell rates. The gold direction is
`unit(common + its own within-cell part)`. The controls are almost the same vector; the gold direction is not:

| site | cosine among within-cell controls | cosine gold with them | share of squared norm in `common`: controls / gold |
|---|---|---|---|
| Qwen 3B L25, split 0 / 1 | +0.85 / +0.89 | +0.61 / +0.62 | 0.86 / 0.41, 0.90 / 0.42 |
| Qwen 1.5B L19, split 0 / 1 | +0.87 / +0.92 | +0.63 / +0.68 | 0.89 / 0.43, 0.93 / 0.50 |

All vectors are steered at the same norm, so the gold direction takes a step about twice as large along its own part as a
control does along its noise. Any response vector then puts gold outside the family. The same holds for
`adjacency_pooled_diffmeans` against `adjacency_pooled_shuffled`: shuffling adjacency within string keeps each string's adjacency
rate, so the shuffles share the between-string component (cosine among shuffles +0.89 to +0.93, real with shuffles +0.36 to +0.56).

How established (ran).
- Exact computation for an isotropic random linear response, 4,000 draws, directions from the script's own `build`
  (`f_calib.py`, `g_fix.py`). SD of z of the real direction against its controls, which should be 1:
  within-cell difference of means 1.96 (Qwen 3B split 0, n = 100), 2.3 to 2.5 (split 1, n = 20), 2.05 / 2.47 (Qwen 1.5B);
  P(|z| > 2) 0.33 to 0.45; P(one-sided p <= 0.05) 0.20 to 0.25 on each side. Pooled adjacency: SD of z 3.0 to 3.6,
  P(|z| > 2) 0.51 to 0.59.
- Through the script itself, Qwen 0.5B, 20 stand-in seeds, no planted effect: gold against within-cell controls in split 1 had
  rank 1 or 21 of 21 in 11 of 20 runs (2 expected) and |z| > 2 in 13 of 20; in split 0 (n = 100) |z| > 2 in 8 of 20. Pooled
  adjacency against its shuffles: |z| > 2 in 13 of 20 in each split, up to z = +13.
- Planted effects (Qwen 0.5B, `PLANT=...` in `harness2.py`). A response to the within-string adjacency direction alone made the
  gold direction beat the within-cell controls at z +5.2 to +5.7 (rank 1 of 101, split 0) and +12.6 to +13.2 (rank 1 of 21,
  split 1). They should not be beaten there. With the adjacency direction made orthogonal to the validity-within-cell direction
  the result was z -2.0 (rank 99 of 101) and -0.3 at strength 0.5, and +0.7 and +3.5 (rank 1 of 21) at strength 0.15: not
  interpretable either way. A planted validity-beyond-adjacency response was detected (z +6 to +11), but so is nothing at all.

Not affected: `within_string_*` (cosine among controls 0.00; SD of z 1.03), `adjacency_parent_shuffled` (SD of z 1.01 to 1.17),
and `within_cell_probe` in the probe arm (cosine among controls +0.38, gold with them +0.36; SD of z 1.05 to 1.10).

Smallest fix. Steer the part that the shuffle actually randomises, so that real and control vectors are built the same way and
are mutually uncorrelated: with `Xc = Xf - Xf.mean(0)`, `ybar` = cell mean of the label per item, `tbar` = string mean of
adjacency per item,
- `gold_beyond_adjacency = unit((yf - ybar) @ Xc)` against `within_cell_residual = unit((yc - ybar) @ Xc)` for the same `yc`;
- `adjacency_within_string = unit((tf - tbar) @ Xc)` against `unit((ts - tbar) @ Xc)`.
Checked (`g_fix.py`): cosine among controls 0.00, SD of z 0.91 to 1.01 at both sites and both splits. Keep the present families if
wanted, but label them as sharing a component and do not read z or rank from them. Cost: about 1 + 100 vectors in split 0 and
1 + 20 in split 1 per family.

### F2 (major). The mirror split re-estimates the same direction; the p-value does not account for how the arm was chosen

Lines 27-30 (primary test), 342-349; decision rule in `notes/log.md` ("p <= 0.05: the arm replicates on an independent draw of
the direction").

What happens. The gold direction fitted on the two disjoint halves is nearly one vector: cosine +0.839 (Qwen 3B, difference of
means), +0.533 (Qwen 1.5B, probe), +0.805 / +0.637 (Qwen 0.5B). The controls of split 1 are new, the direction is not. The arm
was chosen because its slope ranked first of 21 in split 0, one of 24 comparisons. Whatever made that slope large, if it is a
property of the direction (which any reproducible direction has), it is still there in split 1. The sign is also taken from
split 0, so the one-sided test is not one-sided in an independent sign.

How established (ran, `h_select.py`, `i_bonf.py`). Response = isotropic random linear vector, which has nothing to do with the
label; 200,000 draws; directions and controls from `build`:

| | Qwen 3B, gold_diffmeans | Qwen 1.5B, gold_probe |
|---|---|---|
| P(p <= 0.05), sign from split 0, no selection | 0.104 | 0.087 |
| given first of 21 in abs slope in split 0: P(p <= 0.05) | 0.570 | 0.283 |
| given that: P(p > 0.20), "does not replicate" | 0.083 | 0.363 |
| given that: median p; P(same sign) | 0.040; 0.998 | 0.129; 0.911 |
| given that, 500 controls, slope beyond all 500 (p = 1/501 <= 0.05/24) | 0.054 | 0.024 |

So the test separates "top rank from estimation noise in one draw" from "this direction's slope is reproducibly in the tail".
It does not separate the second from "one of 24 fixed directions is expected in the tail". Under the label-free account the
pre-registered rule says "replicates" more often than not for Qwen 3B.

Smallest fix, before the run. (a) Amend the log entry now: p <= 0.05 means the slope of this direction is reproducible, not that
the choice among 24 is accounted for. (b) To have a selection-adjusted reading, run split 1 with 500 within-string controls and
state the criterion "beyond all 500". `--n-shuf` cannot do this alone: it also sets split 0 and its within-cell family
(line 251-252); add a separate count for `k == 1 and l == layer`. Cost: +800 vectors per item in split 1 = +120,000
vector-items (+43% for Qwen 3B, about +14 min; Qwen 1.5B about +8 min plus 400 probe fits), which does not fit the present
75-minute budget together with F4.

### F3 (major). Gold is balanced within premise; the within-string controls are not

Lines 113-119 (`shuffle_within(yf, sf, rng)`); `data.c2c_debiased_pool`.

What happens. Every premise contributes exactly one valid and one invalid item (ran: 1,200 premises, 2 items each, one of each
label). In the gold difference of class means the premise-level part of the activations cancels exactly. A within-string
shuffle gives both items of a premise the same label about half the time, so its direction carries premise-level variance. At
these sites that is most of what is left after removing string means:

| site | variance left after string means (share of s squared) | of that, premise-level |
|---|---|---|
| Qwen 0.5B L13 | 0.12 | 0.81 to 0.82 |
| Qwen 1.5B L19 | 0.12 to 0.13 | 0.72 |
| Qwen 3B L25 | 0.10 | 0.69 |

Consequences measured (`d_exch.py`, `e_probe.py`, `f_calib.py`):
- Raw length of the class-mean difference over s, Qwen 3B: gold 0.030 / 0.037; within-string shuffles median 0.018; labellings
  with gold's structure (one valid and one invalid per premise, balanced within string, otherwise random) 0.014 to 0.015.
- Projection onto the first premise-level principal component: within-string controls rms 0.40 to 0.57, gold-structured nulls
  0.07 to 0.15. A sixth to a third of a control direction is that one component.
- Probes: SD of premise means along the direction over SD of within-premise half-differences: gold 0.46 to 0.63, shuffled
  probes 1.17 to 1.38. Natural SD along the direction: gold probe 0.0070 (3B) and 0.0079 (1.5B), shuffled probes 0.0042 and
  0.0045, both in the stored 20-control files.
- Primary test with label-free responses that are not isotropic, P(one-sided p <= 0.05) per side, nominal 0.05:
  Qwen 3B difference of means: 0.03 (response in premise-level directions), 0.18 (within-premise directions), 0.03 (all
  activation variance). Qwen 1.5B probe: 0.08, 0.24, 0.17. For an isotropic response it is 0.05.

So the test is exact for its stated null (gold is one of the within-string shuffles) and that null is false in a second known
respect besides adjacency. Which way it errs depends on where the model's response lies, which I cannot see without the model.
The one piece of real evidence: in the stored 20-control runs the rms slope is almost the same for isotropic, shuffled-probe and
shuffled-difference directions (Qwen 1.5B 0.078 / 0.084 / 0.096; Qwen 3B 0.20 / 0.26 / 0.37) although their natural SDs differ
30-fold, which is closer to the isotropic case. In the same files the Qwen 1.5B gold probe ranks 1 of 21 in abs slope and 3 of 21
in abs slope per natural SD; the Qwen 3B gold difference of means ranks 1 of 21 in both.

Smallest fix. Add one family for the arm's kind in splits 0 and 1: 100 labellings with gold's structure, obtained by reversing
random directed cycles of the graph "string of the valid item -> string of the invalid item" over premises (20 lines; working
version in `eul.py`; it keeps both balances, correlation with gold +0.002, SD 0.047). It is calibrated for isotropic and
all-variance responses for the difference of means (0.04 to 0.05), not for every response. Report the rank of slope / natural SD
next to the rank of slope (this one can be computed afterwards from the saved file).

### F4 (major: a way to lose the session; by my estimate not triggered at the present size). Time projection

Lines 294-300; `gpu_round2.py` lines 238-240.

What happens. The projection uses the first six items of the first configuration. Items are sorted by index; those six have
1,643, 816, 383, 1,113, 1,648 and 702 prompt tokens (mean 1,051) against 489 for the plan as a whole (ran, from `meta.json`).
The cost per vector grows with prompt length (the key-value cache of the prefix is copied and repeated for every sub-batch).
The earlier GPU logs show the size of the error on the same first items: Qwen 3B projected 7.0 min for 145 items, the loop took
about 5.0 (stage 6.9 min less 88 s of fitting and the model load); Qwen 1.5B 4.0 projected, about 2.7 taken. Factor about 1.5,
too high. On a failed projection the script exits (line 300): the cache stage (17 min) and the session are spent and no output
exists.

Estimates (reasoned from those logs; not measured):

| | Qwen 3B, gold_diffmeans | Qwen 1.5B, gold_probe |
|---|---|---|
| vectors per item, by configuration | 739, 371, 6 x 127 | 779, 411, 6 x 167 |
| vector-items in the plan | 280,800 | 328,800 |
| logistic fits before the model loads | 176 | 576 |
| fit time on Kaggle (43 fits plus overhead took 88 s / 73 s there) | 5 to 6 min | 13 to 16 min |
| steering, at 7.0 / 3.7 ms per vector-item | about 33 min | about 20 min |
| projection the script will print | about 45 min | about 30 min |
| allowed (0.85 x minutes left) | about 56 min | about 48 min |

It passes with about 10 and 18 minutes to spare. It would not pass with the additions of F1 to F3, or on a slower machine.
Exceeding the runner's 88 minutes after the projection passed would need the run to be 1.3 to 1.4 times slower than projected
while the projection is 1.5 times too high: not plausible. `total_vi` is right (ran: sum of vectors x items over the plan).
Probe fitting is not a risk in itself: on the laptop 0.03 s (d = 896), 0.05 s (d = 1,536), 0.08 s (d = 2,048, Qwen 3B layer 25,
1,152 to 1,248 items), 180 to 305 iterations, none near 2,000. Memory of the plan is under 50 MB for d = 2,048 (15 MB of
vectors, 7 MB of directions, 20 MB per transient copy of a layer).

Smallest fix. Weight the projection by tokens: cost per (token x vector) from the items done, times the sum of
`meta["n_tokens"][i] * len(vecs)` over what remains. On a failed projection drop configurations from the end of the plan
(extra layers, then random splits) instead of exiting. Raise `--max-minutes` and the stage limit together if families are added.

### F5 (major: cannot be recovered later). The direction vectors are not saved; the partial file is not self-describing

Lines 302-304 (partial: `out_*`, `idx_*` only), 340-341 and 359 (raw file: outputs, items, families, natural SDs, steps).

What happens. With `--no-resid` the run's activations stay on the GPU machine, and Kaggle may give a different GPU than the run
whose activations are on the laptop, so the directions cannot be assumed reproducible here. Without them none of this can be
done afterwards: cosine of the gold direction between splits (F2), cosines among controls (F1), the premise-level share of each
direction (F3), cosine of gold with the adjacency direction, the class gap along control directions. If the stage is killed by
the runner's limit, only the partial file leaves the machine (it is exported: the name starts with `steering`), and it has no
families, steps, natural SDs or `s_unit`; families and steps can be rebuilt from the counts, the rest cannot.

Smallest fix. Right after the plan is built and before the model loads, write `steering_settle_hyp_plan.npz` with, per
configuration, `D` (float16 is enough), `fam`, `alphas`, `own_sd`, `gap_s`, `s_unit`, `idx`. For Qwen 3B that is 848
directions x 2,048 = 3.5 MB in float16. It is exported by the existing rule.

### F6 (minor). The primary configuration runs second, behind the largest one
Line 242. Split 0 at the site is 39% of all vector-items and runs first; nothing is saved inside a configuration. A session that
dies in the first 13 minutes of steering (Qwen 3B) leaves nothing, and the primary test is saved only after 58% of the work.
Fix: put `(1, layer)` first.

### F7 (minor). A stale partial file is accepted if shapes and items match
Line 284. Ran: a run killed in its third configuration, then a run with another stand-in (standing for other code, another C,
other activations) took the first two configurations from the partial file without complaint, and the outputs disagreed with
the directions. A partial file from the other arm is not accepted (all shapes differ: 127 against 167 vectors and so on). Not
reachable in a fresh Kaggle session. Fix: store a checksum of `vecs` per configuration and compare it.

### F8 (minor). `--layer-offsets -2,2` cannot be given on the command line
Line 213. Ran: argparse answers "expected one argument" because the value starts with a minus; `--layer-offsets=-2,2` works.
The runner does not pass the option, so the default applies. Offsets that leave 1..L are dropped (ran: layer 13 of 24 with
-13, -12, 11, 12 keeps layers 1 and 24). Offset 0 repeats the split-0 configuration; the copy is taken from memory and listed
twice.

### F9 (minor). A family with one member crashes the summary after all GPU work
Lines 352-357. Ran: `--n-other 1 --n-splits 3` ends with a KeyError in `arm_across_splits_and_layers_descriptive`. The partial
file survives (it is deleted only after the results are written), so nothing is lost, but no JSON is written. The runner's
smallest value is 4 (tiny mode).

### F10 (minor). Split 1 has 20 within-cell controls
Lines 251-252. The second question of the decision rule can be asked of the primary split only against 20 controls (smallest
possible p 0.048). Separate from F1.

### F11 (minor, label). `class_gap_along_direction_over_s` is always the gold-label gap
Line 258. For `adjacency_*` and `model_out` it is the gap between valid and invalid items along that direction, not the gap
between adjacent and non-adjacent items or between the model's answers. Correct for the gold directions. It is computed on the
whole steered half (not the 150 items) with a direction fitted on the other half, which is the right way round.

### F12 (minor). The cache stage also builds the `c2c_probe` cache, which the settle job does not use
`gpu_round2.py` line 226. Part of the 17 (Qwen 3B) or 8 (Qwen 1.5B) minutes.

## Answers to the questions not covered above

Q1. Splits (ran, `c_structure.py`): fitting and steered halves share no block in any split; split 1 is the exact complement of
`data.selection_mask`; random splits are rejected if equal to or the mirror of an earlier one (checked up to 12); fitting
halves have 12 blocks (13 in split 1) and random splits share 5 to 8 fitting blocks with split 0. Steered items are 75 valid and
75 invalid, not balanced by string or template; real and control directions use the same items. Fitted on the fitting half
only: all directions, the probe standardisation, `s_unit`, natural SDs. Uses all items: the median behind the model-answer label
(positive control only). Uses the steered half: the class gap (descriptive, out of sample). The site was chosen by
cross-validated probe accuracy on the selection half, which split 1 steers: no steering outcome entered that choice and the same
layer and C serve the controls, so it cannot bias the rank; it can make the class gap in split 1 slightly optimistic. The
dependence that matters between the splits is F2.

Q2. `shuffle_within` does what both labels say (ran: counts per cell preserved in every draw; within-cell draws keep every
string half valid and keep the count of valid items in every string x adjacency cell). No cell has a single item. 23 to 25 of
the 96 cells have one class and hold 19 to 22% of the fitting items, which never change label; 3 to 6 of 48 strings are frozen
entirely; in the "parent" template 53 to 67% of items are frozen. A within-cell shuffle changes 32% of labels in expectation
against 50% within string. No class is empty in any split, template or family (parent items on a fitting half: 185 to 199
adjacent, 139 to 165 not; 0 of 200 adjacency shuffles per split gave an empty class). Systematic differences from the real
direction other than the labelling: F1 (shared component) and F3 (premise balance, natural spread).

Q3. Ran `harness2.py`: directions captured from `build` were recomputed independently from the docstring for every family
(first six of each probe family), and every reported number was recomputed from `s_unit x direction . w` with item-dependent
gains: raw columns, slopes, families, `versus`, signed and absolute ranks, `one_sided_p`, `by_template`, smaller steps and their
ranks, class gap, natural SD, share of items with positive slope, the primary test and the across-splits list. All checks passed
for both arms, full and non-full configurations, and the runner's tiny configuration.

Q4. Ran: split 0 uses the same 150 items and the same `s_unit` as the stored 20-control file; natural SDs of the three real
directions, the 20 isotropic directions and the first 20 directions of both shuffled families agree to 3e-8; the 21st differs
from all of them; the two arms agree on every labelling they share. The gold probe at 300 iterations equals the one at 2,000
(cosine 1.000000) at all three sites, so the sign from the 20-control run refers to the same direction. `expected_sign` reads
`real[arm].slope_per_s.est`: +1 for Qwen 3B (+0.738), -1 for Qwen 1.5B (-0.251). An earlier slope of exactly zero gives sign 0
and p = 1. A missing file ends the script at once (after the cache stage); the three files are tracked in git.

Q5. Resume: killed in the third configuration and restarted, the result and the raw file are identical to an uninterrupted run
(ran). Tiny mode passes `--n 10 --n-shuf 6 --n-other 4 --n-splits 3 --max-minutes 20 --allow-new-site`; it runs end to end and
passes the verification (ran). The runner's full command line is accepted (ran to the model load). Both notebook definitions
build in memory, all code cells compile, and the runner's parser accepts their extra arguments (ran). Device and dtype follow
the path of the 20-control script (read).

Q6. `one_sided_p` is the p-value described; ties count against the real direction; the smallest value is 1/101 (ran). What it
tests and what I would change: F2 and F3. What to add now: F5, and the families of F1 and F3.

## Verified as correct

- Layout of `alphas`, `cols`, `plus`, `minus` and the smaller-step columns; all summaries (Q3).
- Reproduction of the 20-control run in split 0, independent of the arm (Q4).
- Splits, balance of steered items, no leakage of the steered half into anything fitted (Q1).
- Both shuffles preserve what they claim; no empty class, no NaN direction (Q2).
- The primary p-value as a Monte Carlo test of its stated null: uniform for an isotropic response (0.052 / 0.054 in the
  exact computation; 2 of 20 seeds at or below 0.05 through the script); a planted validity response gives p = 0.010, the
  smallest possible, for both arms at both strengths tried.
- Resume, tiny mode, runner arguments, notebook definitions, export of the raw and partial files with `--no-resid`.

## Not checked

- Anything on a real model or on CUDA: actual time per vector, half-precision behaviour, GPU memory for 16 vectors on the
  longest prompts (1,675 tokens; items of about 1,650 tokens did run in the 20-control job at the same sub-batch).
- Fit times on the Kaggle CPU (estimated from two log lines).
- Where the real model's response lies, hence the direction and size of the bias in F3.
- Non-linear responses: the stand-in is linear, so the smaller-step comparison was checked for indexing only.
- The generated `.ipynb` files (not written yet) and the ingest of the new result files on the laptop.

# Number tables from `finetune/results/behaviour_synthetic_planted.json`

Mode: synthetic_planted; within-string implementation: reference; premise resamples: 2000; written 2026-11-20T22:12:38Z.

## Registered tests (Holm over the eleven; E2c is a control, not in the family)

| Test | Comparison | D | 95% interval | p | Holm p | Seed SD (x / y) | Premise SE | Outlying seeds | Claimed |
|---|---|---|---|---|---|---|---|---|---|
| E2c | S(Bswap) - 0.5, POOL-LB (control) | +0.004 | [-0.060, +0.067] | 0.8331 | n/a | +0.022 | 0.007 | none | inconclusive: interval not inside 0.5 +/- 0.05 and not excluding 0.5 (expected with 3 seeds; Amendment 1, item 2) |
| E2a | S(B) - S(A), POOL-LB | +0.309 | [+0.290, +0.329] | 0.0000 | 0.0000 | +0.004 / +0.011 | 0.006 | none | above |
| E2b | S(B) - S(A48), POOL-LB | +0.137 | [+0.121, +0.152] | 0.0000 | 0.0001 | +0.004 / +0.004 | 0.005 | none | above |
| E3 | S(A) - S(C), ID48 | +0.125 | [+0.086, +0.163] | 0.0005 | 0.0029 | +0.015 / +0.014 | 0.011 | none | above |
| E4a | S(original) - S(renamed Z..U), A, ID48 | +0.069 | [+0.039, +0.100] | 0.0033 | 0.0119 | +0.013 / n/a | 0.009 | none | above |
| E4b | S(keep order) - S(reverse), A, ID48 | +0.070 | [+0.041, +0.098] | 0.0025 | 0.0119 | +0.009 / n/a | 0.010 | none | above |
| E5a_A | S in cells - 0.5, A | +0.134 | [+0.107, +0.161] | 0.0002 | 0.0012 | +0.019 / n/a | 0.005 | none | above |
| E5a_B | S in cells - 0.5, B | +0.455 | [+0.448, +0.462] | 0.0000 | 0.0000 | +0.003 / n/a | 0.002 | none | above |
| E5b_A | S - memorisation ref., A | -0.060 | [-0.095, -0.025] | 0.0087 | 0.0174 | +0.011 / n/a | 0.012 | none | below |
| E5b_B | S - memorisation ref., B | +0.055 | [+0.033, +0.078] | 0.0024 | 0.0119 | +0.004 / n/a | 0.008 | none | above |
| E5c_A | S - best text ref., A | -0.202 | [-0.230, -0.175] | 0.0000 | 0.0003 | +0.011 / n/a | 0.009 | none | below |
| E5c_B | S - best text ref., B | -0.003 | [-0.019, +0.013] | 0.6610 | 0.6610 | +0.004 / n/a | 0.005 | none | not shown |

Power notes: none

## E1 (gate, A)

S on ID48, mean over seeds: 0.613 (threshold 0.8); per seed 0.633, 0.622, 0.611, 0.592, 0.607
Pooled AUROC on ID, mean: 0.661 (threshold 0.922). Memorisation reference of A on ID48: 0.763. Passed: False.

## Rungs

A: L1  
B: L2

## S per model

| Model | POOL-LB | ID48 |
|---|---|---|
| A48_s0 | 0.819 | 0.764 |
| A48_s1 | 0.819 | 0.771 |
| A48_s2 | 0.812 | 0.759 |
| A_s0 | 0.646 | 0.633 |
| A_s1 | 0.654 | 0.622 |
| A_s2 | 0.651 | 0.611 |
| A_s3 | 0.628 | 0.592 |
| A_s4 | 0.638 | 0.607 |
| B_s0 | 0.952 | 0.871 |
| B_s1 | 0.959 | 0.872 |
| B_s2 | 0.953 | 0.890 |
| B_s3 | 0.948 | 0.884 |
| B_s4 | 0.953 | 0.882 |
| Bswap_s0 | 0.524 | 0.528 |
| Bswap_s1 | 0.480 | 0.476 |
| Bswap_s2 | 0.506 | 0.518 |
| C_s0 | 0.525 | 0.504 |
| C_s1 | 0.470 | 0.479 |
| C_s2 | 0.476 | 0.482 |

## E5 descriptive, A: near and far halves of POOL-LB

| Half | Items | S per seed | Memorisation ref. |
|---|---|---|---|
| near | 2014 | 0.640, 0.647, 0.650, 0.655, 0.644 | 0.706 |
| far | 2692 | 0.651, 0.653, 0.645, 0.610, 0.630 | 0.717 |

E5b(A) against the reviewer's pair-only variant (Amendment 1): D -0.126 [-0.159, -0.093], p 0.0004

## E5 descriptive, B: near and far halves of POOL-LB

| Half | Items | S per seed | Memorisation ref. |
|---|---|---|---|
| near | 2266 | 0.945, 0.949, 0.942, 0.941, 0.942 | 0.898 |
| far | 2440 | 0.936, 0.954, 0.940, 0.946, 0.943 | 0.888 |

E5b(B) against the reviewer's pair-only variant (Amendment 1): D +0.040 [+0.019, +0.061], p 0.0060

## Reference checks

```
{
 "mem_A": {
  "member": "queried|exp|4.0",
  "S_registered": 0.7033511180729205,
  "S_recomputed": 0.7033511180729205
 },
 "mem_variant_A": {
  "member": "queried_pair|exp|4.0",
  "S": 0.7692908163154538
 },
 "text_A": {
  "best": "tuned_stack",
  "S_stored": 0.8457706791618416,
  "S_recomputed": 0.8457706791618416
 },
 "near_A": {
  "median_distance": 4.0,
  "n_near_items": 2014
 },
 "mem_B": {
  "member": "queried|exp|4.0",
  "S_registered": 0.897530343488884,
  "S_recomputed": 0.897530343488884
 },
 "mem_variant_B": {
  "member": "queried_pair|exp|4.0",
  "S": 0.9125734118741435
 },
 "text_B": {
  "best": "tuned_stack",
  "S_stored": 0.9556594610693343,
  "S_recomputed": 0.9556594610693343
 },
 "near_B": {
  "median_distance": 5.0,
  "n_near_items": 2266
 },
 "mem_A48": {
  "member": "queried|exp|4.0",
  "S_registered": 0.7567501853449856,
  "S_recomputed": 0.7567501853449856
 },
 "mem_variant_A48": {
  "member": "queried_pair|exp|4.0",
  "S": 0.815590228012289
 },
 "text_A48": {
  "best": "tuned_stack",
  "S_stored": 0.885086192464024,
  "S_recomputed": 0.885086192464024
 },
 "near_A48": {
  "median_distance": 4.0,
  "n_near_items": 2014
 }
}
```

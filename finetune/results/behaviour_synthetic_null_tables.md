# Number tables from `finetune/results/behaviour_synthetic_null.json`

Mode: synthetic_null; within-string implementation: reference; premise resamples: 2000; written 2026-11-20T22:16:26Z.

## Registered tests (Holm over the eleven; E2c is a control, not in the family)

| Test | Comparison | D | 95% interval | p | Holm p | Seed SD (x / y) | Premise SE | Outlying seeds | Claimed |
|---|---|---|---|---|---|---|---|---|---|
| E2c | S(Bswap) - 0.5, POOL-LB (control) | +0.004 | [-0.060, +0.067] | 0.8331 | n/a | +0.022 | 0.007 | none | inconclusive: interval not inside 0.5 +/- 0.05 and not excluding 0.5 (expected with 3 seeds; Amendment 1, item 2) |
| E2a | S(B) - S(A), POOL-LB | -0.002 | [-0.022, +0.019] | 0.8311 | 1.0000 | +0.012 / +0.010 | 0.006 | none | not shown |
| E2b | S(B) - S(A48), POOL-LB | -0.014 | [-0.035, +0.007] | 0.1542 | 0.7710 | +0.012 / +0.004 | 0.006 | none | not shown |
| E3 | S(A) - S(C), ID48 | +0.015 | [-0.012, +0.043] | 0.2154 | 0.8616 | +0.013 / +0.008 | 0.008 | none | not shown |
| E4a | S(original) - S(renamed Z..U), A, ID48 | +0.005 | [-0.021, +0.030] | 0.6289 | 1.0000 | +0.012 / n/a | 0.007 | none | not shown |
| E4b | S(keep order) - S(reverse), A, ID48 | +0.003 | [-0.022, +0.027] | 0.7840 | 1.0000 | +0.008 / n/a | 0.008 | none | not shown |
| E5a_A | S in cells - 0.5, A | +0.298 | [+0.275, +0.321] | 0.0000 | 0.0000 | +0.016 / n/a | 0.004 | none | above |
| E5a_B | S in cells - 0.5, B | +0.307 | [+0.292, +0.322] | 0.0000 | 0.0000 | +0.008 / n/a | 0.004 | none | above |
| E5b_A | S - memorisation ref., A | +0.101 | [+0.068, +0.134] | 0.0011 | 0.0077 | +0.010 / n/a | 0.011 | none | above |
| E5b_B | S - memorisation ref., B | -0.095 | [-0.123, -0.067] | 0.0007 | 0.0056 | +0.012 / n/a | 0.009 | none | below |
| E5c_A | S - best text ref., A | -0.042 | [-0.068, -0.016] | 0.0114 | 0.0685 | +0.010 / n/a | 0.008 | none | not shown |
| E5c_B | S - best text ref., B | -0.153 | [-0.177, -0.130] | 0.0001 | 0.0005 | +0.012 / n/a | 0.006 | none | below |

Power notes: none

## E1 (gate, A)

S on ID48, mean over seeds: 0.746 (threshold 0.8); per seed 0.766, 0.752, 0.740, 0.731, 0.740
Pooled AUROC on ID, mean: 0.745 (threshold 0.922). Memorisation reference of A on ID48: 0.763. Passed: False.

## Rungs

A: L2  
B: L1

## S per model

| Model | POOL-LB | ID48 |
|---|---|---|
| A48_s0 | 0.819 | 0.764 |
| A48_s1 | 0.819 | 0.771 |
| A48_s2 | 0.812 | 0.759 |
| A_s0 | 0.812 | 0.766 |
| A_s1 | 0.811 | 0.752 |
| A_s2 | 0.808 | 0.740 |
| A_s3 | 0.789 | 0.731 |
| A_s4 | 0.801 | 0.740 |
| B_s0 | 0.788 | 0.711 |
| B_s1 | 0.817 | 0.723 |
| B_s2 | 0.800 | 0.759 |
| B_s3 | 0.794 | 0.739 |
| B_s4 | 0.812 | 0.740 |
| Bswap_s0 | 0.524 | 0.528 |
| Bswap_s1 | 0.480 | 0.476 |
| Bswap_s2 | 0.506 | 0.518 |
| C_s0 | 0.820 | 0.740 |
| C_s1 | 0.774 | 0.728 |
| C_s2 | 0.784 | 0.723 |

## E5 descriptive, A: near and far halves of POOL-LB

| Half | Items | S per seed | Memorisation ref. |
|---|---|---|---|
| near | 2014 | 0.802, 0.795, 0.795, 0.798, 0.799 | 0.706 |
| far | 2692 | 0.803, 0.810, 0.791, 0.778, 0.790 | 0.717 |

E5b(A) against the reviewer's pair-only variant (Amendment 1): D +0.035 [+0.004, +0.066], p 0.0355

## E5 descriptive, B: near and far halves of POOL-LB

| Half | Items | S per seed | Memorisation ref. |
|---|---|---|---|
| near | 2266 | 0.792, 0.815, 0.802, 0.792, 0.817 | 0.898 |
| far | 2440 | 0.771, 0.793, 0.769, 0.781, 0.807 | 0.888 |

E5b(B) against the reviewer's pair-only variant (Amendment 1): D -0.110 [-0.137, -0.083], p 0.0003

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

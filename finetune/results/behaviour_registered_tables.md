# Number tables from `finetune/results/behaviour_registered.json`

Mode: registered; within-string implementation: reference; premise resamples: 2000; written 2026-11-21T14:55:21Z.

## Registered tests (Holm over the eleven; E2c is a control, not in the family)

| Test | Comparison | D | 95% interval | p | Holm p | Seed SD (x / y) | Premise SE | Outlying seeds | Claimed |
|---|---|---|---|---|---|---|---|---|---|
| E2c | S(Bswap) - 0.5, POOL-LB (control) | -0.004 | [-0.034, +0.026] | 0.6148 | n/a | +0.004 | 0.007 | none | pass: interval inside 0.5 +/- 0.05 |
| E2a | S(B) - S(A), POOL-LB | +0.138 | [+0.114, +0.161] | 0.0000 | 0.0000 | +0.008 / +0.013 | 0.007 | none | above |
| E2b | S(B) - S(A48), POOL-LB | +0.077 | [+0.024, +0.130] | 0.0225 | 0.0676 | +0.008 / +0.021 | 0.006 | none | not shown |
| E3 | S(A) - S(C), ID48 | +0.328 | [+0.270, +0.387] | 0.0000 | 0.0001 | +0.013 / +0.007 | 0.023 | none | above |
| E4a | S(original) - S(renamed Z..U), A, ID48 | +0.074 | [+0.046, +0.101] | 0.0018 | 0.0124 | +0.012 / n/a | 0.008 | none | above |
| E4b | S(keep order) - S(reverse), A, ID48 | +0.001 | [-0.016, +0.017] | 0.9214 | 0.9214 | +0.006 / n/a | 0.005 | none | not shown |
| E5a_A | S in cells - 0.5, A | +0.179 | [+0.137, +0.221] | 0.0003 | 0.0023 | +0.027 / n/a | 0.009 | none | above |
| E5a_B | S in cells - 0.5, B | +0.366 | [+0.334, +0.399] | 0.0000 | 0.0001 | +0.019 / n/a | 0.008 | none | above |
| E5b_A | S - memorisation ref., A | +0.075 | [+0.041, +0.109] | 0.0036 | 0.0177 | +0.013 / n/a | 0.011 | none | above |
| E5b_B | S - memorisation ref., B | +0.018 | [-0.006, +0.043] | 0.1044 | 0.2088 | +0.008 / n/a | 0.008 | none | not shown |
| E5c_A | S - best text ref., A | -0.068 | [-0.095, -0.041] | 0.0022 | 0.0132 | +0.013 / n/a | 0.008 | none | below |
| E5c_B | S - best text ref., B | -0.040 | [-0.058, -0.022] | 0.0035 | 0.0177 | +0.008 / n/a | 0.005 | none | below |

Power notes: none

## E1 (gate, A)

S on ID48, mean over seeds: 0.846 (threshold 0.8); per seed 0.831, 0.855, 0.831, 0.856, 0.854
Pooled AUROC on ID, mean: 0.980 (threshold 0.922). Memorisation reference of A on ID48: 0.763. Passed: True.

## Rungs

A: L2 (claimed above the registered memorisation reference but not above the reviewer's pair-only variant; Amendment 1)  
B: L1

## S per model

| Model | POOL-LB | ID48 |
|---|---|---|
| A48_s0 | 0.816 | 0.886 |
| A48_s1 | 0.856 | 0.882 |
| A48_s2 | 0.846 | 0.888 |
| A_s0 | 0.794 | 0.831 |
| A_s1 | 0.784 | 0.855 |
| A_s2 | 0.760 | 0.831 |
| A_s3 | 0.771 | 0.856 |
| A_s4 | 0.781 | 0.854 |
| B_s0 | 0.905 | 0.752 |
| B_s1 | 0.920 | 0.765 |
| B_s2 | 0.924 | 0.770 |
| B_s3 | 0.921 | 0.765 |
| B_s4 | 0.910 | 0.769 |
| Bswap_s0 | 0.491 | 0.511 |
| Bswap_s1 | 0.499 | 0.506 |
| Bswap_s2 | 0.497 | 0.493 |
| Bswap_s3 (extra seed, not in any registered test) | 0.501 | 0.506 |
| Bswap_s4 (extra seed, not in any registered test) | 0.514 | 0.512 |
| C_s0 | 0.534 | 0.510 |
| C_s1 | 0.519 | 0.518 |
| C_s2 | 0.519 | 0.524 |

## E3 descriptive: text references fitted on A's training set, S on ID48

| Reference | S |
|---|---|
| lookup | 0.628 |
| hand_parse | 0.777 |
| tuned_stack | 0.831 |
| premise_counts_logistic | 0.590 |
| premise_counts_boosting | 0.584 |

Share of A's pooled AUROC on ID available without the premise, (AUROC_C - 0.5)/(AUROC_A - 0.5): 0.889 [0.862, 0.914]

## E5 descriptive, A48 (intervals only, no test)

| Quantity | D | 95% interval | Seed SD | Note |
|---|---|---|---|---|
| S_cells_minus_half | +0.271 | [+0.174, +0.367] | 0.036 | seed SD 0.036 > 0.03 |
| S_minus_memorisation | +0.082 | [+0.016, +0.149] | 0.021 |  |
| S_minus_best_text | -0.046 | [-0.105, +0.014] | 0.021 |  |

## E5 descriptive, A: near and far halves of POOL-LB

| Half | Items | S per seed | Memorisation ref. |
|---|---|---|---|
| near | 2014 | 0.778, 0.773, 0.739, 0.769, 0.766 | 0.706 |
| far | 2692 | 0.774, 0.776, 0.758, 0.766, 0.773 | 0.717 |

E5b(A) against the reviewer's pair-only variant (Amendment 1): D +0.009 [-0.023, +0.041], p 0.4927

## E5 descriptive, B: near and far halves of POOL-LB

| Half | Items | S per seed | Memorisation ref. |
|---|---|---|---|
| near | 2266 | 0.892, 0.917, 0.909, 0.911, 0.898 | 0.898 |
| far | 2440 | 0.888, 0.905, 0.913, 0.904, 0.903 | 0.888 |

E5b(B) against the reviewer's pair-only variant (Amendment 1): D +0.003 [-0.019, +0.026], p 0.7074

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

| Model | Site (layer) | Template | Direction | Observed: AUROC of answer for adjacency (within string) | Observed: logit-difference gap, adjacent − not adjacent | Steering slope per natural SD | Signed percentile among matched controls | Shift to the other class: effect towards Yes per push towards adjacent | …as a share of the observed gap | Percentile among matched controls | AUROC of answer for adjacency: unsteered | …after the shift | …after the shift, control mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | parent | adjacency probe weights | 0.531 | +0.050 | -0.0003 [-0.0005, -0.0000] | 40 | -0.001 [-0.001, +0.000] | -1% | 25 | 0.614 | 0.616 | 0.615 |
| Qwen2.5-0.5B | final token, best layer (15/24) | parent | adjacency difference of means | 0.531 | +0.050 | -0.2600 [-0.2641, -0.2558] | 5 | -0.079 [-0.081, -0.077] | -158% | 5 | 0.614 | 0.732 | 0.608 |
| Qwen2.5-0.5B | final token, best layer (15/24) | ancestor | adjacency probe weights | 0.452 | -0.042 | -0.0001 [-0.0003, +0.0002] | 60 | +0.000 [-0.001, +0.001] | -0% | 40 | 0.543 | 0.543 | 0.543 |
| Qwen2.5-0.5B | final token, best layer (15/24) | ancestor | adjacency difference of means | 0.452 | -0.042 | -0.2494 [-0.2523, -0.2463] | 5 | -0.075 [-0.076, -0.074] | +177% | 5 | 0.543 | 0.661 | 0.542 |
| Qwen2.5-0.5B | final token, best layer (15/24) | descendant | adjacency probe weights | 0.570 | +0.055 | -0.0001 [-0.0003, +0.0001] | 55 | +0.000 [-0.001, +0.001] | +0% | 40 | 0.567 | 0.562 | 0.562 |
| Qwen2.5-0.5B | final token, best layer (15/24) | descendant | adjacency difference of means | 0.570 | +0.055 | -0.2285 [-0.2324, -0.2247] | 5 | -0.069 [-0.071, -0.068] | -126% | 5 | 0.567 | 0.742 | 0.555 |
| Qwen2.5-0.5B | final token, best layer (15/24) | collider | adjacency probe weights | 0.261 | -0.230 | -0.0003 [-0.0006, -0.0001] | 30 | -0.000 [-0.001, +0.001] | +0% | 60 | 0.269 | 0.271 | 0.270 |
| Qwen2.5-0.5B | final token, best layer (15/24) | collider | adjacency difference of means | 0.261 | -0.230 | -0.2262 [-0.2304, -0.2223] | 5 | -0.069 [-0.070, -0.067] | +30% | 5 | 0.269 | 0.384 | 0.270 |
| Qwen2.5-0.5B | final token, best layer (15/24) | confounder | adjacency probe weights | 0.375 | -0.113 | +0.0000 [-0.0002, +0.0003] | 70 | -0.000 [-0.001, +0.000] | +0% | 35 | 0.490 | 0.493 | 0.492 |
| Qwen2.5-0.5B | final token, best layer (15/24) | confounder | adjacency difference of means | 0.375 | -0.113 | -0.2331 [-0.2361, -0.2302] | 5 | -0.070 [-0.072, -0.069] | +62% | 5 | 0.490 | 0.598 | 0.489 |

<sub>Source: `results/feature_causal_qwen0.5b_c2c_debiased_adj_final_best.json`</sub>

| Graph type whose top-10 features are removed | Evaluated on | Items | Mean |Δ logit difference|, top features | …random features (mean of 3 sets) | …each random set | Top − random (difference of the two means) | Mean signed Δ | Flip rate, top | Flip rate, random | Accuracy before | Accuracy after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| confounding | items of that graph type | 40 | 0.318 [0.248, 0.384] | 0.171 | 0.17 / 0.18 / 0.16 | +0.147 | +0.307 | 0.07 | 0.07 | 0.53 | 0.45 |
| confounding | items of other graph types | 40 | 0.336 [0.276, 0.384] | 0.189 | 0.18 / 0.20 / 0.19 | +0.146 | +0.318 | 0.07 | 0.03 | 0.50 | 0.47 |
| collision | items of that graph type | 27 | 0.223 [0.199, 0.241] | 0.125 | 0.04 / 0.19 / 0.14 | +0.097 | +0.211 | 0.04 | 0.05 | 0.48 | 0.52 |
| collision | items of other graph types | 40 | 0.137 [0.111, 0.188] | 0.112 | 0.06 / 0.21 / 0.07 | +0.025 | +0.134 | 0.00 | 0.03 | 0.40 | 0.40 |
| chain | items of that graph type | 40 | 0.442 [0.381, 0.517] | 0.242 | 0.24 / 0.13 / 0.36 | +0.200 | -0.226 | 0.07 | 0.09 | 0.42 | 0.45 |
| chain | items of other graph types | 40 | 0.360 [0.272, 0.430] | 0.214 | 0.23 / 0.10 / 0.31 | +0.146 | -0.223 | 0.17 | 0.06 | 0.57 | 0.55 |
| fork | items of that graph type | 40 | 0.439 [0.386, 0.480] | 0.121 | 0.06 / 0.17 / 0.13 | +0.318 | -0.171 | 0.12 | 0.02 | 0.45 | 0.38 |
| fork | items of other graph types | 40 | 0.415 [0.324, 0.512] | 0.111 | 0.09 / 0.14 / 0.11 | +0.304 | -0.234 | 0.23 | 0.02 | 0.53 | 0.50 |
| mediation | items of that graph type | 40 | 0.190 [0.144, 0.239] | 0.269 | 0.17 / 0.51 / 0.13 | -0.079 | -0.137 | 0.07 | 0.06 | 0.53 | 0.45 |
| mediation | items of other graph types | 40 | 0.186 [0.146, 0.232] | 0.270 | 0.16 / 0.48 / 0.17 | -0.084 | -0.180 | 0.05 | 0.06 | 0.53 | 0.53 |

<sub>Source: `results/sae_ablation.json`</sub>

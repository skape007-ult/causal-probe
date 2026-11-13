| Split | Items | Distinct premises | Median items per premise | Fraction valid | Distinct hypothesis strings | Strings with a single label in this split | Items left if labels are balanced within each string |
|---|---|---|---|---|---|---|---|
| train | 205734 | 2338 | 90 | 0.185 | 120 | 72 | 63744 |
| dev | 1076 | 22 | 60 | 0.129 | 120 | 80 | 258 |
| test | 1162 | 21 | 60 | 0.155 | 120 | 79 | 296 |

<sub>Source: `results/benchmark_baselines.json`</sub>

| Template (train) | Example hypothesis | Items, first-named variable earlier in alphabet | Fraction valid | Items, first-named variable later in alphabet | Fraction valid  |
|---|---|---|---|---|---|
| parent | B directly causes C. | 34292 | 0.355 | 0 | – |
| child | C directly causes B. | 0 | – | 34291 | 0.000 |
| non-parent ancestor | B causes something else which causes C. | 17199 | 0.249 | 17157 | 0.000 |
| non-child descendant | C is a cause for B, but not a direct one. | 17093 | 0.253 | 17134 | 0.000 |

<sub>Source: `results/benchmark_baselines.json`</sub>

| Quantity (Corr2Cause) | Value |
|---|---|
| Distinct hypothesis strings in train | 120 |
| ...never valid | 72 |
| ...always valid | 0 |
| ...with both labels | 48 |
| Share of train items whose hypothesis string is never valid | 0.566 |
| Premises shared by train and dev | 2 |
| Premises shared by train and test | 0 |
| Premises shared by dev and test | 3 |
| Dev items whose premise is in train | 42 |

<sub>Source: `results/benchmark_baselines.json`</sub>

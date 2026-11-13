| Corr2Cause item set | Text-only baseline (premise-held-out cross-validation) | Items | Premises | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|---|---|
| Balanced pool | Hypothesis only (string prior) | 2400 | 300 | 0.859 [0.846, 0.871] | 0.857 [0.844, 0.870] | 0.913 [0.902, 0.924] |
| Balanced pool | Hypothesis only (TF-IDF) | 2400 | 300 | 0.842 [0.829, 0.856] | 0.841 [0.827, 0.856] | 0.909 [0.898, 0.920] |
| Balanced pool | Premise only (TF-IDF) | 2400 | 300 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.500 [0.500, 0.500] |
| Balanced pool | Whole prompt (TF-IDF) | 2400 | 300 | 0.753 [0.734, 0.770] | 0.753 [0.734, 0.770] | 0.812 [0.793, 0.830] |
| Balanced pool | Parsed features + hypothesis prior (GBM) | 2400 | 300 | 0.924 [0.913, 0.934] | 0.924 [0.913, 0.934] | 0.965 [0.957, 0.973] |
| Debiased pool | Hypothesis only (string prior) | 2400 | 1200 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.500 [0.500, 0.500] |
| Debiased pool | Hypothesis only (TF-IDF) | 2400 | 1200 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.500 [0.500, 0.500] |
| Debiased pool | Premise only (TF-IDF) | 2400 | 1200 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.500 [0.500, 0.500] |
| Debiased pool | Whole prompt (TF-IDF) | 2400 | 1200 | 0.495 [0.476, 0.514] | 0.493 [0.474, 0.512] | 0.512 [0.491, 0.535] |
| Debiased pool | Parsed features + hypothesis prior (GBM) | 2400 | 1200 | 0.812 [0.797, 0.829] | 0.812 [0.797, 0.829] | 0.901 [0.889, 0.913] |
| Rejected first design (strings balanced over the whole pool only) | Hypothesis only (string prior) | 2400 | 1498 | 0.315 [0.296, 0.334] | 0.314 [0.296, 0.333] | 0.247 [0.228, 0.267] |
| Rejected first design (strings balanced over the whole pool only) | Hypothesis only (TF-IDF) | 2400 | 1498 | 0.341 [0.323, 0.359] | 0.341 [0.323, 0.359] | 0.283 [0.263, 0.303] |
| Rejected first design (strings balanced over the whole pool only) | Premise only (TF-IDF) | 2400 | 1498 | 0.722 [0.701, 0.742] | 0.721 [0.700, 0.741] | 0.797 [0.776, 0.818] |
| Rejected first design (strings balanced over the whole pool only) | Whole prompt (TF-IDF) | 2400 | 1498 | 0.713 [0.692, 0.733] | 0.712 [0.691, 0.732] | 0.792 [0.770, 0.813] |
| Rejected first design (strings balanced over the whole pool only) | Parsed features + hypothesis prior (GBM) | 2400 | 1498 | 0.808 [0.790, 0.824] | 0.808 [0.790, 0.824] | 0.892 [0.878, 0.905] |

<sub>Source: `results/benchmark_baselines.json`</sub>

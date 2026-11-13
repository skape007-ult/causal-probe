| Text-only baseline (story-held-out cross-validation) | Items | Stories | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|---|
| Graph x query-type prior | 10112 | 47 | 0.446 [0.439, 0.454] | 0.446 [0.439, 0.453] | 0.429 [0.420, 0.439] |
| Question only (TF-IDF) | 10112 | 47 | 0.513 [0.499, 0.530] | 0.504 [0.488, 0.523] | 0.522 [0.499, 0.549] |
| Whole prompt (TF-IDF) | 10112 | 47 | 0.517 [0.499, 0.536] | 0.513 [0.494, 0.533] | 0.527 [0.501, 0.557] |
| Quoted numbers + question polarity (GBM) | 10112 | 47 | 0.835 [0.818, 0.853] | 0.835 [0.818, 0.853] | 0.936 [0.924, 0.949] |

<sub>Source: `results/benchmark_baselines.json`</sub>

| Query type / rung | Items | Fraction yes | Accuracy of numbers + polarity baseline | AUROC |
|---|---|---|---|---|
| ate | 1422 | 0.50 | 0.903 [0.862, 0.941] | 0.968 |
| backadj | 1580 | 0.50 | 1.000 [1.000, 1.000] | 1.000 |
| collider_bias | 158 | 0.50 | 1.000 [1.000, 1.000] | 1.000 |
| correlation | 1422 | 0.50 | 0.840 [0.799, 0.880] | 0.911 |
| det-counterfactual | 1422 | 0.50 | 0.480 [0.460, 0.501] | 0.481 |
| ett | 1264 | 0.50 | 0.907 [0.866, 0.948] | 0.971 |
| exp_away | 158 | 0.50 | 0.734 [0.598, 0.838] | 0.756 |
| marginal | 1580 | 0.50 | 0.941 [0.919, 0.956] | 0.984 |
| nde | 316 | 0.50 | 0.642 [0.578, 0.693] | 0.659 |
| nie | 790 | 0.50 | 0.746 [0.669, 0.814] | 0.851 |
| rung 1 | 3160 | 0.50 | 0.885 [0.858, 0.907] | 0.948 |
| rung 2 | 3160 | 0.50 | 0.956 [0.937, 0.973] | 0.994 |
| rung 3 | 3792 | 0.50 | 0.691 [0.666, 0.722] | 0.807 |

<sub>Source: `results/benchmark_baselines.json`</sub>

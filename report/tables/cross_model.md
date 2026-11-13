| Model | Params (B) | Dataset | Model acc. | Gold probe acc. | Probe − TF-IDF | Probe − random-init | Selected layer (rel. depth) | First layer within 1 pt (rel.) | Selectivity |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.494 | Corr2Cause pool | 0.501 | 0.807 [0.784, 0.828] | +0.019 [-0.007, +0.046] | +0.186 [+0.156, +0.219] | 15/24 (0.62) | 0.62 | +0.117 |
| Qwen2.5-0.5B | 0.494 | CLadder | 0.506 | 0.561 [0.534, 0.586] | +0.034 [-0.009, +0.074] | +0.063 [+0.024, +0.099] | 16/24 (0.67) | 0.67 | -0.234 |
| Qwen2.5-1.5B | 1.54 | Corr2Cause pool | 0.505 | 0.870 [0.850, 0.890] | +0.083 [+0.057, +0.106] | +0.276 [+0.244, +0.307] | 21/28 (0.75) | 0.64 | +0.088 |
| Qwen2.5-1.5B | 1.54 | CLadder | 0.506 | 0.605 [0.580, 0.631] | +0.078 [+0.039, +0.113] | +0.096 [+0.061, +0.134] | 24/28 (0.86) | 0.79 | -0.221 |

<sub>Source: `results/cross_model.json`</sub>

| Comparison (gold-probe accuracy, paired) | Difference |
|---|---|
| c2c_probe:qwen1.5b_minus_qwen0.5b | +0.063 [+0.040, +0.087] |
| cladder:qwen1.5b_minus_qwen0.5b | +0.044 [+0.014, +0.076] |

<sub>Source: `results/cross_model.json`</sub>

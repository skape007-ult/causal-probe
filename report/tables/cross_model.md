| Model | Params (B) | Dataset | Model acc. | Gold probe acc. | Probe − TF-IDF | Probe − random-init | Selected layer (rel. depth) | First layer within 1 pt (rel.) | Selectivity |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.494 | Corr2Cause pool | 0.501 | 0.807 [0.784, 0.828] | +0.019 [-0.007, +0.046] | +0.186 [+0.156, +0.219] | 15/24 (0.62) | 0.62 | +0.117 |
| Qwen2.5-0.5B | 0.494 | CLadder | 0.506 | 0.561 [0.534, 0.586] | +0.034 [-0.009, +0.074] | +0.063 [+0.024, +0.099] | 16/24 (0.67) | 0.67 | -0.234 |

<sub>Source: `results/cross_model.json`</sub>

| Comparison (gold-probe accuracy, paired) | Difference |
|---|---|

<sub>Source: `results/cross_model.json`</sub>

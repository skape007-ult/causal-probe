| Model | Item set | Gold probe | Parsed features | Probe + parsed features (stacked) | Stacked − parsed alone | Correlation of the two scores | Probe right when parsed features are wrong | Probe AUROC where parsed says valid | Probe AUROC where parsed says invalid |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause debiased pool | 0.633 [0.611, 0.657] | 0.816 [0.792, 0.837] | 0.830 [0.808, 0.851] | +0.014 [+0.002, +0.026] | 0.36 | 0.57 (n = 230) | 0.621 [0.559, 0.677] | 0.642 [0.588, 0.692] |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 0.752 [0.728, 0.774] | 0.816 [0.792, 0.837] | 0.839 [0.817, 0.858] | +0.023 [+0.006, +0.040] | 0.59 | 0.63 (n = 230) | 0.769 [0.723, 0.810] | 0.784 [0.742, 0.825] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

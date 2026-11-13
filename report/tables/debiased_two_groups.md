| Model | Group | Scorer | Accuracy | AUROC | Gold probe − this (accuracy, paired) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | — | **Gold-label probe** | 0.633 [0.611, 0.657] | 0.676 [0.650, 0.704] | — |
| Qwen2.5-0.5B | (a) shallow confound control | Bag-of-words, whole prompt | 0.518 [0.495, 0.541] | 0.526 [0.501, 0.552] | +0.115 [+0.085, +0.147] |
| Qwen2.5-0.5B | (a) shallow confound control | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.133 [+0.111, +0.157] |
| Qwen2.5-0.5B | (a) shallow confound control | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.133 [+0.111, +0.157] |
| Qwen2.5-0.5B | (a) shallow confound control | Probe trained on shuffled labels | 0.526 [0.503, 0.551] | 0.534 [0.508, 0.560] | +0.107 [+0.071, +0.139] |
| Qwen2.5-0.5B | (a) shallow confound control | Probe on a randomly initialised copy of the model | 0.604 [0.580, 0.628] | 0.627 [0.599, 0.653] | +0.029 [+0.002, +0.060] |
| Qwen2.5-0.5B | (b) structure-aware reference | Parsed features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.183 [-0.216, -0.151] |
| Qwen2.5-1.5B | — | **Gold-label probe** | 0.752 [0.728, 0.774] | 0.829 [0.807, 0.850] | — |
| Qwen2.5-1.5B | (a) shallow confound control | Bag-of-words, whole prompt | 0.518 [0.495, 0.541] | 0.526 [0.501, 0.552] | +0.235 [+0.201, +0.265] |
| Qwen2.5-1.5B | (a) shallow confound control | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.252 [+0.228, +0.274] |
| Qwen2.5-1.5B | (a) shallow confound control | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.252 [+0.228, +0.274] |
| Qwen2.5-1.5B | (a) shallow confound control | Probe trained on shuffled labels | 0.497 [0.471, 0.524] | 0.498 [0.471, 0.528] | +0.256 [+0.221, +0.287] |
| Qwen2.5-1.5B | (a) shallow confound control | Probe on a randomly initialised copy of the model | 0.579 [0.554, 0.604] | 0.617 [0.589, 0.646] | +0.174 [+0.142, +0.205] |
| Qwen2.5-1.5B | (b) structure-aware reference | Parsed features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.063 [-0.095, -0.036] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

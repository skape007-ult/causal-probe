| Model | Params (B) | Dataset | Model acc. | Gold probe acc. | Probe − strongest text baseline | Probe − random-init | Selected layer (rel. depth) | First layer within 1 pt (rel.) | Selectivity |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.494 | Corr2Cause pool | 0.501 | 0.807 [0.784, 0.828] | -0.113 [-0.136, -0.091] | +0.186 [+0.156, +0.219] | 15/24 (0.62) | 0.62 | +0.117 |
| Qwen2.5-0.5B | 0.494 | CLadder | 0.506 | 0.561 [0.534, 0.586] | -0.240 [-0.285, -0.203] | +0.063 [+0.024, +0.099] | 16/24 (0.67) | 0.67 | -0.234 |
| Qwen2.5-1.5B | 1.54 | Corr2Cause pool | 0.505 | 0.870 [0.850, 0.890] | -0.049 [-0.068, -0.032] | +0.276 [+0.244, +0.307] | 21/28 (0.75) | 0.64 | +0.088 |
| Qwen2.5-1.5B | 1.54 | CLadder | 0.506 | 0.605 [0.580, 0.631] | -0.196 [-0.234, -0.162] | +0.096 [+0.061, +0.134] | 24/28 (0.86) | 0.79 | -0.221 |
| Gemma-2-2B | 2.61 | Corr2Cause pool | 0.507 | 0.863 [0.841, 0.886] | -0.056 [-0.078, -0.033] | +0.267 [+0.236, +0.302] | 18/26 (0.69) | 0.62 | +0.071 |
| Gemma-2-2B | 2.61 | CLadder | 0.506 | 0.626 [0.590, 0.664] | -0.174 [-0.220, -0.133] | +0.107 [+0.064, +0.150] | 20/26 (0.77) | 0.65 | -0.194 |
| Qwen2.5-3B | 3.09 | Corr2Cause pool | 0.509 | 0.862 [0.843, 0.880] | -0.057 [-0.077, -0.038] | +0.251 [+0.220, +0.282] | 27/36 (0.75) | 0.69 | +0.071 |
| Qwen2.5-3B | 3.09 | CLadder † | 0.528 | 0.628 [0.600, 0.654] | -0.135 [-0.176, -0.096] | +0.112 [+0.078, +0.145] | 27/36 (0.75) | 0.75 | -0.195 |

<sub>Source: `results/cross_model.json`</sub><br><sub>† computed on the GPU machine's draw of the 3,000-item CLadder sample (2,271 items shared with the other models' draw; Section 7).</sub>

| Comparison (gold-probe accuracy, paired) | Difference |
|---|---|
| c2c_probe:qwen1.5b_minus_qwen0.5b | +0.063 [+0.040, +0.087] |
| c2c_probe:gemma2b_minus_qwen0.5b | +0.057 [+0.033, +0.081] |
| c2c_probe:qwen3b_minus_qwen0.5b | +0.055 [+0.032, +0.079] |
| c2c_probe:gemma2b_minus_qwen1.5b | -0.007 [-0.029, +0.017] |
| c2c_probe:qwen3b_minus_qwen1.5b | -0.008 [-0.028, +0.012] |
| c2c_probe:qwen3b_minus_gemma2b | -0.002 [-0.023, +0.019] |
| cladder:qwen1.5b_minus_qwen0.5b | +0.044 [+0.014, +0.076] |
| cladder:gemma2b_minus_qwen0.5b | +0.065 [+0.029, +0.103] |
| cladder:qwen3b_minus_qwen0.5b | +0.089 [+0.056, +0.126] |
| cladder:gemma2b_minus_qwen1.5b | +0.021 [-0.013, +0.056] |
| cladder:qwen3b_minus_qwen1.5b | +0.036 [-0.002, +0.074] |
| cladder:qwen3b_minus_gemma2b | +0.015 [-0.024, +0.052] |

<sub>Source: `results/cross_model.json`</sub>

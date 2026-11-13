| Model | Item set | Layer | Direction | Slope: Δ(Yes−No logit) per unit s | Share of items with positive slope | One unit of s in the direction's own SDs | Slope per own SD | Effect of moving across the natural gold-class gap | AUROC for gold label | AUROC for model's answer |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold-label probe weights | -0.053 [-0.059, -0.048] | 0.08 | 119.0 | -0.0004 | -0.0007 | 0.864 | 0.593 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold difference of class means | +0.867 [+0.859, +0.876] | 1.00 | 2.9 | +0.2956 | +0.1229 | 0.670 | 0.695 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | model-answer probe weights (positive control) | +1.705 [+1.694, +1.717] | 1.00 | 15.3 | +0.1112 | +0.0351 | 0.587 | 0.994 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 0 | +0.028 [+0.023, +0.034] | 0.77 | 22.3 | +0.0013 | -0.0003 | 0.403 | 0.349 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 1 | +0.485 [+0.481, +0.489] | 1.00 | 45.6 | +0.0106 | +0.0023 | 0.560 | 0.620 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 2 | -0.271 [-0.276, -0.265] | 0.00 | 34.5 | -0.0079 | +0.0019 | 0.438 | 0.254 |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | gold-label probe weights | +0.159 [+0.154, +0.164] | 1.00 | 188.1 | +0.0008 | +0.0012 | 0.800 | 0.579 |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | gold difference of class means | -0.082 [-0.088, -0.077] | 0.01 | 3.2 | -0.0259 | -0.0077 | 0.600 | 0.560 |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | model-answer probe weights (positive control) | -0.013 [-0.018, -0.009] | 0.31 | 83.2 | -0.0002 | -0.0000 | 0.593 | 0.926 |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | random direction 0 | -0.081 [-0.084, -0.077] | 0.00 | 28.8 | -0.0028 | +0.0004 | 0.445 | 0.481 |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | random direction 1 | +0.105 [+0.101, +0.109] | 1.00 | 37.0 | +0.0028 | +0.0003 | 0.525 | 0.627 |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | random direction 2 | +0.238 [+0.234, +0.241] | 1.00 | 29.4 | +0.0081 | +0.0003 | 0.526 | 0.566 |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | gold-label probe weights | +0.108 [+0.106, +0.110] | 1.00 | 146.2 | +0.0007 | +0.0011 | 0.840 | 0.606 |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | gold difference of class means | +0.822 [+0.818, +0.826] | 1.00 | 2.7 | +0.2989 | +0.1272 | 0.619 | 0.842 |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | model-answer probe weights (positive control) | +1.363 [+1.357, +1.368] | 1.00 | 6.7 | +0.2033 | +0.0646 | 0.585 | 0.998 |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | random direction 0 | +0.080 [+0.078, +0.082] | 1.00 | 27.1 | +0.0030 | +0.0001 | 0.511 | 0.570 |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | random direction 1 | +0.170 [+0.168, +0.172] | 1.00 | 32.0 | +0.0053 | -0.0001 | 0.487 | 0.615 |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | random direction 2 | -0.075 [-0.077, -0.074] | 0.00 | 17.9 | -0.0042 | +0.0010 | 0.418 | 0.269 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | gold-label probe weights | -0.228 [-0.232, -0.224] | 0.00 | 322.9 | -0.0007 | -0.0009 | 0.672 | 0.502 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | gold difference of class means | +0.153 [+0.149, +0.158] | 1.00 | 8.8 | +0.0174 | +0.0027 | 0.525 | 0.458 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | model-answer probe weights (positive control) | +0.525 [+0.520, +0.531] | 1.00 | 42.8 | +0.0123 | -0.0001 | 0.486 | 0.960 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | random direction 0 | -0.048 [-0.053, -0.043] | 0.11 | 36.4 | -0.0013 | -0.0000 | 0.508 | 0.469 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | random direction 1 | +0.127 [+0.123, +0.130] | 1.00 | 38.5 | +0.0033 | -0.0000 | 0.508 | 0.513 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | random direction 2 | -0.072 [-0.076, -0.069] | 0.00 | 39.8 | -0.0018 | +0.0000 | 0.496 | 0.515 |
| Qwen2.5-0.5B | CLadder | 16/24 | gold-label probe weights | +0.232 [+0.227, +0.240] | 1.00 | 135.1 | +0.0017 | +0.0018 | 0.547 | 0.448 |
| Qwen2.5-0.5B | CLadder | 16/24 | gold difference of class means | -0.814 [-0.829, -0.797] | 0.00 | 2.5 | -0.3274 | -0.0430 | 0.487 | 0.172 |
| Qwen2.5-0.5B | CLadder | 16/24 | model-answer probe weights (positive control) | +1.447 [+1.431, +1.461] | 1.00 | 12.0 | +0.1211 | -0.0105 | 0.505 | 0.956 |
| Qwen2.5-0.5B | CLadder | 16/24 | random direction 0 | +0.095 [+0.083, +0.104] | 0.99 | 20.4 | +0.0046 | -0.0001 | 0.488 | 0.474 |
| Qwen2.5-0.5B | CLadder | 16/24 | random direction 1 | +0.240 [+0.226, +0.253] | 1.00 | 36.7 | +0.0065 | -0.0003 | 0.498 | 0.636 |
| Qwen2.5-0.5B | CLadder | 16/24 | random direction 2 | -0.074 [-0.083, -0.063] | 0.08 | 30.3 | -0.0024 | -0.0002 | 0.481 | 0.345 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold-label probe weights | -0.366 [-0.376, -0.357] | 0.00 | 63.1 | -0.0058 | -0.0087 | 0.915 | 0.539 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold difference of class means | +1.989 [+1.952, +2.029] | 1.00 | 1.9 | +1.0589 | +0.4599 | 0.650 | 0.942 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | model-answer probe weights (positive control) | +2.353 [+2.319, +2.390] | 1.00 | 2.9 | +0.8227 | +0.2218 | 0.581 | 0.999 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 0 | -0.135 [-0.145, -0.124] | 0.06 | 30.9 | -0.0044 | -0.0001 | 0.530 | 0.792 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 1 | -0.351 [-0.366, -0.338] | 0.00 | 64.4 | -0.0055 | +0.0005 | 0.477 | 0.237 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 2 | -0.089 [-0.099, -0.078] | 0.12 | 58.8 | -0.0015 | -0.0003 | 0.538 | 0.318 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | gold-label probe weights | -0.211 [-0.224, -0.198] | 0.03 | 280.4 | -0.0008 | -0.0011 | 0.779 | 0.490 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | gold difference of class means | +1.874 [+1.850, +1.897] | 1.00 | 2.8 | +0.6639 | +0.0757 | 0.526 | 0.950 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | model-answer probe weights (positive control) | +1.705 [+1.688, +1.723] | 1.00 | 3.3 | +0.5142 | +0.0232 | 0.507 | 0.999 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | random direction 0 | +0.022 [+0.017, +0.027] | 0.67 | 32.9 | +0.0007 | +0.0000 | 0.507 | 0.896 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | random direction 1 | -0.376 [-0.383, -0.369] | 0.00 | 38.5 | -0.0098 | +0.0002 | 0.502 | 0.174 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | random direction 2 | +0.216 [+0.211, +0.222] | 1.00 | 51.2 | +0.0042 | -0.0002 | 0.492 | 0.268 |
| Qwen2.5-1.5B | CLadder | 24/28 | gold-label probe weights | -0.058 [-0.068, -0.050] | 0.17 | 149.6 | -0.0004 | -0.0005 | 0.622 | 0.525 |
| Qwen2.5-1.5B | CLadder | 24/28 | gold difference of class means | +0.720 [+0.695, +0.742] | 1.00 | 2.5 | +0.2851 | +0.0337 | 0.497 | 0.496 |
| Qwen2.5-1.5B | CLadder | 24/28 | model-answer probe weights (positive control) | +3.550 [+3.517, +3.582] | 1.00 | 5.2 | +0.6865 | +0.0017 | 0.518 | 0.997 |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 0 | +0.140 [+0.122, +0.163] | 0.78 | 34.2 | +0.0041 | +0.0001 | 0.496 | 0.577 |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 1 | -0.257 [-0.268, -0.246] | 0.00 | 68.3 | -0.0038 | -0.0002 | 0.473 | 0.393 |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 2 | +0.198 [+0.186, +0.210] | 0.97 | 32.9 | +0.0060 | +0.0004 | 0.525 | 0.654 |

<sub>Source: `results/steering_analysis_qwen0.5b_c2c_probe.json`, `results/steering_analysis_qwen0.5b_c2c_probe_d0.5.json`, `results/steering_analysis_qwen0.5b_c2c_probe_d0.85.json`, `results/steering_analysis_qwen0.5b_c2c_debiased.json`, `results/steering_analysis_qwen0.5b_cladder.json`, `results/steering_analysis_qwen1.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_c2c_debiased.json`, `results/steering_analysis_qwen1.5b_cladder.json`</sub>

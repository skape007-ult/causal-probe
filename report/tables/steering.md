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
| Gemma-2-2B | Corr2Cause pool | 18/26 | gold-label probe weights | +0.247 [+0.224, +0.269] | 0.87 | 151.0 | +0.0016 | +0.0026 | 0.917 | 0.555 |
| Gemma-2-2B | Corr2Cause pool | 18/26 | gold difference of class means | +2.091 [+1.963, +2.215] | 0.94 | 1.7 | +1.2253 | +0.4718 | 0.637 | 0.994 |
| Gemma-2-2B | Corr2Cause pool | 18/26 | model-answer probe weights (positive control) | +1.796 [+1.759, +1.831] | 1.00 | 3.2 | +0.5564 | +0.0785 | 0.560 | 1.000 |
| Gemma-2-2B | Corr2Cause pool | 18/26 | random direction 0 | +0.969 [+0.901, +1.036] | 0.96 | 38.6 | +0.0251 | +0.0073 | 0.591 | 0.943 |
| Gemma-2-2B | Corr2Cause pool | 18/26 | random direction 1 | -0.094 [-0.117, -0.072] | 0.42 | 32.9 | -0.0028 | -0.0007 | 0.572 | 0.993 |
| Gemma-2-2B | Corr2Cause pool | 18/26 | random direction 2 | +0.063 [+0.044, +0.082] | 0.67 | 66.5 | +0.0009 | +0.0000 | 0.524 | 0.958 |
| Gemma-2-2B | Corr2Cause pool | 13/26 | gold-label probe weights | -0.259 [-0.278, -0.239] | 0.07 | 219.9 | -0.0012 | -0.0019 | 0.861 | 0.559 |
| Gemma-2-2B | Corr2Cause pool | 13/26 | gold difference of class means | +0.136 [+0.112, +0.159] | 0.73 | 3.6 | +0.0373 | +0.0128 | 0.626 | 0.755 |
| Gemma-2-2B | Corr2Cause pool | 13/26 | model-answer probe weights (positive control) | +0.113 [+0.104, +0.122] | 0.94 | 15.9 | +0.0071 | +0.0006 | 0.538 | 0.996 |
| Gemma-2-2B | Corr2Cause pool | 13/26 | random direction 0 | +0.008 [-0.011, +0.026] | 0.59 | 54.9 | +0.0001 | +0.0000 | 0.543 | 0.755 |
| Gemma-2-2B | Corr2Cause pool | 13/26 | random direction 1 | -0.182 [-0.193, -0.171] | 0.01 | 87.3 | -0.0021 | +0.0004 | 0.433 | 0.578 |
| Gemma-2-2B | Corr2Cause pool | 13/26 | random direction 2 | +0.322 [+0.309, +0.334] | 1.00 | 85.5 | +0.0038 | +0.0003 | 0.549 | 0.417 |
| Gemma-2-2B | Corr2Cause pool | 22/26 | gold-label probe weights | -0.182 [-0.195, -0.169] | 0.00 | 180.6 | -0.0010 | -0.0016 | 0.906 | 0.544 |
| Gemma-2-2B | Corr2Cause pool | 22/26 | gold difference of class means | +2.896 [+2.801, +2.992] | 1.00 | 1.6 | +1.8215 | +0.5959 | 0.615 | 1.000 |
| Gemma-2-2B | Corr2Cause pool | 22/26 | model-answer probe weights (positive control) | +2.376 [+2.333, +2.418] | 1.00 | 3.2 | +0.7374 | +0.1157 | 0.570 | 1.000 |
| Gemma-2-2B | Corr2Cause pool | 22/26 | random direction 0 | +0.373 [+0.366, +0.381] | 1.00 | 71.4 | +0.0052 | -0.0003 | 0.514 | 0.959 |
| Gemma-2-2B | Corr2Cause pool | 22/26 | random direction 1 | +0.050 [+0.042, +0.057] | 0.82 | 48.7 | +0.0010 | -0.0000 | 0.499 | 0.917 |
| Gemma-2-2B | Corr2Cause pool | 22/26 | random direction 2 | +0.158 [+0.151, +0.165] | 1.00 | 78.0 | +0.0020 | -0.0002 | 0.483 | 0.891 |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | gold-label probe weights | +0.468 [+0.453, +0.481] | 1.00 | 349.1 | +0.0013 | +0.0021 | 0.780 | 0.473 |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | gold difference of class means | -1.766 [-1.814, -1.718] | 0.00 | 3.3 | -0.5297 | -0.0609 | 0.525 | 0.013 |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | model-answer probe weights (positive control) | +1.854 [+1.804, +1.901] | 1.00 | 4.2 | +0.4376 | -0.0123 | 0.486 | 1.000 |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | random direction 0 | +0.553 [+0.497, +0.607] | 0.82 | 42.6 | +0.0130 | -0.0003 | 0.496 | 0.845 |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | random direction 1 | +0.046 [+0.018, +0.073] | 0.57 | 43.7 | +0.0011 | -0.0000 | 0.489 | 0.888 |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | random direction 2 | +0.170 [+0.149, +0.191] | 0.78 | 55.7 | +0.0031 | -0.0000 | 0.496 | 0.618 |
| Gemma-2-2B | CLadder | 20/26 | gold-label probe weights | -0.719 [-0.743, -0.689] | 0.00 | 285.8 | -0.0025 | -0.0040 | 0.648 | 0.557 |
| Gemma-2-2B | CLadder | 20/26 | gold difference of class means | +1.007 [+0.985, +1.032] | 1.00 | 1.8 | +0.5486 | +0.0634 | 0.505 | 0.852 |
| Gemma-2-2B | CLadder | 20/26 | model-answer probe weights (positive control) | +3.943 [+3.831, +4.064] | 1.00 | 4.8 | +0.8279 | +0.0582 | 0.517 | 0.998 |
| Gemma-2-2B | CLadder | 20/26 | random direction 0 | +0.278 [+0.234, +0.326] | 0.75 | 62.5 | +0.0045 | -0.0007 | 0.476 | 0.360 |
| Gemma-2-2B | CLadder | 20/26 | random direction 1 | -0.147 [-0.171, -0.123] | 0.18 | 62.1 | -0.0024 | +0.0000 | 0.486 | 0.593 |
| Gemma-2-2B | CLadder | 20/26 | random direction 2 | +0.313 [+0.279, +0.342] | 1.00 | 66.6 | +0.0047 | +0.0005 | 0.511 | 0.688 |
| Gemma-2-2B | CLadder | 13/26 | gold-label probe weights | -0.145 [-0.166, -0.120] | 0.07 | 352.0 | -0.0004 | -0.0006 | 0.580 | 0.488 |
| Gemma-2-2B | CLadder | 13/26 | gold difference of class means | -0.248 [-0.264, -0.232] | 0.00 | 3.3 | -0.0756 | -0.0111 | 0.478 | 0.562 |
| Gemma-2-2B | CLadder | 13/26 | model-answer probe weights (positive control) | +0.226 [+0.201, +0.248] | 0.96 | 64.3 | +0.0035 | -0.0000 | 0.513 | 0.911 |
| Gemma-2-2B | CLadder | 13/26 | random direction 0 | -0.030 [-0.056, -0.005] | 0.48 | 41.5 | -0.0007 | -0.0000 | 0.481 | 0.527 |
| Gemma-2-2B | CLadder | 13/26 | random direction 1 | -0.042 [-0.061, -0.026] | 0.39 | 41.9 | -0.0010 | +0.0001 | 0.509 | 0.559 |
| Gemma-2-2B | CLadder | 13/26 | random direction 2 | -0.017 [-0.031, -0.001] | 0.38 | 58.6 | -0.0003 | -0.0000 | 0.491 | 0.443 |
| Gemma-2-2B | CLadder | 22/26 | gold-label probe weights | +0.099 [+0.078, +0.119] | 0.83 | 280.3 | +0.0004 | +0.0006 | 0.656 | 0.532 |
| Gemma-2-2B | CLadder | 22/26 | gold difference of class means | +1.030 [+0.982, +1.085] | 1.00 | 1.8 | +0.5591 | +0.0626 | 0.506 | 0.855 |
| Gemma-2-2B | CLadder | 22/26 | model-answer probe weights (positive control) | +4.413 [+4.311, +4.526] | 1.00 | 4.5 | +0.9794 | +0.0636 | 0.516 | 0.999 |
| Gemma-2-2B | CLadder | 22/26 | random direction 0 | +0.501 [+0.485, +0.520] | 1.00 | 58.3 | +0.0086 | -0.0009 | 0.482 | 0.274 |
| Gemma-2-2B | CLadder | 22/26 | random direction 1 | +0.082 [+0.073, +0.092] | 0.95 | 46.0 | +0.0018 | +0.0001 | 0.494 | 0.833 |
| Gemma-2-2B | CLadder | 22/26 | random direction 2 | +0.251 [+0.223, +0.279] | 1.00 | 57.4 | +0.0044 | +0.0005 | 0.510 | 0.673 |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | gold-label probe weights | -0.776 [-0.813, -0.737] | 0.01 | 64.4 | -0.0121 | -0.0197 | 0.919 | 0.622 |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | gold difference of class means | +4.768 [+4.390, +5.144] | 0.92 | 1.7 | +2.7888 | +1.4156 | 0.657 | 0.985 |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | model-answer probe weights (positive control) | +4.711 [+4.362, +5.046] | 0.99 | 3.5 | +1.3557 | +0.4743 | 0.613 | 0.999 |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | random direction 0 | +0.641 [+0.598, +0.679] | 1.00 | 30.7 | +0.0209 | +0.0056 | 0.617 | 0.924 |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | random direction 1 | +0.219 [+0.149, +0.297] | 0.59 | 53.4 | +0.0041 | +0.0016 | 0.604 | 0.813 |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | random direction 2 | -0.329 [-0.370, -0.290] | 0.14 | 59.4 | -0.0055 | +0.0002 | 0.458 | 0.272 |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | gold-label probe weights | +0.118 [+0.106, +0.130] | 0.90 | 103.8 | +0.0011 | +0.0015 | 0.798 | 0.614 |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | gold difference of class means | +0.131 [+0.106, +0.158] | 0.62 | 3.1 | +0.0422 | +0.0104 | 0.597 | 0.882 |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | model-answer probe weights (positive control) | +0.342 [+0.324, +0.359] | 1.00 | 19.0 | +0.0179 | +0.0054 | 0.603 | 0.970 |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | random direction 0 | +0.152 [+0.143, +0.162] | 0.96 | 40.8 | +0.0037 | -0.0000 | 0.507 | 0.457 |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | random direction 1 | +0.160 [+0.146, +0.172] | 0.97 | 39.8 | +0.0040 | +0.0002 | 0.536 | 0.578 |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | random direction 2 | +0.032 [+0.012, +0.053] | 0.50 | 61.5 | +0.0005 | -0.0000 | 0.507 | 0.616 |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | gold-label probe weights | +0.670 [+0.651, +0.689] | 1.00 | 105.0 | +0.0064 | +0.0104 | 0.901 | 0.623 |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | gold difference of class means | +4.907 [+4.796, +5.026] | 1.00 | 1.5 | +3.3455 | +1.2556 | 0.625 | 0.994 |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | model-answer probe weights (positive control) | +5.710 [+5.625, +5.787] | 1.00 | 3.0 | +1.8815 | +0.4651 | 0.600 | 1.000 |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | random direction 0 | +0.855 [+0.829, +0.881] | 1.00 | 68.4 | +0.0125 | -0.0037 | 0.439 | 0.508 |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | random direction 1 | -0.076 [-0.088, -0.066] | 0.21 | 49.3 | -0.0015 | -0.0005 | 0.622 | 0.963 |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | random direction 2 | -0.105 [-0.122, -0.088] | 0.18 | 47.9 | -0.0022 | +0.0005 | 0.427 | 0.071 |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | gold-label probe weights | -0.244 [-0.303, -0.182] | 0.23 | 228.3 | -0.0011 | -0.0017 | 0.791 | 0.491 |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | gold difference of class means | +3.912 [+3.663, +4.150] | 0.95 | 4.5 | +0.8618 | +0.1548 | 0.544 | 0.941 |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | model-answer probe weights (positive control) | +5.173 [+4.886, +5.436] | 1.00 | 5.2 | +1.0032 | +0.0333 | 0.512 | 0.999 |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | random direction 0 | +0.641 [+0.601, +0.681] | 1.00 | 33.9 | +0.0189 | +0.0006 | 0.523 | 0.897 |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | random direction 1 | +0.124 [+0.062, +0.186] | 0.57 | 62.1 | +0.0020 | +0.0001 | 0.523 | 0.734 |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | random direction 2 | -0.336 [-0.369, -0.302] | 0.11 | 52.0 | -0.0065 | -0.0001 | 0.510 | 0.337 |
| Qwen2.5-3B | CLadder | 29/36 | gold-label probe weights | +0.797 [+0.692, +0.937] | 0.91 | 130.9 | +0.0061 | +0.0093 | 0.639 | 0.621 |
| Qwen2.5-3B | CLadder | 29/36 | gold difference of class means | +12.240 [+12.052, +12.427] | 1.00 | 2.1 | +5.7158 | +0.9348 | 0.536 | 0.987 |
| Qwen2.5-3B | CLadder | 29/36 | model-answer probe weights (positive control) | +11.255 [+11.061, +11.419] | 1.00 | 4.3 | +2.6211 | +0.3838 | 0.541 | 0.997 |
| Qwen2.5-3B | CLadder | 29/36 | random direction 0 | +0.790 [+0.734, +0.857] | 1.00 | 29.5 | +0.0268 | +0.0006 | 0.516 | 0.803 |
| Qwen2.5-3B | CLadder | 29/36 | random direction 1 | -0.743 [-0.795, -0.698] | 0.00 | 23.6 | -0.0315 | +0.0021 | 0.492 | 0.122 |
| Qwen2.5-3B | CLadder | 29/36 | random direction 2 | -0.424 [-0.479, -0.376] | 0.12 | 39.9 | -0.0106 | +0.0002 | 0.506 | 0.342 |

<sub>Source: `results/steering_analysis_qwen0.5b_c2c_probe.json`, `results/steering_analysis_qwen0.5b_c2c_probe_d0.5.json`, `results/steering_analysis_qwen0.5b_c2c_probe_d0.85.json`, `results/steering_analysis_qwen0.5b_c2c_debiased.json`, `results/steering_analysis_qwen0.5b_cladder.json`, `results/steering_analysis_qwen1.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_c2c_debiased.json`, `results/steering_analysis_qwen1.5b_cladder.json`, `results/steering_analysis_gemma2b_c2c_probe.json`, `results/steering_analysis_gemma2b_c2c_probe_d0.5.json`, `results/steering_analysis_gemma2b_c2c_probe_d0.85.json`, `results/steering_analysis_gemma2b_c2c_debiased.json`, `results/steering_analysis_gemma2b_cladder.json`, `results/steering_analysis_gemma2b_cladder_d0.5.json`, `results/steering_analysis_gemma2b_cladder_d0.85.json`, `results/steering_analysis_qwen3b_c2c_probe.json`, `results/steering_analysis_qwen3b_c2c_probe_d0.5.json`, `results/steering_analysis_qwen3b_c2c_probe_d0.85.json`, `results/steering_analysis_qwen3b_c2c_debiased.json`, `results/steering_analysis_qwen3b_cladder.json`</sub>

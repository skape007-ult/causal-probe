| Model | Item set | Layer | Direction | Slope: Δ(Yes−No logit) per unit s | Share of items with positive slope | One unit of s in the direction's own SDs | Slope per own SD | Effect of moving across the natural gold-class gap | AUROC for gold label | AUROC for model's answer |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold-label probe weights | -0.053 [-0.059, -0.048] | 0.08 | 119.0 | -0.0004 | -0.0007 | 0.864 | 0.593 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold difference of class means | +0.867 [+0.859, +0.876] | 1.00 | 2.9 | +0.2956 | +0.1229 | 0.670 | 0.695 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | model-answer probe weights (positive control) | +1.705 [+1.694, +1.717] | 1.00 | 15.3 | +0.1112 | +0.0351 | 0.587 | 0.994 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 0 | +0.028 [+0.023, +0.034] | 0.77 | 22.3 | +0.0013 | -0.0003 | 0.403 | 0.349 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 1 | +0.485 [+0.481, +0.489] | 1.00 | 45.6 | +0.0106 | +0.0023 | 0.560 | 0.620 |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 2 | -0.271 [-0.276, -0.265] | 0.00 | 34.5 | -0.0079 | +0.0019 | 0.438 | 0.254 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold-label probe weights | -0.366 [-0.376, -0.357] | 0.00 | 63.1 | -0.0058 | -0.0087 | 0.915 | 0.539 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold difference of class means | +1.989 [+1.952, +2.029] | 1.00 | 1.9 | +1.0589 | +0.4599 | 0.650 | 0.942 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | model-answer probe weights (positive control) | +2.353 [+2.319, +2.390] | 1.00 | 2.9 | +0.8227 | +0.2218 | 0.581 | 0.999 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 0 | -0.135 [-0.145, -0.124] | 0.06 | 30.9 | -0.0044 | -0.0001 | 0.530 | 0.792 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 1 | -0.351 [-0.366, -0.338] | 0.00 | 64.4 | -0.0055 | +0.0005 | 0.477 | 0.237 |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 2 | -0.089 [-0.099, -0.078] | 0.12 | 58.8 | -0.0015 | -0.0003 | 0.538 | 0.318 |
| Qwen2.5-1.5B | CLadder | 24/28 | gold-label probe weights | -0.058 [-0.068, -0.050] | 0.17 | 149.6 | -0.0004 | -0.0005 | 0.622 | 0.525 |
| Qwen2.5-1.5B | CLadder | 24/28 | gold difference of class means | +0.720 [+0.695, +0.742] | 1.00 | 2.5 | +0.2851 | +0.0337 | 0.497 | 0.496 |
| Qwen2.5-1.5B | CLadder | 24/28 | model-answer probe weights (positive control) | +3.550 [+3.517, +3.582] | 1.00 | 5.2 | +0.6865 | +0.0017 | 0.518 | 0.997 |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 0 | +0.140 [+0.122, +0.163] | 0.78 | 34.2 | +0.0041 | +0.0001 | 0.496 | 0.577 |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 1 | -0.257 [-0.268, -0.246] | 0.00 | 68.3 | -0.0038 | -0.0002 | 0.473 | 0.393 |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 2 | +0.198 [+0.186, +0.210] | 0.97 | 32.9 | +0.0060 | +0.0004 | 0.525 | 0.654 |

<sub>Source: `results/steering_analysis_qwen0.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_cladder.json`</sub>

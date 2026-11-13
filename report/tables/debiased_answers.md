| Model | Hypothesis template | Items | Answer logit: AUROC for gold within hypothesis string | …selection half | …report half | Answer logit: AUROC for pair adjacency | Correlation of answer logit with probe score (string means removed) | Gold probe: AUROC within hypothesis string |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | all templates | 2400 | 0.506 [0.487, 0.525] | – | – | – | +0.025 [-0.022, +0.073] | – |
| Qwen2.5-0.5B | has_collider | 500 | 0.586 [0.536, 0.636] | 0.588 | 0.588 | 0.261 [0.220, 0.302] | +0.167 [+0.056, +0.270] | 0.645 [0.598, 0.692] |
| Qwen2.5-0.5B | has_confounder | 300 | 0.465 [0.396, 0.530] | 0.515 | 0.422 | 0.375 [0.305, 0.441] | +0.018 [-0.098, +0.127] | 0.563 [0.499, 0.633] |
| Qwen2.5-0.5B | non-child descendant | 450 | 0.449 [0.392, 0.504] | 0.453 | 0.450 | 0.570 [0.513, 0.632] | -0.170 [-0.251, -0.089] | 0.711 [0.663, 0.762] |
| Qwen2.5-0.5B | non-parent ancestor | 450 | 0.445 [0.388, 0.498] | 0.455 | 0.441 | 0.452 [0.394, 0.515] | -0.182 [-0.291, -0.068] | 0.630 [0.576, 0.686] |
| Qwen2.5-0.5B | parent: "X directly causes Y" | 700 | 0.541 [0.502, 0.586] | 0.557 | 0.530 | 0.531 [0.489, 0.573] | +0.132 [+0.055, +0.212] | 0.781 [0.744, 0.816] |
| Qwen2.5-1.5B | all templates | 2400 | 0.507 [0.486, 0.529] | – | – | – | +0.126 [+0.079, +0.174] | – |
| Qwen2.5-1.5B | has_collider | 500 | 0.500 [0.445, 0.559] | 0.535 | 0.475 | 0.381 [0.318, 0.446] | +0.050 [-0.039, +0.137] | 0.759 [0.717, 0.800] |
| Qwen2.5-1.5B | has_confounder | 300 | 0.442 [0.371, 0.513] | 0.477 | 0.419 | 0.427 [0.356, 0.496] | -0.202 [-0.308, -0.089] | 0.782 [0.725, 0.842] |
| Qwen2.5-1.5B | non-child descendant | 450 | 0.430 [0.375, 0.485] | 0.394 | 0.460 | 0.469 [0.409, 0.532] | -0.127 [-0.224, -0.028] | 0.817 [0.778, 0.859] |
| Qwen2.5-1.5B | non-parent ancestor | 450 | 0.449 [0.393, 0.508] | 0.448 | 0.446 | 0.434 [0.374, 0.496] | -0.133 [-0.229, -0.046] | 0.776 [0.730, 0.819] |
| Qwen2.5-1.5B | parent: "X directly causes Y" | 700 | 0.628 [0.581, 0.673] | 0.644 | 0.613 | 0.649 [0.603, 0.693] | +0.359 [+0.283, +0.425] | 0.886 [0.859, 0.912] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

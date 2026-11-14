| Model | Site (layer) | Off-diagonal contrast with the target template's sign | …with the source template's sign | Full-mediation benchmark | Cosines between the five class-mean differences (range) | Cosine of the direct-causation difference with each of the others |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | -0.042 [-0.043, -0.042] | +0.111 [+0.111, +0.112] | 0.112 | -0.60 to +0.87 | ancestor -0.18 · descendant -0.60 · collider -0.49 · confounder -0.59 |
| Qwen2.5-0.5B | end of hypothesis, best layer (13/24) | -0.001 [-0.002, -0.001] | +0.010 [+0.009, +0.010] | 0.112 | -0.44 to +0.74 | ancestor +0.02 · descendant -0.44 · collider -0.10 · confounder -0.07 |
| Qwen2.5-0.5B | final token, earlier layer (11/24) | -0.002 [-0.002, -0.001] | +0.001 [+0.001, +0.001] | 0.112 | -0.75 to +0.94 | ancestor -0.34 · descendant -0.64 · collider -0.62 · confounder -0.75 |
| Qwen2.5-0.5B | end of hypothesis, earlier layer (5/24) | -0.002 [-0.003, -0.002] | +0.002 [+0.002, +0.002] | 0.112 | -0.67 to +0.92 | ancestor -0.28 · descendant -0.58 · collider -0.67 · confounder -0.62 |

<sub>Source: `results/adjacency_qwen0.5b_c2c_debiased_final_best.json`, `results/adjacency_qwen0.5b_c2c_debiased_hyp_best.json`, `results/adjacency_qwen0.5b_c2c_debiased_final_first50.json`, `results/adjacency_qwen0.5b_c2c_debiased_hyp_first50.json`</sub>

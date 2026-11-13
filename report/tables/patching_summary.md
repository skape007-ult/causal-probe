| Model | Pair kind | Pairs | Mean D = LD(clean) − LD(corrupt) | Frac. D > 0 | Both answered correctly | First layer with ≥ 0.5 recovery at final token | …as relative depth | Max |control| at diff tokens | Max |control| at final token | Pairs with control |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | hyp | 60 | +0.009 [-0.060, +0.078] | 0.50 [0.38, 0.62] | 0.00 | 15 | 0.62 | 0.96 | 0.33 | 16 |
| Qwen2.5-0.5B | prem | 60 | +0.023 [-0.042, +0.090] | 0.58 [0.45, 0.71] | 0.00 | 15 | 0.62 | 1.00 | 0.45 | 16 |
| Qwen2.5-1.5B | hyp | 40 | -0.088 [-0.420, +0.267] | 0.55 [0.43, 0.68] | 0.00 | 18 | 0.64 | 0.91 | 0.65 | 6 |
| Qwen2.5-1.5B | prem | 40 | -0.087 [-0.424, +0.269] | 0.47 [0.33, 0.61] | 0.03 | 18 | 0.64 | 0.19 | 1.00 | 6 |

<sub>Source: `results/patching_qwen0.5b.json`, `results/patching_qwen1.5b.json`</sub>

| Model | Dataset | Layer | Direction | Flip rate α=1 | Flip rate α=4 | Acc. towards gold α=4 | Ablation mean |ΔLD| | Proj. std / s |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15 | gold probe | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.06] | 0.51 [0.47, 0.56] | 0.005 | 0.026 |
| Qwen2.5-0.5B | Corr2Cause pool | 15 | gold diff-of-means | 0.30 [0.25, 0.36] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.268 | 0.341 |
| Qwen2.5-0.5B | Corr2Cause pool | 15 | model output | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.170 | 0.105 |
| Qwen2.5-0.5B | Corr2Cause pool | 15 | random (mean of 3) | 0.03 | 0.33 | 0.58 | 0.007 | – |
| Qwen2.5-0.5B | CLadder | 22 | gold probe | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.49 [0.44, 0.55] | 0.004 | 0.018 |
| Qwen2.5-0.5B | CLadder | 22 | gold diff-of-means | 0.05 [0.02, 0.08] | 0.05 [0.02, 0.08] | 0.00 [0.00, 0.00] | 0.284 | 0.404 |
| Qwen2.5-0.5B | CLadder | 22 | model output | 0.93 [0.89, 0.96] | 0.95 [0.92, 0.98] | 1.00 [1.00, 1.00] | 0.236 | 0.136 |
| Qwen2.5-0.5B | CLadder | 22 | random (mean of 3) | 0.06 | 0.35 | 0.67 | 0.006 | – |

<sub>Source: `results/steering_qwen0.5b_c2c_probe.json`, `results/steering_qwen0.5b_cladder.json`</sub>

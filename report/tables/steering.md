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
| Qwen2.5-1.5B | Corr2Cause pool | 21 | gold probe | 0.07 [0.04, 0.10] | 0.10 [0.07, 0.14] | 0.31 [0.26, 0.35] | 0.008 | 0.016 |
| Qwen2.5-1.5B | Corr2Cause pool | 21 | gold diff-of-means | 0.39 [0.34, 0.45] | 0.89 [0.85, 0.93] | 1.00 [1.00, 1.00] | 1.020 | 0.532 |
| Qwen2.5-1.5B | Corr2Cause pool | 21 | model output | 0.43 [0.38, 0.48] | 0.89 [0.85, 0.93] | 1.00 [1.00, 1.00] | 0.741 | 0.350 |
| Qwen2.5-1.5B | Corr2Cause pool | 21 | random (mean of 3) | 0.01 | 0.08 | 0.47 | 0.007 | – |
| Qwen2.5-1.5B | CLadder | 24 | gold probe | 0.01 [0.00, 0.02] | 0.03 [0.01, 0.05] | 0.54 [0.48, 0.61] | 0.005 | 0.007 |
| Qwen2.5-1.5B | CLadder | 24 | gold diff-of-means | 0.19 [0.15, 0.22] | 0.37 [0.31, 0.43] | 0.89 [0.85, 0.92] | 0.167 | 0.396 |
| Qwen2.5-1.5B | CLadder | 24 | model output | 0.36 [0.29, 0.42] | 0.37 [0.31, 0.43] | 1.00 [1.00, 1.00] | 0.614 | 0.194 |
| Qwen2.5-1.5B | CLadder | 24 | random (mean of 3) | 0.04 | 0.25 | 0.54 | 0.007 | – |

<sub>Source: `results/steering_qwen0.5b_c2c_probe.json`, `results/steering_qwen0.5b_cladder.json`, `results/steering_qwen1.5b_c2c_probe.json`, `results/steering_qwen1.5b_cladder.json`</sub>

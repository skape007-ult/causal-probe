| Model | Item set | Layer | Direction | Flip rate +4s | Flip rate −4s | Controllability towards gold (4s) | Controllability away from gold (4s) | Ablation mean |ΔLD| | Noise floor | Ablation above 3× floor |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold-label probe weights | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.50 [0.46, 0.55] | 0.50 [0.46, 0.55] | 0.0048 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold difference of class means | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.2683 | 0.0058 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | model-answer probe weights (positive control) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.0948 | 0.0058 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 0 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.50 [0.46, 0.55] | 0.50 [0.45, 0.54] | 0.0047 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 1 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.0100 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 2 | 0.00 [0.00, 0.00] | 0.57 [0.51, 0.63] | 0.24 [0.19, 0.28] | 0.81 [0.76, 0.85] | 0.0077 | 0.0058 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold-label probe weights | 0.10 [0.07, 0.14] | 0.30 [0.26, 0.35] | 0.31 [0.26, 0.35] | 0.71 [0.66, 0.76] | 0.0076 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold difference of class means | 0.89 [0.85, 0.93] | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.0198 | 0.0080 | yes |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | model-answer probe weights (positive control) | 0.89 [0.85, 0.93] | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.7409 | 0.0080 | yes |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 0 | 0.04 [0.02, 0.06] | 0.05 [0.03, 0.07] | 0.51 [0.47, 0.56] | 0.51 [0.47, 0.56] | 0.0071 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 1 | 0.08 [0.05, 0.12] | 0.21 [0.17, 0.26] | 0.35 [0.31, 0.39] | 0.64 [0.59, 0.69] | 0.0074 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 2 | 0.13 [0.10, 0.17] | 0.06 [0.03, 0.09] | 0.55 [0.51, 0.60] | 0.48 [0.43, 0.52] | 0.0065 | 0.0080 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | gold-label probe weights | 0.03 [0.01, 0.05] | 0.08 [0.05, 0.10] | 0.54 [0.48, 0.61] | 0.45 [0.39, 0.53] | 0.0047 | 0.0061 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | gold difference of class means | 0.37 [0.31, 0.43] | 0.39 [0.34, 0.44] | 0.89 [0.85, 0.92] | 0.13 [0.09, 0.18] | 0.1670 | 0.0061 | yes |
| Qwen2.5-1.5B | CLadder | 24/28 | model-answer probe weights (positive control) | 0.37 [0.31, 0.43] | 0.63 [0.57, 0.69] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.6141 | 0.0061 | yes |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 0 | 0.17 [0.13, 0.21] | 0.14 [0.10, 0.18] | 0.65 [0.58, 0.73] | 0.35 [0.29, 0.41] | 0.0076 | 0.0061 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 1 | 0.33 [0.27, 0.38] | 0.18 [0.13, 0.22] | 0.26 [0.20, 0.32] | 0.76 [0.71, 0.82] | 0.0053 | 0.0061 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 2 | 0.27 [0.22, 0.32] | 0.12 [0.08, 0.16] | 0.71 [0.65, 0.78] | 0.32 [0.26, 0.39] | 0.0069 | 0.0061 | no |

<sub>Source: `results/steering_analysis_qwen0.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_cladder.json`</sub>

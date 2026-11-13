| Model | Item set | Layer | Direction | Flip rate +4s | Flip rate −4s | Controllability towards gold (4s) | Controllability away from gold (4s) | Ablation mean |ΔLD| | Noise floor | Ablation above 3× floor |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold-label probe weights | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.50 [0.46, 0.55] | 0.50 [0.46, 0.55] | 0.0048 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | gold difference of class means | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.2683 | 0.0058 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | model-answer probe weights (positive control) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.0948 | 0.0058 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 0 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.50 [0.46, 0.55] | 0.50 [0.45, 0.54] | 0.0047 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 1 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.0100 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | random direction 2 | 0.00 [0.00, 0.00] | 0.57 [0.51, 0.63] | 0.24 [0.19, 0.28] | 0.81 [0.76, 0.85] | 0.0077 | 0.0058 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | gold-label probe weights | 0.12 [0.08, 0.16] | 0.00 [0.00, 0.00] | 0.58 [0.53, 0.63] | 0.46 [0.42, 0.51] | 0.0045 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | gold difference of class means | 0.00 [0.00, 0.00] | 0.06 [0.03, 0.09] | 0.49 [0.44, 0.53] | 0.54 [0.50, 0.59] | 0.0262 | 0.0059 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | model-answer probe weights (positive control) | 0.02 [0.01, 0.04] | 0.01 [0.00, 0.02] | 0.51 [0.47, 0.56] | 0.50 [0.46, 0.55] | 0.0043 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | random direction 0 | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] | 0.50 [0.45, 0.54] | 0.51 [0.47, 0.56] | 0.0050 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | random direction 1 | 0.15 [0.11, 0.19] | 0.00 [0.00, 0.00] | 0.61 [0.56, 0.66] | 0.46 [0.42, 0.51] | 0.0053 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 12/24 | random direction 2 | 0.27 [0.21, 0.33] | 0.00 [0.00, 0.00] | 0.67 [0.62, 0.72] | 0.40 [0.35, 0.45] | 0.0076 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | gold-label probe weights | 0.06 [0.03, 0.09] | 0.00 [0.00, 0.00] | 0.55 [0.50, 0.59] | 0.49 [0.45, 0.54] | 0.0044 | 0.0056 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | gold difference of class means | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.2374 | 0.0056 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | model-answer probe weights (positive control) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.1710 | 0.0056 | yes |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | random direction 0 | 0.07 [0.04, 0.10] | 0.00 [0.00, 0.00] | 0.55 [0.50, 0.60] | 0.49 [0.44, 0.53] | 0.0052 | 0.0056 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | random direction 1 | 0.36 [0.30, 0.42] | 0.00 [0.00, 0.00] | 0.71 [0.66, 0.77] | 0.36 [0.31, 0.41] | 0.0056 | 0.0056 | no |
| Qwen2.5-0.5B | Corr2Cause pool | 20/24 | random direction 2 | 0.00 [0.00, 0.00] | 0.05 [0.03, 0.08] | 0.49 [0.45, 0.54] | 0.54 [0.50, 0.59] | 0.0058 | 0.0056 | no |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | gold-label probe weights | 0.00 [0.00, 0.01] | 0.17 [0.12, 0.21] | 0.42 [0.37, 0.47] | 0.59 [0.54, 0.64] | 0.0046 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | gold difference of class means | 0.09 [0.06, 0.13] | 0.00 [0.00, 0.01] | 0.55 [0.49, 0.60] | 0.45 [0.40, 0.50] | 0.0144 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | model-answer probe weights (positive control) | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.01] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.01] | 0.0107 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | random direction 0 | 0.00 [0.00, 0.01] | 0.03 [0.01, 0.05] | 0.49 [0.44, 0.54] | 0.52 [0.47, 0.57] | 0.0046 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | random direction 1 | 0.13 [0.09, 0.16] | 0.00 [0.00, 0.01] | 0.56 [0.51, 0.61] | 0.43 [0.38, 0.48] | 0.0052 | 0.0059 | no |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14/24 | random direction 2 | 0.00 [0.00, 0.01] | 0.05 [0.03, 0.08] | 0.47 [0.42, 0.52] | 0.53 [0.48, 0.58] | 0.0052 | 0.0059 | no |
| Qwen2.5-0.5B | CLadder | 16/24 | gold-label probe weights | 0.12 [0.09, 0.16] | 0.04 [0.02, 0.06] | 0.59 [0.53, 0.64] | 0.43 [0.37, 0.48] | 0.0040 | 0.0048 | no |
| Qwen2.5-0.5B | CLadder | 16/24 | gold difference of class means | 0.05 [0.02, 0.08] | 0.95 [0.92, 0.98] | 0.00 [0.00, 0.01] | 1.00 [1.00, 1.00] | 0.2176 | 0.0048 | yes |
| Qwen2.5-0.5B | CLadder | 16/24 | model-answer probe weights (positive control) | 0.95 [0.92, 0.98] | 0.05 [0.02, 0.08] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.1021 | 0.0048 | yes |
| Qwen2.5-0.5B | CLadder | 16/24 | random direction 0 | 0.35 [0.28, 0.42] | 0.03 [0.01, 0.04] | 0.70 [0.64, 0.77] | 0.33 [0.28, 0.38] | 0.0052 | 0.0048 | no |
| Qwen2.5-0.5B | CLadder | 16/24 | random direction 1 | 0.68 [0.61, 0.76] | 0.04 [0.02, 0.07] | 0.88 [0.83, 0.92] | 0.15 [0.09, 0.20] | 0.0066 | 0.0048 | no |
| Qwen2.5-0.5B | CLadder | 16/24 | random direction 2 | 0.02 [0.00, 0.03] | 0.54 [0.49, 0.60] | 0.25 [0.20, 0.30] | 0.78 [0.72, 0.83] | 0.0041 | 0.0048 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold-label probe weights | 0.10 [0.07, 0.14] | 0.30 [0.26, 0.35] | 0.31 [0.26, 0.35] | 0.71 [0.66, 0.76] | 0.0076 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | gold difference of class means | 0.89 [0.85, 0.93] | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.0198 | 0.0080 | yes |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | model-answer probe weights (positive control) | 0.89 [0.85, 0.93] | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.7409 | 0.0080 | yes |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 0 | 0.04 [0.02, 0.06] | 0.05 [0.03, 0.07] | 0.51 [0.47, 0.56] | 0.51 [0.47, 0.56] | 0.0071 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 1 | 0.08 [0.05, 0.12] | 0.21 [0.17, 0.26] | 0.35 [0.31, 0.39] | 0.64 [0.59, 0.69] | 0.0074 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | random direction 2 | 0.13 [0.10, 0.17] | 0.06 [0.03, 0.09] | 0.55 [0.51, 0.60] | 0.48 [0.43, 0.52] | 0.0065 | 0.0080 | no |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | gold-label probe weights | 0.05 [0.03, 0.07] | 0.02 [0.00, 0.03] | 0.45 [0.40, 0.50] | 0.51 [0.46, 0.56] | 0.0042 | 0.0060 | no |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | gold difference of class means | 0.88 [0.84, 0.91] | 0.12 [0.09, 0.16] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.5521 | 0.0060 | yes |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | model-answer probe weights (positive control) | 0.88 [0.84, 0.91] | 0.12 [0.09, 0.16] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.4372 | 0.0060 | yes |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | random direction 0 | 0.04 [0.02, 0.07] | 0.03 [0.01, 0.05] | 0.52 [0.47, 0.57] | 0.44 [0.39, 0.49] | 0.0047 | 0.0060 | no |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | random direction 1 | 0.10 [0.07, 0.14] | 0.38 [0.32, 0.43] | 0.25 [0.20, 0.29] | 0.73 [0.68, 0.78] | 0.0088 | 0.0060 | no |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22/28 | random direction 2 | 0.24 [0.19, 0.28] | 0.02 [0.00, 0.03] | 0.60 [0.54, 0.65] | 0.37 [0.31, 0.42] | 0.0061 | 0.0060 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | gold-label probe weights | 0.03 [0.01, 0.05] | 0.08 [0.05, 0.10] | 0.54 [0.48, 0.61] | 0.45 [0.39, 0.53] | 0.0047 | 0.0061 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | gold difference of class means | 0.37 [0.31, 0.43] | 0.39 [0.34, 0.44] | 0.89 [0.85, 0.92] | 0.13 [0.09, 0.18] | 0.1670 | 0.0061 | yes |
| Qwen2.5-1.5B | CLadder | 24/28 | model-answer probe weights (positive control) | 0.37 [0.31, 0.43] | 0.63 [0.57, 0.69] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.6141 | 0.0061 | yes |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 0 | 0.17 [0.13, 0.21] | 0.14 [0.10, 0.18] | 0.65 [0.58, 0.73] | 0.35 [0.29, 0.41] | 0.0076 | 0.0061 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 1 | 0.33 [0.27, 0.38] | 0.18 [0.13, 0.22] | 0.26 [0.20, 0.32] | 0.76 [0.71, 0.82] | 0.0053 | 0.0061 | no |
| Qwen2.5-1.5B | CLadder | 24/28 | random direction 2 | 0.27 [0.22, 0.32] | 0.12 [0.08, 0.16] | 0.71 [0.65, 0.78] | 0.32 [0.26, 0.39] | 0.0069 | 0.0061 | no |
| Gemma-2-2B | Corr2Cause pool | 18/26 | gold-label probe weights | 0.04 [0.02, 0.07] | 0.16 [0.12, 0.21] | 0.54 [0.49, 0.58] | 0.42 [0.37, 0.46] | 0.0050 | 0.0062 | no |
| Gemma-2-2B | Corr2Cause pool | 18/26 | gold difference of class means | 0.47 [0.41, 0.53] | 0.28 [0.22, 0.33] | 0.89 [0.86, 0.93] | 0.15 [0.11, 0.19] | 1.1813 | 0.0062 | yes |
| Gemma-2-2B | Corr2Cause pool | 18/26 | model-answer probe weights (positive control) | 0.72 [0.67, 0.78] | 0.28 [0.22, 0.33] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.5263 | 0.0062 | yes |
| Gemma-2-2B | Corr2Cause pool | 18/26 | random direction 0 | 0.19 [0.14, 0.24] | 0.18 [0.14, 0.23] | 0.68 [0.63, 0.73] | 0.31 [0.27, 0.36] | 0.0263 | 0.0062 | yes |
| Gemma-2-2B | Corr2Cause pool | 18/26 | random direction 1 | 0.00 [0.00, 0.00] | 0.08 [0.05, 0.11] | 0.51 [0.46, 0.56] | 0.45 [0.40, 0.50] | 0.0073 | 0.0062 | no |
| Gemma-2-2B | Corr2Cause pool | 18/26 | random direction 2 | 0.04 [0.02, 0.06] | 0.16 [0.12, 0.21] | 0.56 [0.51, 0.61] | 0.41 [0.36, 0.46] | 0.0050 | 0.0062 | no |
| Gemma-2-2B | Corr2Cause pool | 13/26 | gold-label probe weights | 0.11 [0.08, 0.15] | 0.09 [0.06, 0.12] | 0.38 [0.34, 0.43] | 0.58 [0.54, 0.64] | 0.0050 | 0.0059 | no |
| Gemma-2-2B | Corr2Cause pool | 13/26 | gold difference of class means | 0.08 [0.05, 0.11] | 0.17 [0.13, 0.22] | 0.61 [0.56, 0.66] | 0.36 [0.31, 0.41] | 0.0547 | 0.0059 | yes |
| Gemma-2-2B | Corr2Cause pool | 13/26 | model-answer probe weights (positive control) | 0.04 [0.02, 0.07] | 0.13 [0.09, 0.17] | 0.57 [0.52, 0.62] | 0.40 [0.35, 0.45] | 0.0089 | 0.0059 | no |
| Gemma-2-2B | Corr2Cause pool | 13/26 | random direction 0 | 0.04 [0.02, 0.07] | 0.09 [0.06, 0.12] | 0.41 [0.37, 0.47] | 0.55 [0.50, 0.60] | 0.0052 | 0.0059 | no |
| Gemma-2-2B | Corr2Cause pool | 13/26 | random direction 1 | 0.12 [0.08, 0.16] | 0.06 [0.03, 0.08] | 0.39 [0.34, 0.44] | 0.55 [0.51, 0.60] | 0.0048 | 0.0059 | no |
| Gemma-2-2B | Corr2Cause pool | 13/26 | random direction 2 | 0.10 [0.07, 0.13] | 0.18 [0.13, 0.23] | 0.62 [0.57, 0.67] | 0.34 [0.30, 0.39] | 0.0056 | 0.0059 | no |
| Gemma-2-2B | Corr2Cause pool | 22/26 | gold-label probe weights | 0.02 [0.00, 0.03] | 0.05 [0.02, 0.07] | 0.46 [0.41, 0.52] | 0.50 [0.45, 0.55] | 0.0045 | 0.0058 | no |
| Gemma-2-2B | Corr2Cause pool | 22/26 | gold difference of class means | 0.72 [0.67, 0.78] | 0.28 [0.22, 0.33] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.7006 | 0.0058 | yes |
| Gemma-2-2B | Corr2Cause pool | 22/26 | model-answer probe weights (positive control) | 0.72 [0.67, 0.78] | 0.28 [0.22, 0.33] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.6990 | 0.0058 | yes |
| Gemma-2-2B | Corr2Cause pool | 22/26 | random direction 0 | 0.11 [0.07, 0.14] | 0.11 [0.07, 0.15] | 0.59 [0.54, 0.64] | 0.38 [0.33, 0.43] | 0.0068 | 0.0058 | no |
| Gemma-2-2B | Corr2Cause pool | 22/26 | random direction 1 | 0.04 [0.02, 0.06] | 0.04 [0.02, 0.06] | 0.48 [0.44, 0.53] | 0.50 [0.45, 0.55] | 0.0048 | 0.0058 | no |
| Gemma-2-2B | Corr2Cause pool | 22/26 | random direction 2 | 0.09 [0.06, 0.13] | 0.11 [0.07, 0.15] | 0.58 [0.53, 0.64] | 0.38 [0.33, 0.43] | 0.0051 | 0.0058 | no |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | gold-label probe weights | 0.24 [0.19, 0.28] | 0.19 [0.15, 0.23] | 0.70 [0.65, 0.75] | 0.27 [0.22, 0.32] | 0.0045 | 0.0063 | no |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | gold difference of class means | 0.24 [0.19, 0.28] | 0.68 [0.63, 0.74] | 0.04 [0.02, 0.06] | 0.96 [0.93, 0.98] | 0.4507 | 0.0063 | yes |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | model-answer probe weights (positive control) | 0.76 [0.72, 0.81] | 0.24 [0.19, 0.28] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.3792 | 0.0063 | yes |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | random direction 0 | 0.13 [0.09, 0.16] | 0.06 [0.04, 0.09] | 0.55 [0.49, 0.61] | 0.36 [0.31, 0.42] | 0.0152 | 0.0063 | no |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | random direction 1 | 0.08 [0.05, 0.11] | 0.06 [0.03, 0.08] | 0.53 [0.47, 0.58] | 0.40 [0.35, 0.46] | 0.0067 | 0.0063 | no |
| Gemma-2-2B | Corr2Cause debiased pool | 17/26 | random direction 2 | 0.07 [0.04, 0.10] | 0.08 [0.05, 0.11] | 0.48 [0.43, 0.54] | 0.48 [0.42, 0.53] | 0.0056 | 0.0063 | no |
| Gemma-2-2B | CLadder | 20/26 | gold-label probe weights | 0.04 [0.02, 0.06] | 0.22 [0.17, 0.28] | 0.42 [0.33, 0.50] | 0.65 [0.58, 0.73] | 0.0047 | 0.0052 | no |
| Gemma-2-2B | CLadder | 20/26 | gold difference of class means | 0.36 [0.30, 0.41] | 0.49 [0.40, 0.56] | 0.96 [0.94, 0.99] | 0.12 [0.06, 0.19] | 0.4658 | 0.0052 | yes |
| Gemma-2-2B | CLadder | 20/26 | model-answer probe weights (positive control) | 0.38 [0.32, 0.44] | 0.62 [0.56, 0.68] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.7086 | 0.0052 | yes |
| Gemma-2-2B | CLadder | 20/26 | random direction 0 | 0.33 [0.28, 0.38] | 0.07 [0.04, 0.11] | 0.67 [0.59, 0.74] | 0.39 [0.32, 0.48] | 0.0071 | 0.0052 | no |
| Gemma-2-2B | CLadder | 20/26 | random direction 1 | 0.04 [0.01, 0.06] | 0.09 [0.06, 0.12] | 0.57 [0.49, 0.66] | 0.52 [0.44, 0.59] | 0.0048 | 0.0052 | no |
| Gemma-2-2B | CLadder | 20/26 | random direction 2 | 0.16 [0.12, 0.21] | 0.26 [0.18, 0.33] | 0.75 [0.70, 0.80] | 0.33 [0.25, 0.43] | 0.0060 | 0.0052 | no |
| Gemma-2-2B | CLadder | 13/26 | gold-label probe weights | 0.12 [0.07, 0.16] | 0.10 [0.06, 0.13] | 0.42 [0.34, 0.51] | 0.63 [0.55, 0.70] | 0.0043 | 0.0053 | no |
| Gemma-2-2B | CLadder | 13/26 | gold difference of class means | 0.18 [0.13, 0.22] | 0.12 [0.08, 0.16] | 0.38 [0.29, 0.46] | 0.68 [0.60, 0.74] | 0.0613 | 0.0053 | yes |
| Gemma-2-2B | CLadder | 13/26 | model-answer probe weights (positive control) | 0.14 [0.10, 0.17] | 0.14 [0.09, 0.18] | 0.67 [0.61, 0.73] | 0.39 [0.31, 0.48] | 0.0051 | 0.0053 | no |
| Gemma-2-2B | CLadder | 13/26 | random direction 0 | 0.06 [0.03, 0.09] | 0.03 [0.01, 0.06] | 0.55 [0.48, 0.63] | 0.54 [0.46, 0.62] | 0.0047 | 0.0053 | no |
| Gemma-2-2B | CLadder | 13/26 | random direction 1 | 0.09 [0.05, 0.12] | 0.08 [0.04, 0.13] | 0.49 [0.39, 0.58] | 0.60 [0.52, 0.68] | 0.0046 | 0.0053 | no |
| Gemma-2-2B | CLadder | 13/26 | random direction 2 | 0.05 [0.03, 0.08] | 0.07 [0.04, 0.10] | 0.56 [0.48, 0.64] | 0.53 [0.46, 0.59] | 0.0046 | 0.0053 | no |
| Gemma-2-2B | CLadder | 22/26 | gold-label probe weights | 0.20 [0.14, 0.26] | 0.06 [0.03, 0.09] | 0.61 [0.53, 0.68] | 0.46 [0.37, 0.54] | 0.0040 | 0.0052 | no |
| Gemma-2-2B | CLadder | 22/26 | gold difference of class means | 0.38 [0.32, 0.44] | 0.49 [0.39, 0.57] | 0.96 [0.94, 0.99] | 0.10 [0.04, 0.17] | 0.3950 | 0.0052 | yes |
| Gemma-2-2B | CLadder | 22/26 | model-answer probe weights (positive control) | 0.38 [0.32, 0.44] | 0.62 [0.56, 0.68] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.8671 | 0.0052 | yes |
| Gemma-2-2B | CLadder | 22/26 | random direction 0 | 0.28 [0.22, 0.33] | 0.07 [0.05, 0.10] | 0.71 [0.65, 0.78] | 0.36 [0.28, 0.45] | 0.0088 | 0.0052 | no |
| Gemma-2-2B | CLadder | 22/26 | random direction 1 | 0.03 [0.00, 0.05] | 0.05 [0.02, 0.07] | 0.57 [0.50, 0.64] | 0.53 [0.45, 0.61] | 0.0042 | 0.0052 | no |
| Gemma-2-2B | CLadder | 22/26 | random direction 2 | 0.22 [0.16, 0.28] | 0.18 [0.12, 0.23] | 0.74 [0.68, 0.79] | 0.33 [0.24, 0.42] | 0.0062 | 0.0052 | no |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | gold-label probe weights | 0.03 [0.01, 0.06] | 0.03 [0.01, 0.05] | 0.50 [0.46, 0.55] | 0.50 [0.46, 0.55] | 0.0151 | 0.0131 | no |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | gold difference of class means | 0.76 [0.70, 0.81] | 0.03 [0.01, 0.06] | 0.94 [0.91, 0.96] | 0.15 [0.11, 0.19] | 2.9668 | 0.0131 | yes |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | model-answer probe weights (positive control) | 0.49 [0.43, 0.55] | 0.03 [0.01, 0.06] | 0.82 [0.77, 0.86] | 0.29 [0.24, 0.34] | 1.1681 | 0.0131 | yes |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | random direction 0 | 0.12 [0.08, 0.16] | 0.03 [0.01, 0.06] | 0.58 [0.53, 0.63] | 0.43 [0.38, 0.47] | 0.0218 | 0.0131 | no |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | random direction 1 | 0.02 [0.00, 0.03] | 0.01 [0.00, 0.02] | 0.51 [0.46, 0.56] | 0.52 [0.47, 0.57] | 0.0156 | 0.0131 | no |
| Qwen2.5-3B | Corr2Cause pool | 25/36 | random direction 2 | 0.03 [0.01, 0.06] | 0.00 [0.00, 0.01] | 0.49 [0.45, 0.53] | 0.53 [0.48, 0.57] | 0.0125 | 0.0131 | no |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | gold-label probe weights | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.02] | 0.51 [0.47, 0.56] | 0.52 [0.47, 0.57] | 0.0105 | 0.0136 | no |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | gold difference of class means | 0.02 [0.00, 0.03] | 0.01 [0.00, 0.02] | 0.53 [0.48, 0.57] | 0.50 [0.46, 0.55] | 0.0573 | 0.0136 | yes |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | model-answer probe weights (positive control) | 0.02 [0.01, 0.04] | 0.01 [0.00, 0.02] | 0.53 [0.48, 0.58] | 0.50 [0.45, 0.54] | 0.0177 | 0.0136 | no |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | random direction 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.52 [0.47, 0.57] | 0.51 [0.47, 0.56] | 0.0106 | 0.0136 | no |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | random direction 1 | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.00] | 0.52 [0.48, 0.57] | 0.51 [0.47, 0.56] | 0.0119 | 0.0136 | no |
| Qwen2.5-3B | Corr2Cause pool | 18/36 | random direction 2 | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.02] | 0.53 [0.48, 0.57] | 0.51 [0.46, 0.55] | 0.0112 | 0.0136 | no |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | gold-label probe weights | 0.07 [0.05, 0.10] | 0.02 [0.01, 0.04] | 0.55 [0.51, 0.60] | 0.46 [0.41, 0.50] | 0.0097 | 0.0114 | no |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | gold difference of class means | 0.20 [0.15, 0.25] | 0.03 [0.01, 0.06] | 0.62 [0.57, 0.67] | 0.39 [0.34, 0.44] | 3.2282 | 0.0114 | yes |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | model-answer probe weights (positive control) | 0.68 [0.62, 0.74] | 0.03 [0.01, 0.06] | 0.88 [0.84, 0.91] | 0.17 [0.13, 0.21] | 1.6372 | 0.0114 | yes |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | random direction 0 | 0.05 [0.02, 0.07] | 0.03 [0.01, 0.06] | 0.54 [0.49, 0.59] | 0.46 [0.42, 0.51] | 0.0129 | 0.0114 | no |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | random direction 1 | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.03] | 0.52 [0.47, 0.57] | 0.52 [0.48, 0.57] | 0.0090 | 0.0114 | no |
| Qwen2.5-3B | Corr2Cause pool | 31/36 | random direction 2 | 0.00 [0.00, 0.00] | 0.05 [0.02, 0.07] | 0.49 [0.44, 0.53] | 0.53 [0.48, 0.58] | 0.0096 | 0.0114 | no |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | gold-label probe weights | 0.03 [0.02, 0.06] | 0.03 [0.02, 0.05] | 0.52 [0.47, 0.58] | 0.46 [0.40, 0.51] | 0.0092 | 0.0125 | no |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | gold difference of class means | 0.81 [0.77, 0.85] | 0.03 [0.02, 0.05] | 0.91 [0.87, 0.94] | 0.06 [0.03, 0.09] | 0.6433 | 0.0125 | yes |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | model-answer probe weights (positive control) | 0.81 [0.77, 0.86] | 0.03 [0.02, 0.05] | 0.91 [0.88, 0.94] | 0.07 [0.04, 0.10] | 0.8207 | 0.0125 | yes |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | random direction 0 | 0.10 [0.07, 0.13] | 0.03 [0.02, 0.05] | 0.57 [0.51, 0.62] | 0.43 [0.38, 0.48] | 0.0205 | 0.0125 | no |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | random direction 1 | 0.02 [0.01, 0.04] | 0.01 [0.00, 0.03] | 0.48 [0.43, 0.53] | 0.50 [0.44, 0.55] | 0.0126 | 0.0125 | no |
| Qwen2.5-3B | Corr2Cause debiased pool | 25/36 | random direction 2 | 0.03 [0.02, 0.05] | 0.01 [0.00, 0.02] | 0.47 [0.42, 0.52] | 0.51 [0.46, 0.56] | 0.0128 | 0.0125 | no |
| Qwen2.5-3B | CLadder | 29/36 | gold-label probe weights | 0.09 [0.05, 0.13] | 0.23 [0.19, 0.27] | 0.71 [0.67, 0.74] | 0.39 [0.34, 0.44] | 0.0123 | 0.0125 | no |
| Qwen2.5-3B | CLadder | 29/36 | gold difference of class means | 0.49 [0.42, 0.55] | 0.51 [0.45, 0.58] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5.5224 | 0.0125 | yes |
| Qwen2.5-3B | CLadder | 29/36 | model-answer probe weights (positive control) | 0.49 [0.42, 0.55] | 0.51 [0.45, 0.58] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 2.6181 | 0.0125 | yes |
| Qwen2.5-3B | CLadder | 29/36 | random direction 0 | 0.06 [0.04, 0.08] | 0.10 [0.07, 0.13] | 0.64 [0.59, 0.67] | 0.47 [0.43, 0.51] | 0.0235 | 0.0125 | no |
| Qwen2.5-3B | CLadder | 29/36 | random direction 1 | 0.19 [0.15, 0.23] | 0.13 [0.10, 0.17] | 0.40 [0.35, 0.44] | 0.72 [0.68, 0.76] | 0.0308 | 0.0125 | no |
| Qwen2.5-3B | CLadder | 29/36 | random direction 2 | 0.12 [0.09, 0.16] | 0.21 [0.17, 0.25] | 0.39 [0.35, 0.43] | 0.72 [0.68, 0.76] | 0.0145 | 0.0125 | no |

<sub>Source: `results/steering_analysis_qwen0.5b_c2c_probe.json`, `results/steering_analysis_qwen0.5b_c2c_probe_d0.5.json`, `results/steering_analysis_qwen0.5b_c2c_probe_d0.85.json`, `results/steering_analysis_qwen0.5b_c2c_debiased.json`, `results/steering_analysis_qwen0.5b_cladder.json`, `results/steering_analysis_qwen1.5b_c2c_probe.json`, `results/steering_analysis_qwen1.5b_c2c_debiased.json`, `results/steering_analysis_qwen1.5b_cladder.json`, `results/steering_analysis_gemma2b_c2c_probe.json`, `results/steering_analysis_gemma2b_c2c_probe_d0.5.json`, `results/steering_analysis_gemma2b_c2c_probe_d0.85.json`, `results/steering_analysis_gemma2b_c2c_debiased.json`, `results/steering_analysis_gemma2b_cladder.json`, `results/steering_analysis_gemma2b_cladder_d0.5.json`, `results/steering_analysis_gemma2b_cladder_d0.85.json`, `results/steering_analysis_qwen3b_c2c_probe.json`, `results/steering_analysis_qwen3b_c2c_probe_d0.5.json`, `results/steering_analysis_qwen3b_c2c_probe_d0.85.json`, `results/steering_analysis_qwen3b_c2c_debiased.json`, `results/steering_analysis_qwen3b_cladder.json`</sub>

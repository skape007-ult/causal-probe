| Model | Dataset | Gold probe acc. | Strongest text baseline | Its acc. | Probe − baseline (acc., all items) | Probe − baseline (AUROC, model leans Yes) | Probe − baseline (AUROC, model leans No) | Audited criterion met | Pre-registered criterion met (as coded) | Pre-registered wrong-items test applicable |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 0.807 [0.784, 0.828] | structural | 0.919 [0.902, 0.935] | -0.113 [-0.136, -0.091] | -0.093 [-0.126, -0.063] | -0.069 [-0.093, -0.045] | no | no | no (100% gold-Yes) |
| Qwen2.5-0.5B | CLadder | 0.561 [0.534, 0.586] | structural | 0.800 [0.767, 0.839] | -0.240 [-0.285, -0.203] | -0.316 [-0.363, -0.271] | -0.252 [-0.320, -0.194] | no | no | no (97% gold-Yes) |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 0.633 [0.609, 0.658] | structural | 0.816 [0.792, 0.837] | -0.183 [-0.214, -0.149] | -0.215 [-0.257, -0.172] | -0.245 [-0.288, -0.201] | no | yes | no (99% gold-Yes) |
| Qwen2.5-1.5B | Corr2Cause pool | 0.870 [0.850, 0.890] | structural | 0.919 [0.902, 0.935] | -0.049 [-0.068, -0.032] | -0.030 [-0.052, -0.009] | -0.027 [-0.044, -0.011] | no | no | no (90% gold-Yes) |
| Qwen2.5-1.5B | CLadder | 0.605 [0.580, 0.631] | structural | 0.800 [0.767, 0.839] | -0.196 [-0.234, -0.162] | -0.174 [-0.214, -0.129] | -0.267 [-0.327, -0.216] | no | yes | yes |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 0.752 [0.729, 0.776] | structural | 0.816 [0.792, 0.837] | -0.063 [-0.094, -0.034] | -0.053 [-0.090, -0.015] | -0.094 [-0.125, -0.065] | no | yes | no (84% gold-Yes) |
| Gemma-2-2B | Corr2Cause pool | 0.863 [0.841, 0.886] | structural | 0.919 [0.902, 0.935] | -0.056 [-0.078, -0.033] | -0.020 [-0.039, -0.001] | -0.030 [-0.051, -0.008] | no | no | yes |
| Gemma-2-2B | CLadder | 0.626 [0.590, 0.664] | structural | 0.800 [0.767, 0.839] | -0.174 [-0.220, -0.133] | -0.159 [-0.215, -0.100] | -0.237 [-0.297, -0.185] | no | yes | yes |
| Qwen2.5-3B | Corr2Cause debiased pool | 0.765 [0.743, 0.788] | structural | 0.816 [0.792, 0.837] | -0.050 [-0.079, -0.020] | -0.039 [-0.080, +0.000] | -0.074 [-0.103, -0.047] | no | yes | no (95% gold-Yes) |

<sub>Source: `results/confound_qwen0.5b_c2c_probe.json`, `results/confound_qwen0.5b_cladder.json`, `results/confound_qwen0.5b_c2c_debiased.json`, `results/confound_qwen1.5b_c2c_probe.json`, `results/confound_qwen1.5b_cladder.json`, `results/confound_qwen1.5b_c2c_debiased.json`, `results/confound_gemma2b_c2c_probe.json`, `results/confound_gemma2b_cladder.json`, `results/confound_qwen3b_c2c_debiased.json`</sub>

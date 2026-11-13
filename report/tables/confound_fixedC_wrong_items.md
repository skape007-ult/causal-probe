| Model | Dataset | Model answer | n wrong | Gold probe acc. | TF-IDF acc. | Type prior acc. | Probe − TF-IDF | Probe − type prior | Criterion met |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | raw | 599 | 0.808 [0.770, 0.846] | 0.775 [0.741, 0.805] | 0.651 [0.608, 0.692] | +0.033 [-0.007, +0.072] | +0.157 [+0.108, +0.207] | no |
| Qwen2.5-0.5B | Corr2Cause pool | calibrated | 524 | 0.668 [0.629, 0.707] | 0.649 [0.612, 0.685] | 0.628 [0.584, 0.672] | +0.019 [-0.022, +0.059] | +0.040 [-0.008, +0.086] | no |
| Qwen2.5-0.5B | CLadder | raw | 810 | 0.564 [0.442, 0.689] | 0.499 [0.428, 0.575] | 0.502 [0.453, 0.557] | +0.065 [-0.038, +0.175] | +0.062 [-0.072, +0.194] | no |
| Qwen2.5-0.5B | CLadder | calibrated | 791 | 0.555 [0.500, 0.614] | 0.511 [0.449, 0.588] | 0.506 [0.458, 0.554] | +0.044 [-0.027, +0.102] | +0.049 [-0.030, +0.122] | no |

<sub>Source: `results/confound_qwen0.5b_c2c_probe_fixedC.json`, `results/confound_qwen0.5b_cladder_fixedC.json`</sub>

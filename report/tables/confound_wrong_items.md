| Model | Dataset | Model answer | n wrong | Gold probe acc. | TF-IDF acc. | Type prior acc. | Probe − TF-IDF | Probe − type prior | Criterion met |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | raw | 599 | 0.835 [0.798, 0.872] | 0.853 [0.821, 0.883] | 0.651 [0.608, 0.692] | -0.018 [-0.063, +0.028] | +0.184 [+0.127, +0.244] | no |
| Qwen2.5-0.5B | Corr2Cause pool | calibrated | 524 | 0.763 [0.723, 0.804] | 0.723 [0.685, 0.761] | 0.628 [0.584, 0.672] | +0.040 [-0.006, +0.085] | +0.135 [+0.079, +0.190] | no |
| Qwen2.5-0.5B | CLadder | raw | 810 | 0.567 [0.502, 0.631] | 0.528 [0.465, 0.592] | 0.502 [0.453, 0.557] | +0.038 [-0.035, +0.107] | +0.064 [-0.009, +0.137] | no |
| Qwen2.5-0.5B | CLadder | calibrated | 791 | 0.558 [0.507, 0.610] | 0.507 [0.457, 0.571] | 0.506 [0.458, 0.554] | +0.051 [-0.032, +0.116] | +0.052 [-0.007, +0.117] | no |

<sub>Source: `results/confound_qwen0.5b_c2c_probe.json`, `results/confound_qwen0.5b_cladder.json`</sub>

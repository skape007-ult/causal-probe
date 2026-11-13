| Model | Dataset | Layer | C | Gold probe acc. | TF-IDF acc. | Type prior acc. | Random-init probe acc. | Shuffled labels | Control task | Selectivity | Probe − TF-IDF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | 1 | 0.807 [0.784, 0.828] | 0.787 [0.762, 0.812] | 0.699 [0.672, 0.726] | 0.621 [0.595, 0.647] | 0.499 | 0.685 | +0.117 | +0.019 [-0.007, +0.046] |
| Qwen2.5-0.5B | CLadder | 16/24 | 0.1 | 0.561 [0.534, 0.586] | 0.527 [0.494, 0.562] | 0.484 [0.462, 0.504] | 0.498 [0.475, 0.522] | 0.510 | 0.791 | -0.234 | +0.034 [-0.009, +0.074] |

<sub>Source: `results/confound_qwen0.5b_c2c_probe.json`, `results/confound_qwen0.5b_cladder.json`, `results/probes_qwen0.5b_c2c_probe_C1.json`, `results/probes_qwen0.5b_cladder_C0.1.json`</sub>

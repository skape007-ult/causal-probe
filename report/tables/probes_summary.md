| Model | Dataset | Layer | C | Gold probe acc. | TF-IDF acc. | Type prior acc. | Random-init probe acc. | Shuffled labels (all items) | Control task (all items) | Selectivity (all items) | Probe − TF-IDF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 15/24 | 1 | 0.807 [0.784, 0.828] | 0.802 [0.778, 0.826] | 0.699 [0.672, 0.726] | 0.621 [0.595, 0.647] | 0.499 | 0.685 | +0.117 | +0.004 [-0.022, +0.030] |
| Qwen2.5-0.5B | CLadder | 16/24 | 0.1 | 0.561 [0.534, 0.586] | 0.527 [0.494, 0.566] | 0.484 [0.462, 0.504] | 0.498 [0.475, 0.522] | 0.510 | 0.791 | -0.234 | +0.034 [-0.014, +0.074] |
| Qwen2.5-1.5B | Corr2Cause pool | 21/28 | 0.1 | 0.870 [0.850, 0.890] | 0.802 [0.778, 0.826] | 0.699 [0.672, 0.726] | 0.594 [0.566, 0.624] | 0.498 | 0.777 | +0.088 | +0.068 [+0.042, +0.091] |
| Qwen2.5-1.5B | CLadder | 24/28 | 0.1 | 0.605 [0.580, 0.631] | 0.527 [0.494, 0.566] | 0.484 [0.462, 0.504] | 0.508 [0.484, 0.535] | 0.500 | 0.807 | -0.221 | +0.077 [+0.034, +0.115] |
| Gemma-2-2B | Corr2Cause pool | 18/26 | 1 | 0.863 [0.841, 0.886] | 0.802 [0.778, 0.826] | 0.699 [0.672, 0.726] | 0.596 [0.568, 0.623] | 0.499 | 0.790 | +0.071 | +0.061 [+0.035, +0.088] |
| Gemma-2-2B | CLadder | 20/26 | 1 | 0.626 [0.590, 0.664] | 0.527 [0.494, 0.566] | 0.484 [0.462, 0.504] | 0.519 [0.494, 0.548] | 0.499 | 0.812 | -0.194 | +0.099 [+0.038, +0.150] |

<sub>Source: `results/confound_gemma2b_c2c_probe.json`, `results/confound_gemma2b_cladder.json`, `results/confound_qwen0.5b_c2c_probe.json`, `results/confound_qwen0.5b_cladder.json`, `results/confound_qwen1.5b_c2c_probe.json`, `results/confound_qwen1.5b_cladder.json`, `results/probes_gemma2b_c2c_probe_C1.json`, `results/probes_gemma2b_cladder_C1.json`, `results/probes_qwen0.5b_c2c_probe_C1.json`, `results/probes_qwen0.5b_cladder_C0.1.json`, `results/probes_qwen1.5b_c2c_probe_C0.1.json`, `results/probes_qwen1.5b_cladder_C0.1.json`</sub>

<sub>Layer index counts the embeddings as 0, so layer l is the output of block l−1. Probe/TF-IDF/prior/random-init columns: report half, scores averaged over 3 seeds. Shuffled/control/selectivity: all items, mean of per-seed accuracies at the selected layer and C.</sub>

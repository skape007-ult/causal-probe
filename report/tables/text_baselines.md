| Model | Dataset | Gold probe | TF-IDF (full prompt) | Template / type prior | TF-IDF (question only) | Hypothesis-string prior | Parsed features + prior (GBM) | Random-init probe | Model's own answer |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause pool | 0.807 [0.784, 0.828] | 0.802 [0.778, 0.826] | 0.699 [0.672, 0.726] | 0.848 [0.827, 0.868] | 0.859 [0.839, 0.879] | 0.919 [0.902, 0.935] | 0.621 [0.595, 0.647] | 0.501 [0.500, 0.502] |
| Qwen2.5-0.5B | CLadder | 0.561 [0.534, 0.586] | 0.527 [0.494, 0.566] | 0.484 [0.462, 0.504] | 0.510 [0.482, 0.541] | – | 0.800 [0.767, 0.839] | 0.498 [0.475, 0.522] | 0.506 [0.483, 0.525] |
| Qwen2.5-1.5B | Corr2Cause pool | 0.870 [0.850, 0.890] | 0.802 [0.778, 0.826] | 0.699 [0.672, 0.726] | 0.848 [0.827, 0.868] | 0.859 [0.839, 0.879] | 0.919 [0.902, 0.935] | 0.594 [0.566, 0.624] | 0.505 [0.489, 0.520] |
| Qwen2.5-1.5B | CLadder | 0.605 [0.580, 0.631] | 0.527 [0.494, 0.566] | 0.484 [0.462, 0.504] | 0.510 [0.482, 0.541] | – | 0.800 [0.767, 0.839] | 0.508 [0.484, 0.535] | 0.506 [0.468, 0.545] |

<sub>Source: `results/confound_qwen0.5b_c2c_probe.json`, `results/confound_qwen0.5b_cladder.json`, `results/confound_qwen1.5b_c2c_probe.json`, `results/confound_qwen1.5b_cladder.json`</sub>

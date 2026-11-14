| Model | Group | Scorer | Accuracy | AUROC | Gold probe − this (accuracy, paired) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | — | **Gold-label probe, final token, layer 14** | 0.633 [0.611, 0.657] | 0.676 [0.650, 0.704] | — |
| Qwen2.5-0.5B | (a0) single-input control: 0.50 by construction | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.133 [+0.111, +0.157] |
| Qwen2.5-0.5B | (a0) single-input control: 0.50 by construction | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.133 [+0.111, +0.157] |
| Qwen2.5-0.5B | (a0) single-input control: 0.50 by construction | Bag-of-words, whole prompt (linear) | 0.489 [0.460, 0.516] | 0.514 [0.481, 0.545] | +0.144 [+0.110, +0.180] |
| Qwen2.5-0.5B | (a0) single-input control: 0.50 by construction | Probe trained on shuffled labels (20 shuffles) | 0.497 ± 0.013 (max 0.517) | – | – |
| Qwen2.5-0.5B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: premise-level counts | 0.593 [0.565, 0.619] | 0.619 [0.588, 0.650] | +0.040 [+0.006, +0.071] |
| Qwen2.5-0.5B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: bag-of-words of the premise | 0.659 [0.630, 0.688] | 0.706 [0.676, 0.736] | -0.026 [-0.062, +0.006] |
| Qwen2.5-0.5B | (a) shallow confound control: interaction, nothing bound to the queried variables | Probe on a randomly initialised copy of the model | 0.604 [0.580, 0.628] | 0.627 [0.599, 0.653] | +0.029 [+0.002, +0.060] |
| Qwen2.5-0.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statement counts for each queried variable separately | 0.743 [0.717, 0.768] | 0.819 [0.795, 0.843] | -0.110 [-0.141, -0.076] |
| Qwen2.5-0.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statements naming the queried pair (lookup) | 0.765 [0.741, 0.786] | 0.834 [0.811, 0.855] | -0.132 [-0.167, -0.098] |
| Qwen2.5-0.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–2-grams (no parser) | 0.742 [0.717, 0.767] | 0.821 [0.797, 0.844] | -0.109 [-0.141, -0.079] |
| Qwen2.5-0.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–4-grams (no parser) | 0.805 [0.784, 0.827] | 0.885 [0.866, 0.904] | -0.172 [-0.201, -0.141] |
| Qwen2.5-0.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand-written parse, all features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.183 [-0.216, -0.151] |
| Qwen2.5-0.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand parse stacked with per-string 4-grams (strongest text-only model here) | 0.865 [0.845, 0.884] | 0.944 [0.931, 0.955] | -0.232 [-0.260, -0.202] |
| Qwen2.5-1.5B | — | **Gold-label probe, final token, layer 22** | 0.752 [0.728, 0.774] | 0.829 [0.807, 0.850] | — |
| Qwen2.5-1.5B | (a0) single-input control: 0.50 by construction | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.252 [+0.228, +0.274] |
| Qwen2.5-1.5B | (a0) single-input control: 0.50 by construction | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.252 [+0.228, +0.274] |
| Qwen2.5-1.5B | (a0) single-input control: 0.50 by construction | Bag-of-words, whole prompt (linear) | 0.489 [0.460, 0.516] | 0.514 [0.481, 0.545] | +0.264 [+0.226, +0.300] |
| Qwen2.5-1.5B | (a0) single-input control: 0.50 by construction | Probe trained on shuffled labels (20 shuffles) | 0.498 ± 0.013 (max 0.519) | – | – |
| Qwen2.5-1.5B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: premise-level counts | 0.593 [0.565, 0.619] | 0.619 [0.588, 0.650] | +0.159 [+0.126, +0.192] |
| Qwen2.5-1.5B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: bag-of-words of the premise | 0.659 [0.630, 0.688] | 0.706 [0.676, 0.736] | +0.093 [+0.060, +0.125] |
| Qwen2.5-1.5B | (a) shallow confound control: interaction, nothing bound to the queried variables | Probe on a randomly initialised copy of the model | 0.579 [0.554, 0.604] | 0.617 [0.589, 0.646] | +0.174 [+0.142, +0.205] |
| Qwen2.5-1.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statement counts for each queried variable separately | 0.743 [0.717, 0.768] | 0.819 [0.795, 0.843] | +0.010 [-0.022, +0.038] |
| Qwen2.5-1.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statements naming the queried pair (lookup) | 0.765 [0.741, 0.786] | 0.834 [0.811, 0.855] | -0.013 [-0.045, +0.016] |
| Qwen2.5-1.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–2-grams (no parser) | 0.742 [0.717, 0.767] | 0.821 [0.797, 0.844] | +0.010 [-0.020, +0.038] |
| Qwen2.5-1.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–4-grams (no parser) | 0.805 [0.784, 0.827] | 0.885 [0.866, 0.904] | -0.053 [-0.080, -0.026] |
| Qwen2.5-1.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand-written parse, all features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.063 [-0.095, -0.036] |
| Qwen2.5-1.5B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand parse stacked with per-string 4-grams (strongest text-only model here) | 0.865 [0.845, 0.884] | 0.944 [0.931, 0.955] | -0.113 [-0.139, -0.088] |
| Gemma-2-2B | — | **Gold-label probe, final token, layer 17** | 0.763 [0.739, 0.787] | 0.840 [0.820, 0.859] | — |
| Gemma-2-2B | (a0) single-input control: 0.50 by construction | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.263 [+0.239, +0.287] |
| Gemma-2-2B | (a0) single-input control: 0.50 by construction | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.263 [+0.239, +0.287] |
| Gemma-2-2B | (a0) single-input control: 0.50 by construction | Bag-of-words, whole prompt (linear) | 0.489 [0.460, 0.516] | 0.514 [0.481, 0.545] | +0.274 [+0.238, +0.309] |
| Gemma-2-2B | (a0) single-input control: 0.50 by construction | Probe trained on shuffled labels (20 shuffles) | 0.500 ± 0.016 (max 0.524) | – | – |
| Gemma-2-2B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: premise-level counts | 0.593 [0.565, 0.619] | 0.619 [0.588, 0.650] | +0.170 [+0.136, +0.201] |
| Gemma-2-2B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: bag-of-words of the premise | 0.659 [0.630, 0.688] | 0.706 [0.676, 0.736] | +0.103 [+0.069, +0.139] |
| Gemma-2-2B | (a) shallow confound control: interaction, nothing bound to the queried variables | Probe on a randomly initialised copy of the model | 0.568 [0.541, 0.593] | 0.593 [0.563, 0.622] | +0.195 [+0.163, +0.228] |
| Gemma-2-2B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statement counts for each queried variable separately | 0.743 [0.717, 0.768] | 0.819 [0.795, 0.843] | +0.020 [-0.008, +0.049] |
| Gemma-2-2B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statements naming the queried pair (lookup) | 0.765 [0.741, 0.786] | 0.834 [0.811, 0.855] | -0.002 [-0.035, +0.030] |
| Gemma-2-2B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–2-grams (no parser) | 0.742 [0.717, 0.767] | 0.821 [0.797, 0.844] | +0.021 [-0.009, +0.052] |
| Gemma-2-2B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–4-grams (no parser) | 0.805 [0.784, 0.827] | 0.885 [0.866, 0.904] | -0.042 [-0.071, -0.012] |
| Gemma-2-2B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand-written parse, all features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.053 [-0.082, -0.025] |
| Gemma-2-2B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand parse stacked with per-string 4-grams (strongest text-only model here) | 0.865 [0.845, 0.884] | 0.944 [0.931, 0.955] | -0.103 [-0.131, -0.073] |
| Qwen2.5-3B | — | **Gold-label probe, final token, layer 25** | 0.765 [0.743, 0.788] | 0.845 [0.826, 0.866] | — |
| Qwen2.5-3B | (a0) single-input control: 0.50 by construction | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.265 [+0.243, +0.288] |
| Qwen2.5-3B | (a0) single-input control: 0.50 by construction | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.265 [+0.243, +0.288] |
| Qwen2.5-3B | (a0) single-input control: 0.50 by construction | Bag-of-words, whole prompt (linear) | 0.489 [0.460, 0.516] | 0.514 [0.481, 0.545] | +0.276 [+0.242, +0.312] |
| Qwen2.5-3B | (a0) single-input control: 0.50 by construction | Probe trained on shuffled labels (20 shuffles) | 0.499 ± 0.014 (max 0.525) | – | – |
| Qwen2.5-3B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: premise-level counts | 0.593 [0.565, 0.619] | 0.619 [0.588, 0.650] | +0.172 [+0.139, +0.206] |
| Qwen2.5-3B | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: bag-of-words of the premise | 0.659 [0.630, 0.688] | 0.706 [0.676, 0.736] | +0.106 [+0.072, +0.142] |
| Qwen2.5-3B | (a) shallow confound control: interaction, nothing bound to the queried variables | Probe on a randomly initialised copy of the model | 0.594 [0.568, 0.619] | 0.629 [0.600, 0.658] | +0.171 [+0.140, +0.204] |
| Qwen2.5-3B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statement counts for each queried variable separately | 0.743 [0.717, 0.768] | 0.819 [0.795, 0.843] | +0.022 [-0.006, +0.051] |
| Qwen2.5-3B | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statements naming the queried pair (lookup) | 0.765 [0.741, 0.786] | 0.834 [0.811, 0.855] | +0.000 [-0.030, +0.029] |
| Qwen2.5-3B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–2-grams (no parser) | 0.742 [0.717, 0.767] | 0.821 [0.797, 0.844] | +0.023 [-0.006, +0.055] |
| Qwen2.5-3B | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–4-grams (no parser) | 0.805 [0.784, 0.827] | 0.885 [0.866, 0.904] | -0.040 [-0.068, -0.013] |
| Qwen2.5-3B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand-written parse, all features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.050 [-0.081, -0.022] |
| Qwen2.5-3B | (b) structure-aware text reference: binds premise statements to the queried variables | Hand parse stacked with per-string 4-grams (strongest text-only model here) | 0.865 [0.845, 0.884] | 0.944 [0.931, 0.955] | -0.100 [-0.127, -0.073] |
| Qwen2.5-7B (4-bit) | — | **Gold-label probe, final token, layer 19** | 0.784 [0.762, 0.806] | 0.863 [0.844, 0.881] | — |
| Qwen2.5-7B (4-bit) | (a0) single-input control: 0.50 by construction | Hypothesis-string prior | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.284 [+0.262, +0.306] |
| Qwen2.5-7B (4-bit) | (a0) single-input control: 0.50 by construction | Bag-of-words, premise only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | +0.284 [+0.262, +0.306] |
| Qwen2.5-7B (4-bit) | (a0) single-input control: 0.50 by construction | Bag-of-words, whole prompt (linear) | 0.489 [0.460, 0.516] | 0.514 [0.481, 0.545] | +0.295 [+0.259, +0.332] |
| Qwen2.5-7B (4-bit) | (a0) single-input control: 0.50 by construction | Probe trained on shuffled labels (20 shuffles) | 0.500 ± 0.015 (max 0.529) | – | – |
| Qwen2.5-7B (4-bit) | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: premise-level counts | 0.593 [0.565, 0.619] | 0.619 [0.588, 0.650] | +0.191 [+0.160, +0.220] |
| Qwen2.5-7B (4-bit) | (a) shallow confound control: interaction, nothing bound to the queried variables | Per template: bag-of-words of the premise | 0.659 [0.630, 0.688] | 0.706 [0.676, 0.736] | +0.124 [+0.091, +0.155] |
| Qwen2.5-7B (4-bit) | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statement counts for each queried variable separately | 0.743 [0.717, 0.768] | 0.819 [0.795, 0.843] | +0.041 [+0.012, +0.067] |
| Qwen2.5-7B (4-bit) | (b) structure-aware text reference: binds premise statements to the queried variables | Per template: statements naming the queried pair (lookup) | 0.765 [0.741, 0.786] | 0.834 [0.811, 0.855] | +0.018 [-0.011, +0.049] |
| Qwen2.5-7B (4-bit) | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–2-grams (no parser) | 0.742 [0.717, 0.767] | 0.821 [0.797, 0.844] | +0.042 [+0.014, +0.071] |
| Qwen2.5-7B (4-bit) | (b) structure-aware text reference: binds premise statements to the queried variables | Per hypothesis string: bag of premise 1–4-grams (no parser) | 0.805 [0.784, 0.827] | 0.885 [0.866, 0.904] | -0.022 [-0.048, +0.004] |
| Qwen2.5-7B (4-bit) | (b) structure-aware text reference: binds premise statements to the queried variables | Hand-written parse, all features (gradient boosting) | 0.816 [0.792, 0.837] | 0.908 [0.891, 0.922] | -0.032 [-0.063, -0.005] |
| Qwen2.5-7B (4-bit) | (b) structure-aware text reference: binds premise statements to the queried variables | Hand parse stacked with per-string 4-grams (strongest text-only model here) | 0.865 [0.845, 0.884] | 0.944 [0.931, 0.955] | -0.082 [-0.107, -0.057] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`, `results/probe_content_gemma2b_c2c_debiased.json`, `results/probe_content_qwen3b_c2c_debiased.json`, `results/probe_content_qwen7b-4bit_c2c_debiased.json`</sub>

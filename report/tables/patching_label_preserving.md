| Model | Pair kind | Pairs | n | of which Yes/Yes, No/No | Mean D = LD(original) − LD(edited) | Mean |D| | Mean |D|: Yes/Yes, No/No | First layer with ≥ 0.5 recovery at final token (on the layers both runs have) | …as relative depth |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | hyp | label changes (Yes → No) | 60 | – | +0.009 [-0.060, +0.078] | 0.253 | – | 16 | 0.67 |
| Qwen2.5-0.5B | hyp | label preserved | 60 | 31 / 29 | +0.084 [+0.012, +0.156] | 0.233 | 0.219 / 0.249 | 16 | 0.67 |
| Qwen2.5-0.5B | prem | label changes (Yes → No) | 60 | – | +0.023 [-0.042, +0.090] | 0.236 | – | 16 | 0.67 |
| Qwen2.5-0.5B | prem | label preserved | 60 | 31 / 29 | +0.223 [+0.173, +0.270] | 0.239 | 0.264 / 0.212 | 16 | 0.67 |
| Qwen2.5-1.5B | hyp | label changes (Yes → No) | 40 | – | -0.088 [-0.420, +0.267] | 1.110 | – | 18 | 0.64 |
| Qwen2.5-1.5B | hyp | label preserved | 60 | 31 / 29 | +0.409 [+0.087, +0.731] | 0.994 | 1.267 / 0.703 | 18 | 0.64 |
| Qwen2.5-1.5B | prem | label changes (Yes → No) | 40 | – | -0.087 [-0.424, +0.269] | 1.055 | – | 18 | 0.64 |
| Qwen2.5-1.5B | prem | label preserved | 60 | 31 / 29 | -0.022 [-0.258, +0.190] | 0.686 | 0.787 / 0.577 | 18 | 0.64 |
| Qwen2.5-3B | hyp | label changes (Yes → No) | 60 | – | +0.673 [-0.022, +1.376] | 1.884 | – | 26 | 0.72 |
| Qwen2.5-3B | hyp | label preserved | 60 | 31 / 29 | +1.003 [+0.149, +1.886] | 2.643 | 2.899 / 2.370 | 26 | 0.72 |
| Qwen2.5-3B | prem | label changes (Yes → No) | 60 | – | +0.002 [-0.593, +0.580] | 1.875 | – | 26 | 0.72 |
| Qwen2.5-3B | prem | label preserved | 60 | 31 / 29 | +0.379 [-0.376, +1.065] | 1.947 | 1.916 / 1.981 | 26 | 0.72 |

<sub>Source: `results/patching_qwen0.5b_same.json`, `results/patching_qwen0.5b.json`, `results/patching_qwen1.5b_same.json`, `results/patching_qwen1.5b.json`, `results/patching_qwen3b_same.json`, `results/patching_qwen3b.json`</sub><br><sub>Mean |D| here is not standardised by hypothesis template; the two kinds of pair have different template mixes. The comparison within template is in `results/patching_by_template_<model>.json` (report 4.2).</sub><br><sub>The label-changing pairs of Qwen2.5-1.5B (40) are from an earlier run on the laptop and are a different draw from the 60 of the other runs.</sub>

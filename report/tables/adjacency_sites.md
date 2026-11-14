| Model | Site (layer) | Items | Within-question answer linearly predictable from this site (held-out R²) | Adjacency AUROC on held-out items: before | …after erasure | Adjacency AUROC on the evaluated items: before | …after erasure | (Probe trained on the half the eraser was fitted on: after erasure) | (…after removing the probe direction only) | Erasure effect vs −coefficient × translation effect (r over items) | Self-check: shortcut vs full forward pass (logits) | SD of the answer logit difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | 500 | 0.96 | 0.858 | 0.698 | 0.852 | 0.650 | 0.434 | 0.873 | 0.998 | 0.006 | 0.39 |
| Qwen2.5-0.5B | end of hypothesis, best layer (13/24) | 500 | 0.79 | 0.912 | 0.702 | 0.912 | 0.640 | 0.535 | 0.918 | 0.939 | 0.008 | 0.39 |
| Qwen2.5-0.5B | final token, earlier layer (11/24) | 500 | 0.43 | 0.773 | 0.692 | 0.745 | 0.642 | 0.421 | 0.743 | 0.876 | 0.010 | 0.39 |
| Qwen2.5-0.5B | end of hypothesis, earlier layer (5/24) | 500 | 0.31 | 0.793 | 0.691 | 0.757 | 0.618 | 0.564 | 0.807 | 0.549 | 0.007 | 0.39 |
| Qwen2.5-1.5B | final token, best layer (19/28) | 500 | 0.98 | 0.953 | 0.752 | 0.940 | 0.677 | 0.439 | 0.955 | 0.999 | 0.010 | 2.03 |
| Qwen2.5-1.5B | end of hypothesis, best layer (18/28) | 500 | 0.91 | 0.966 | 0.718 | 0.959 | 0.682 | 0.568 | 0.960 | 0.983 | 0.006 | 2.03 |
| Qwen2.5-1.5B | final token, earlier layer (10/28) | 500 | 0.38 | 0.792 | 0.700 | 0.778 | 0.657 | 0.417 | 0.784 | 0.893 | 0.005 | 2.03 |
| Qwen2.5-1.5B | end of hypothesis, earlier layer (7/28) | 500 | 0.51 | 0.815 | 0.701 | 0.801 | 0.647 | 0.569 | 0.814 | 0.373 | 0.014 | 2.03 |
| Qwen2.5-3B | final token, best layer (25/36) | 500 | 0.96 | 0.964 | 0.736 | 0.952 | 0.699 | 0.512 | 0.947 | 0.999 | 0.027 | 7.83 |
| Qwen2.5-3B | end of hypothesis, best layer (25/36) | 500 | 0.89 | 0.980 | 0.761 | 0.972 | 0.672 | 0.554 | 0.965 | 0.974 | 0.016 | 7.83 |
| Qwen2.5-3B | final token, earlier layer (19/36) | 500 | 0.40 | 0.828 | 0.702 | 0.827 | 0.669 | 0.427 | 0.818 | 0.984 | 0.026 | 7.83 |
| Qwen2.5-3B | end of hypothesis, earlier layer (6/36) | 500 | 0.41 | 0.827 | 0.724 | 0.786 | 0.648 | 0.428 | 0.817 | 0.439 | 0.019 | 7.83 |

<sub>Source: `results/adjacency_qwen0.5b_c2c_debiased_final_best.json`, `results/adjacency_qwen0.5b_c2c_debiased_hyp_best.json`, `results/adjacency_qwen0.5b_c2c_debiased_final_first50.json`, `results/adjacency_qwen0.5b_c2c_debiased_hyp_first50.json`, `results/adjacency_qwen1.5b_c2c_debiased_final_best.json`, `results/adjacency_qwen1.5b_c2c_debiased_hyp_best.json`, `results/adjacency_qwen1.5b_c2c_debiased_final_first50.json`, `results/adjacency_qwen1.5b_c2c_debiased_hyp_first50.json`, `results/adjacency_qwen3b_c2c_debiased_final_best.json`, `results/adjacency_qwen3b_c2c_debiased_hyp_best.json`, `results/adjacency_qwen3b_c2c_debiased_final_first50.json`, `results/adjacency_qwen3b_c2c_debiased_hyp_first50.json`</sub>

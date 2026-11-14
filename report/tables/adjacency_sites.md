| Model | Site (layer) | Items | Within-question answer linearly predictable from this site (held-out R²) | Adjacency AUROC on held-out items: before | …after erasure | Adjacency AUROC on the evaluated items: before | …after erasure | (Probe trained on the half the eraser was fitted on: after erasure) | (…after removing the probe direction only) | Erasure effect vs −coefficient × translation effect (r over items) | Self-check: shortcut vs full forward pass (logits) | SD of the answer logit difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | 500 | 0.96 | 0.858 | 0.698 | 0.852 | 0.650 | 0.434 | 0.873 | 0.998 | 0.006 | 0.39 |
| Qwen2.5-0.5B | end of hypothesis, best layer (13/24) | 500 | 0.79 | 0.912 | 0.702 | 0.912 | 0.640 | 0.535 | 0.918 | 0.939 | 0.008 | 0.39 |
| Qwen2.5-0.5B | final token, earlier layer (11/24) | 500 | 0.43 | 0.773 | 0.692 | 0.745 | 0.642 | 0.421 | 0.743 | 0.876 | 0.010 | 0.39 |
| Qwen2.5-0.5B | end of hypothesis, earlier layer (5/24) | 500 | 0.31 | 0.793 | 0.691 | 0.757 | 0.618 | 0.564 | 0.807 | 0.549 | 0.007 | 0.39 |

<sub>Source: `results/adjacency_qwen0.5b_c2c_debiased_final_best.json`, `results/adjacency_qwen0.5b_c2c_debiased_hyp_best.json`, `results/adjacency_qwen0.5b_c2c_debiased_final_first50.json`, `results/adjacency_qwen0.5b_c2c_debiased_hyp_first50.json`</sub>

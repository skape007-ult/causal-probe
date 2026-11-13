| Model | Site (layer) | Items | Adjacency AUROC of a fresh probe, report half: before | …after LEACE erasure | …after removing the probe direction only | …after 5 control erasures (shuffled labels) | AUROC along the probe direction, by template (range) | AUROC along the template's class-mean difference (range) | Size of the erasure edit (units of s) | …control erasures | Self-check: shortcut vs full forward pass (logits) | Max |unsteered − cached| logit difference | SD of the answer logit difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | 500 | 0.876 | 0.434 | 0.873 | 0.872 to 0.877 | 0.79 to 0.96 | 0.60 to 0.74 | 0.099 | 0.031 to 0.049 | 0.006 | 0.05 | 0.39 |

<sub>Source: `results/adjacency_qwen0.5b_c2c_debiased_final_best.json`</sub>

| Model | Site (layer) | Adjacency AUROC, selection half (CV) | …report half, probe direction | …report half, difference of means | Class gap along probe direction (own SDs) | …along difference of means | cos(probe, diff. of means) | cos(adjacency probe, gold probe) | cos(adjacency probe, model-answer probe) | SD of the answer logit difference | Max |unsteered − cached| logit difference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | 0.840 | 0.876 | 0.609 | 1.54 | 0.30 | 0.09 | -0.05 | -0.03 | 0.38 | 0.04 |

<sub>Source: `results/feature_causal_qwen0.5b_c2c_debiased_adj_final_best.json`</sub>

| Model | Parsed features only | Parsed features + probe score | Gain in accuracy | Gain in AUROC | Probe right when parsed features wrong | Parsed features right when probe wrong | Correlation of the two scores |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.816 [0.792, 0.837] | 0.814 [0.792, 0.836] | -0.002 [-0.019, +0.016] | -0.001 [-0.009, +0.008] | 0.57 (n = 230) | 0.78 (n = 458) | 0.36 |
| Qwen2.5-1.5B | 0.816 [0.792, 0.837] | 0.822 [0.799, 0.843] | +0.006 [-0.015, +0.026] | +0.005 [-0.005, +0.015] | 0.63 (n = 230) | 0.73 (n = 309) | 0.59 |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

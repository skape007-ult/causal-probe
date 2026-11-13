| Model | Probe added to | Base accuracy | Base + probe | Gain in accuracy (report half) | Gain in AUROC (report half) | Gain in accuracy (all items) |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | hand-written parse | 0.817 [0.794, 0.840] | 0.826 [0.803, 0.847] | +0.009 [-0.002, +0.018] | +0.003 [-0.000, +0.006] | +0.006 [-0.001, +0.013] |
| Qwen2.5-0.5B | strongest text-only model (parse + per-string 4-grams) | 0.865 [0.845, 0.884] | 0.868 [0.849, 0.886] | +0.002 [-0.003, +0.008] | +0.000 [-0.000, +0.001] | -0.000 [-0.004, +0.003] |
| Qwen2.5-1.5B | hand-written parse | 0.817 [0.794, 0.840] | 0.839 [0.818, 0.858] | +0.022 [+0.005, +0.038] | +0.017 [+0.010, +0.024] | +0.019 [+0.008, +0.029] |
| Qwen2.5-1.5B | strongest text-only model (parse + per-string 4-grams) | 0.865 [0.845, 0.884] | 0.865 [0.844, 0.884] | -0.001 [-0.010, +0.008] | +0.003 [+0.001, +0.005] | -0.001 [-0.008, +0.005] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

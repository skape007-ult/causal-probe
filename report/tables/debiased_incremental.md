| Model | Probe added to | Base accuracy | Base + probe | Gain in accuracy (report half) | Gain in AUROC (report half) | Gain in accuracy (all items) |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | hand-written parse | 0.817 [0.794, 0.840] | 0.826 [0.803, 0.847] | +0.009 [-0.002, +0.018] | +0.003 [-0.000, +0.006] | +0.006 [-0.001, +0.013] |
| Qwen2.5-0.5B | strongest text-only model (parse + per-string 4-grams) | 0.865 [0.845, 0.884] | 0.868 [0.849, 0.886] | +0.002 [-0.003, +0.008] | +0.000 [-0.000, +0.001] | -0.000 [-0.004, +0.003] |
| Qwen2.5-1.5B | hand-written parse | 0.817 [0.794, 0.840] | 0.839 [0.818, 0.858] | +0.022 [+0.005, +0.038] | +0.017 [+0.010, +0.024] | +0.019 [+0.008, +0.029] |
| Qwen2.5-1.5B | strongest text-only model (parse + per-string 4-grams) | 0.865 [0.845, 0.884] | 0.865 [0.844, 0.884] | -0.001 [-0.010, +0.008] | +0.003 [+0.001, +0.005] | -0.001 [-0.008, +0.005] |
| Gemma-2-2B | hand-written parse | 0.817 [0.794, 0.840] | 0.852 [0.832, 0.870] | +0.034 [+0.021, +0.048] | +0.020 [+0.014, +0.026] | +0.025 [+0.015, +0.035] |
| Gemma-2-2B | strongest text-only model (parse + per-string 4-grams) | 0.865 [0.845, 0.884] | 0.874 [0.855, 0.891] | +0.009 [-0.002, +0.019] | +0.003 [+0.001, +0.006] | +0.007 [+0.000, +0.014] |
| Qwen2.5-3B | hand-written parse | 0.817 [0.794, 0.840] | 0.850 [0.829, 0.869] | +0.033 [+0.018, +0.047] | +0.022 [+0.016, +0.029] | +0.020 [+0.008, +0.031] |
| Qwen2.5-3B | strongest text-only model (parse + per-string 4-grams) | 0.865 [0.845, 0.884] | 0.874 [0.856, 0.893] | +0.009 [+0.000, +0.018] | +0.004 [+0.002, +0.007] | +0.005 [-0.002, +0.012] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`, `results/probe_content_gemma2b_c2c_debiased.json`, `results/probe_content_qwen3b_c2c_debiased.json`</sub>

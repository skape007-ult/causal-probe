| Model | Item set | Layer | Property read off the prompt by a parser | Majority rate | Probe accuracy, final token | Probe accuracy, end of hypothesis | Gold-label accuracy from this property alone |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14 | the two named variables correlate | 0.923 | 0.959 [0.948, 0.970] | 0.983 [0.975, 0.990] | 0.525 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14 | they are adjacent in the skeleton (never separated) | 0.597 | 0.825 [0.803, 0.845] | 0.872 [0.854, 0.889] | 0.539 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14 | some conditioning set separates them | 0.600 | 0.821 [0.799, 0.842] | 0.872 [0.854, 0.889] | 0.544 |
| Qwen2.5-0.5B | Corr2Cause debiased pool | 14 | first-named variable is earlier in the alphabet | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.500 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22 | the two named variables correlate | 0.923 | 0.987 [0.980, 0.994] | 0.986 [0.978, 0.992] | 0.525 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22 | they are adjacent in the skeleton (never separated) | 0.597 | 0.928 [0.913, 0.942] | 0.931 [0.916, 0.944] | 0.539 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22 | some conditioning set separates them | 0.600 | 0.933 [0.917, 0.946] | 0.932 [0.917, 0.945] | 0.544 |
| Qwen2.5-1.5B | Corr2Cause debiased pool | 22 | first-named variable is earlier in the alphabet | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.500 |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

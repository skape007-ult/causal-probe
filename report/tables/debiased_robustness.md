| Model | Probe accuracy, 3-seed ensemble (report half) | Single seeds | Shuffled-label null (20 shuffles) | Probe trained on the selection half only |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.633 | 0.624 / 0.634 / 0.624 | 0.497 ± 0.013 | 0.620 [0.596, 0.646] |
| Qwen2.5-1.5B | 0.752 | 0.736 / 0.741 / 0.754 | 0.498 ± 0.013 | 0.708 [0.684, 0.730] |
| Gemma-2-2B | 0.763 | 0.746 / 0.754 / 0.759 | 0.500 ± 0.016 | 0.712 [0.687, 0.736] |
| Qwen2.5-3B | 0.765 | 0.754 / 0.754 / 0.763 | 0.499 ± 0.014 | 0.714 [0.690, 0.736] |
| Qwen2.5-7B (4-bit) | 0.784 | 0.783 / 0.773 / 0.778 | 0.500 ± 0.015 | 0.746 [0.723, 0.768] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`, `results/probe_content_gemma2b_c2c_debiased.json`, `results/probe_content_qwen3b_c2c_debiased.json`, `results/probe_content_qwen7b-4bit_c2c_debiased.json`</sub>

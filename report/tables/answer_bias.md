| Model | Item set | Prompt | Items | Yes-rate | Accuracy | Majority-class accuracy | AUROC of Yes−No logit difference | Accuracy at best threshold (cross-fitted) | Reading |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause test | zero-shot | 1162 | 0.00 | 0.845 [0.803, 0.898] | 0.845 | 0.589 [0.531, 0.660] | 0.843 [0.801, 0.895] | informative (AUROC CI above 0.5) |
| Qwen2.5-0.5B | Corr2Cause pool | zero-shot | 2400 | 0.00 | 0.502 [0.500, 0.504] | 0.500 | 0.591 [0.569, 0.614] | 0.578 [0.558, 0.598] | informative (AUROC CI above 0.5) |
| Qwen2.5-0.5B | Corr2Cause debiased pool | zero-shot | 2400 | 0.00 | 0.498 [0.496, 0.501] | 0.500 | 0.505 [0.485, 0.526] | 0.498 [0.484, 0.512] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-0.5B | CLadder | zero-shot | 3000 | 0.04 | 0.500 [0.483, 0.515] | 0.500 | 0.499 [0.468, 0.527] | 0.509 [0.484, 0.531] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-0.5B | Corr2Cause test, variables renamed | zero-shot, variables renamed | 1162 | 0.01 | 0.842 [0.800, 0.895] | 0.845 | 0.511 [0.455, 0.572] | 0.843 [0.800, 0.896] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-0.5B | Corr2Cause test (500 items) | 4-shot | 500 | 0.81 | 0.314 [0.260, 0.367] | 0.840 | 0.660 [0.579, 0.741] | 0.846 [0.814, 0.887] | informative (AUROC CI above 0.5) |
| Qwen2.5-0.5B | CLadder (500 items) | 4-shot | 500 | 0.20 | 0.516 [0.481, 0.552] | 0.500 | 0.495 [0.451, 0.542] | 0.486 [0.452, 0.521] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-1.5B | Corr2Cause test | zero-shot | 1162 | 0.10 | 0.773 [0.711, 0.846] | 0.845 | 0.561 [0.503, 0.614] | 0.843 [0.801, 0.895] | informative (AUROC CI above 0.5) |
| Qwen2.5-1.5B | Corr2Cause pool | zero-shot | 2400 | 0.10 | 0.500 [0.488, 0.512] | 0.500 | 0.568 [0.545, 0.590] | 0.565 [0.551, 0.580] | informative (AUROC CI above 0.5) |
| Qwen2.5-1.5B | Corr2Cause debiased pool | zero-shot | 2400 | 0.16 | 0.492 [0.478, 0.507] | 0.500 | 0.510 [0.488, 0.533] | 0.511 [0.495, 0.526] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-1.5B | CLadder | zero-shot | 3000 | 0.64 | 0.497 [0.474, 0.519] | 0.500 | 0.509 [0.478, 0.542] | 0.499 [0.480, 0.519] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-1.5B | Corr2Cause test, variables renamed | zero-shot, variables renamed | 1162 | 0.17 | 0.729 [0.667, 0.803] | 0.845 | 0.579 [0.523, 0.630] | 0.843 [0.800, 0.897] | informative (AUROC CI above 0.5) |
| Qwen2.5-1.5B | Corr2Cause test (500 items) | 4-shot | 500 | 0.05 | 0.800 [0.752, 0.856] | 0.840 | 0.556 [0.502, 0.606] | 0.818 [0.770, 0.877] | informative (AUROC CI above 0.5) |
| Qwen2.5-1.5B | CLadder (500 items) | 4-shot | 500 | 0.72 | 0.528 [0.487, 0.565] | 0.500 | 0.559 [0.511, 0.601] | 0.496 [0.460, 0.533] | informative (AUROC CI above 0.5) |
| Gemma-2-2B | Corr2Cause test | zero-shot | 1162 | 0.34 | 0.615 [0.573, 0.658] | 0.845 | 0.536 [0.495, 0.566] | 0.843 [0.800, 0.897] | at chance (AUROC CI includes 0.5) |
| Gemma-2-2B | Corr2Cause pool | zero-shot | 2400 | 0.25 | 0.505 [0.487, 0.525] | 0.500 | 0.559 [0.537, 0.582] | 0.565 [0.553, 0.576] | informative (AUROC CI above 0.5) |
| Gemma-2-2B | Corr2Cause debiased pool | zero-shot | 2400 | 0.31 | 0.496 [0.477, 0.515] | 0.500 | 0.492 [0.468, 0.515] | 0.497 [0.481, 0.513] | at chance (AUROC CI includes 0.5) |
| Gemma-2-2B | CLadder | zero-shot | 3000 | 0.63 | 0.508 [0.482, 0.535] | 0.500 | 0.519 [0.482, 0.557] | 0.500 [0.483, 0.520] | at chance (AUROC CI includes 0.5) |
| Gemma-2-2B | Corr2Cause test, variables renamed | zero-shot, variables renamed | 1162 | 0.32 | 0.602 [0.562, 0.641] | 0.845 | 0.504 [0.464, 0.531] | 0.844 [0.801, 0.897] | at chance (AUROC CI includes 0.5) |
| Gemma-2-2B | Corr2Cause test (500 items) | 4-shot | 500 | 0.54 | 0.486 [0.445, 0.526] | 0.840 | 0.569 [0.523, 0.613] | 0.834 [0.793, 0.886] | informative (AUROC CI above 0.5) |
| Gemma-2-2B | CLadder (500 items) | 4-shot | 500 | 0.83 | 0.504 [0.458, 0.546] | 0.500 | 0.543 [0.498, 0.585] | 0.504 [0.465, 0.543] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-3B | Corr2Cause test | zero-shot | 1162 | 0.05 | 0.812 [0.768, 0.862] | 0.845 | 0.534 [0.493, 0.573] | 0.838 [0.794, 0.892] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-3B | Corr2Cause pool | zero-shot | 2400 | 0.05 | 0.509 [0.500, 0.519] | 0.500 | 0.581 [0.559, 0.603] | 0.593 [0.573, 0.613] | informative (AUROC CI above 0.5) |
| Qwen2.5-3B | Corr2Cause debiased pool | zero-shot | 2400 | 0.04 | 0.495 [0.487, 0.503] | 0.500 | 0.507 [0.484, 0.530] | 0.506 [0.490, 0.524] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-3B | CLadder † | zero-shot | 3000 | 0.50 | 0.526 [0.502, 0.548] | 0.500 | 0.543 [0.518, 0.568] | 0.507 [0.487, 0.528] | informative (AUROC CI above 0.5) |
| Qwen2.5-3B | Corr2Cause test, variables renamed | zero-shot, variables renamed | 1162 | 0.04 | 0.822 [0.783, 0.868] | 0.845 | 0.512 [0.463, 0.553] | 0.844 [0.802, 0.896] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-3B | Corr2Cause test (500 items) | 4-shot | 500 | 0.46 | 0.556 [0.517, 0.596] | 0.840 | 0.538 [0.507, 0.569] | 0.838 [0.797, 0.890] | informative (AUROC CI above 0.5) |
| Qwen2.5-3B | CLadder (500 items) † | 4-shot | 500 | 0.56 | 0.560 [0.518, 0.602] | 0.500 | 0.564 [0.517, 0.608] | 0.510 [0.476, 0.543] | informative (AUROC CI above 0.5) |
| Qwen2.5-7B (4-bit) | Corr2Cause test | zero-shot | 1162 | 0.08 | 0.786 [0.741, 0.839] | 0.845 | 0.581 [0.545, 0.611] | 0.844 [0.802, 0.897] | informative (AUROC CI above 0.5) |
| Qwen2.5-7B (4-bit) | Corr2Cause pool | zero-shot | 2400 | 0.09 | 0.496 [0.485, 0.508] | 0.500 | 0.594 [0.572, 0.618] | 0.601 [0.582, 0.620] | informative (AUROC CI above 0.5) |
| Qwen2.5-7B (4-bit) | Corr2Cause debiased pool | zero-shot | 2400 | 0.07 | 0.506 [0.496, 0.517] | 0.500 | 0.514 [0.490, 0.536] | 0.525 [0.505, 0.543] | at chance (AUROC CI includes 0.5) |
| Qwen2.5-7B (4-bit) | CLadder | zero-shot | 3000 | 0.49 | 0.544 [0.520, 0.568] | 0.500 | 0.570 [0.541, 0.601] | 0.534 [0.511, 0.555] | informative (AUROC CI above 0.5) |

<sub>Source: `results/answer_bias.json`</sub><br><sub>† computed on the GPU machine's draw of the 3,000-item CLadder sample (2,271 items shared with the other models' draw; Section 7).</sub>

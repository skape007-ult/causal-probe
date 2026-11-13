| Model | Stratified by | Strata used | Gold accuracy from the strata alone | Gold-probe AUROC, unstratified | Gold-probe AUROC within strata (pair-weighted) | Lowest single stratum |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | hypothesis template (5) | 5 | 0.500 | 0.676 [0.650, 0.704] | 0.691 [0.663, 0.719] | 0.535 (n = 156) |
| Qwen2.5-0.5B | pair adjacent in skeleton | 2 | 0.539 | 0.676 [0.650, 0.704] | 0.671 [0.645, 0.698] | 0.656 (n = 727) |
| Qwen2.5-0.5B | pair has a separating set | 2 | 0.544 | 0.676 [0.650, 0.704] | 0.671 [0.644, 0.698] | 0.657 (n = 733) |
| Qwen2.5-0.5B | pair correlated | 2 | 0.525 | 0.676 [0.650, 0.704] | 0.665 [0.637, 0.694] | 0.664 (n = 1143) |
| Qwen2.5-0.5B | pair has a common neighbour | 2 | 0.553 | 0.676 [0.650, 0.704] | 0.667 [0.637, 0.697] | 0.664 (n = 1038) |
| Qwen2.5-0.5B | common neighbour in no separating set | 2 | 0.510 | 0.676 [0.650, 0.704] | 0.668 [0.641, 0.699] | 0.665 (n = 1039) |
| Qwen2.5-0.5B | parsed-feature classifier's prediction | 2 | 0.816 | 0.676 [0.650, 0.704] | 0.631 [0.594, 0.670] | 0.621 (n = 624) |
| Qwen2.5-0.5B | template × adjacent × common neighbour | 11 | 0.760 | 0.676 [0.650, 0.704] | 0.623 [0.581, 0.663] | 0.402 (n = 55) |
| Qwen2.5-1.5B | hypothesis template (5) | 5 | 0.500 | 0.829 [0.807, 0.850] | 0.835 [0.813, 0.856] | 0.772 (n = 260) |
| Qwen2.5-1.5B | pair adjacent in skeleton | 2 | 0.539 | 0.829 [0.807, 0.850] | 0.824 [0.801, 0.845] | 0.809 (n = 727) |
| Qwen2.5-1.5B | pair has a separating set | 2 | 0.544 | 0.829 [0.807, 0.850] | 0.823 [0.800, 0.844] | 0.808 (n = 733) |
| Qwen2.5-1.5B | pair correlated | 2 | 0.525 | 0.829 [0.807, 0.850] | 0.819 [0.794, 0.841] | 0.818 (n = 1143) |
| Qwen2.5-1.5B | pair has a common neighbour | 2 | 0.553 | 0.829 [0.807, 0.850] | 0.821 [0.796, 0.843] | 0.818 (n = 1038) |
| Qwen2.5-1.5B | common neighbour in no separating set | 2 | 0.510 | 0.829 [0.807, 0.850] | 0.821 [0.797, 0.843] | 0.818 (n = 1039) |
| Qwen2.5-1.5B | parsed-feature classifier's prediction | 2 | 0.816 | 0.829 [0.807, 0.850] | 0.776 [0.746, 0.807] | 0.769 (n = 624) |
| Qwen2.5-1.5B | template × adjacent × common neighbour | 11 | 0.760 | 0.829 [0.807, 0.850] | 0.766 [0.730, 0.798] | 0.698 (n = 55) |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

| Model | Stratified by | Strata | Gold accuracy from the strata alone | Gold-probe AUROC, unstratified | Gold-probe AUROC within strata (pair-weighted) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | hypothesis template | 5 | 0.500 | 0.676 [0.650, 0.704] | 0.691 [0.663, 0.719] |
| Qwen2.5-0.5B | pair adjacent in skeleton | 2 | 0.539 | 0.676 [0.650, 0.704] | 0.671 [0.645, 0.698] |
| Qwen2.5-0.5B | pair correlated | 2 | 0.525 | 0.676 [0.650, 0.704] | 0.665 [0.637, 0.694] |
| Qwen2.5-0.5B | pair has a common neighbour | 2 | 0.553 | 0.676 [0.650, 0.704] | 0.667 [0.637, 0.697] |
| Qwen2.5-0.5B | template × adjacent × common neighbour | 20 | 0.760 | 0.676 [0.650, 0.704] | 0.623 [0.581, 0.663] |
| Qwen2.5-0.5B | hand parse's prediction (2 strata) | 2 | 0.817 | 0.676 [0.650, 0.704] | 0.631 [0.592, 0.670] |
| Qwen2.5-0.5B | quintiles of the hand-parse score | 5 | 0.818 | 0.676 [0.650, 0.704] | 0.586 [0.543, 0.633] |
| Qwen2.5-0.5B | hand-parse quintile × per-string 4-gram quintile | 25 | 0.877 | 0.676 [0.650, 0.704] | 0.500 [0.446, 0.566] |
| Qwen2.5-1.5B | hypothesis template | 5 | 0.500 | 0.829 [0.807, 0.850] | 0.835 [0.813, 0.856] |
| Qwen2.5-1.5B | pair adjacent in skeleton | 2 | 0.539 | 0.829 [0.807, 0.850] | 0.824 [0.801, 0.845] |
| Qwen2.5-1.5B | pair correlated | 2 | 0.525 | 0.829 [0.807, 0.850] | 0.819 [0.794, 0.841] |
| Qwen2.5-1.5B | pair has a common neighbour | 2 | 0.553 | 0.829 [0.807, 0.850] | 0.821 [0.796, 0.843] |
| Qwen2.5-1.5B | template × adjacent × common neighbour | 20 | 0.760 | 0.829 [0.807, 0.850] | 0.766 [0.730, 0.798] |
| Qwen2.5-1.5B | hand parse's prediction (2 strata) | 2 | 0.817 | 0.829 [0.807, 0.850] | 0.775 [0.745, 0.806] |
| Qwen2.5-1.5B | quintiles of the hand-parse score | 5 | 0.818 | 0.829 [0.807, 0.850] | 0.723 [0.683, 0.761] |
| Qwen2.5-1.5B | hand-parse quintile × per-string 4-gram quintile | 25 | 0.877 | 0.829 [0.807, 0.850] | 0.607 [0.550, 0.664] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`</sub>

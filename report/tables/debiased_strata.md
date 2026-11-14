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
| Gemma-2-2B | hypothesis template | 5 | 0.500 | 0.840 [0.820, 0.859] | 0.848 [0.827, 0.867] |
| Gemma-2-2B | pair adjacent in skeleton | 2 | 0.539 | 0.840 [0.820, 0.859] | 0.835 [0.814, 0.856] |
| Gemma-2-2B | pair correlated | 2 | 0.525 | 0.840 [0.820, 0.859] | 0.839 [0.818, 0.860] |
| Gemma-2-2B | pair has a common neighbour | 2 | 0.553 | 0.840 [0.820, 0.859] | 0.838 [0.817, 0.860] |
| Gemma-2-2B | template × adjacent × common neighbour | 20 | 0.760 | 0.840 [0.820, 0.859] | 0.778 [0.747, 0.810] |
| Gemma-2-2B | hand parse's prediction (2 strata) | 2 | 0.817 | 0.840 [0.820, 0.859] | 0.778 [0.747, 0.808] |
| Gemma-2-2B | quintiles of the hand-parse score | 5 | 0.818 | 0.840 [0.820, 0.859] | 0.739 [0.700, 0.776] |
| Gemma-2-2B | hand-parse quintile × per-string 4-gram quintile | 25 | 0.877 | 0.840 [0.820, 0.859] | 0.661 [0.603, 0.715] |
| Qwen2.5-3B | hypothesis template | 5 | 0.500 | 0.845 [0.826, 0.866] | 0.851 [0.831, 0.873] |
| Qwen2.5-3B | pair adjacent in skeleton | 2 | 0.539 | 0.845 [0.826, 0.866] | 0.839 [0.819, 0.859] |
| Qwen2.5-3B | pair correlated | 2 | 0.525 | 0.845 [0.826, 0.866] | 0.843 [0.824, 0.864] |
| Qwen2.5-3B | pair has a common neighbour | 2 | 0.553 | 0.845 [0.826, 0.866] | 0.847 [0.827, 0.869] |
| Qwen2.5-3B | template × adjacent × common neighbour | 20 | 0.760 | 0.845 [0.826, 0.866] | 0.790 [0.759, 0.822] |
| Qwen2.5-3B | hand parse's prediction (2 strata) | 2 | 0.817 | 0.845 [0.826, 0.866] | 0.798 [0.771, 0.826] |
| Qwen2.5-3B | quintiles of the hand-parse score | 5 | 0.818 | 0.845 [0.826, 0.866] | 0.756 [0.720, 0.791] |
| Qwen2.5-3B | hand-parse quintile × per-string 4-gram quintile | 25 | 0.877 | 0.845 [0.826, 0.866] | 0.656 [0.605, 0.708] |
| Qwen2.5-7B (4-bit) | hypothesis template | 5 | 0.500 | 0.863 [0.844, 0.881] | 0.871 [0.850, 0.889] |
| Qwen2.5-7B (4-bit) | pair adjacent in skeleton | 2 | 0.539 | 0.863 [0.844, 0.881] | 0.857 [0.836, 0.877] |
| Qwen2.5-7B (4-bit) | pair correlated | 2 | 0.525 | 0.863 [0.844, 0.881] | 0.857 [0.838, 0.877] |
| Qwen2.5-7B (4-bit) | pair has a common neighbour | 2 | 0.553 | 0.863 [0.844, 0.881] | 0.863 [0.843, 0.882] |
| Qwen2.5-7B (4-bit) | template × adjacent × common neighbour | 20 | 0.760 | 0.863 [0.844, 0.881] | 0.795 [0.760, 0.828] |
| Qwen2.5-7B (4-bit) | hand parse's prediction (2 strata) | 2 | 0.817 | 0.863 [0.844, 0.881] | 0.806 [0.776, 0.834] |
| Qwen2.5-7B (4-bit) | quintiles of the hand-parse score | 5 | 0.818 | 0.863 [0.844, 0.881] | 0.751 [0.712, 0.788] |
| Qwen2.5-7B (4-bit) | hand-parse quintile × per-string 4-gram quintile | 25 | 0.877 | 0.863 [0.844, 0.881] | 0.644 [0.589, 0.694] |

<sub>Source: `results/probe_content_qwen0.5b_c2c_debiased.json`, `results/probe_content_qwen1.5b_c2c_debiased.json`, `results/probe_content_gemma2b_c2c_debiased.json`, `results/probe_content_qwen3b_c2c_debiased.json`, `results/probe_content_qwen7b-4bit_c2c_debiased.json`</sub>

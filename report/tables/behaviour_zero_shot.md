| Model | Dataset | n | clusters | Accuracy | Macro-F1 | AUROC of logit diff | Pred. Yes | Majority acc. | Majority macro-F1 | Random macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Corr2Cause test | 1162 | 21 | 0.845 [0.803, 0.898] | 0.458 [0.445, 0.473] | 0.589 [0.531, 0.660] | 0.00 | 0.845 | 0.458 | 0.433 |
| Qwen2.5-0.5B | CLadder | 3000 | 47 | 0.500 [0.483, 0.515] | 0.366 [0.352, 0.380] | 0.499 [0.468, 0.527] | 0.04 | 0.500 | 0.333 | 0.500 |
| Qwen2.5-1.5B | Corr2Cause test | 1162 | 21 | 0.773 [0.711, 0.846] | 0.489 [0.453, 0.525] | 0.561 [0.503, 0.614] | 0.10 | 0.845 | 0.458 | 0.433 |
| Qwen2.5-1.5B | CLadder | 3000 | 47 | 0.497 [0.474, 0.519] | 0.486 [0.464, 0.509] | 0.509 [0.478, 0.542] | 0.64 | 0.500 | 0.333 | 0.500 |

<sub>Source: `results/behaviour_qwen0.5b.json`, `results/behaviour_qwen1.5b.json`</sub>

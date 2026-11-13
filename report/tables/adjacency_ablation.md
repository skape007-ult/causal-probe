| Model | Site (layer) | Removed | Template | AUROC of answer for gold label: unsteered | Change after removal | Change, range over 20 controls | AUROC of answer for adjacency: unsteered | Change after removal | Change, range over 20 controls | Mean |Δ logit difference| | …largest control |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency probe weights | parent | 0.600 | +0.001 | -0.002 to +0.002 | 0.614 | +0.001 | -0.002 to +0.002 | 0.003 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency probe weights | ancestor | 0.396 | +0.002 | -0.002 to +0.003 | 0.543 | -0.000 | -0.001 to +0.002 | 0.004 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency probe weights | descendant | 0.462 | -0.000 | +0.000 to +0.006 | 0.567 | -0.004 | -0.009 to -0.003 | 0.004 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency probe weights | collider | 0.556 | +0.000 | -0.002 to +0.002 | 0.269 | +0.000 | -0.002 to +0.002 | 0.003 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency probe weights | confounder | 0.423 | -0.001 | -0.002 to +0.002 | 0.490 | +0.002 | -0.000 to +0.003 | 0.004 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency difference of means | parent | 0.600 | +0.017 | -0.049 to +0.071 | 0.614 | +0.043 | -0.048 to +0.085 | 0.241 | 0.338 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency difference of means | ancestor | 0.396 | -0.025 | -0.032 to +0.023 | 0.543 | +0.201 | -0.069 to +0.237 | 0.184 | 0.275 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency difference of means | descendant | 0.462 | -0.055 | -0.065 to +0.082 | 0.567 | +0.090 | -0.132 to +0.089 | 0.169 | 0.259 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency difference of means | collider | 0.556 | +0.031 | -0.049 to +0.040 | 0.269 | +0.194 | -0.029 to +0.242 | 0.203 | 0.313 |
| Qwen2.5-0.5B | final token, best layer (15/24) | adjacency difference of means | confounder | 0.423 | +0.083 | -0.033 to +0.093 | 0.490 | +0.097 | -0.045 to +0.104 | 0.207 | 0.284 |
| Qwen2.5-0.5B | final token, best layer (15/24) | gold-label probe weights | parent | 0.600 | -0.004 | -0.002 to +0.002 | 0.614 | -0.003 | -0.002 to +0.002 | 0.004 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | gold-label probe weights | ancestor | 0.396 | +0.002 | -0.002 to +0.003 | 0.543 | +0.001 | -0.001 to +0.002 | 0.004 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | gold-label probe weights | descendant | 0.462 | +0.005 | +0.000 to +0.006 | 0.567 | -0.005 | -0.009 to -0.003 | 0.003 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | gold-label probe weights | collider | 0.556 | -0.001 | -0.002 to +0.002 | 0.269 | -0.000 | -0.002 to +0.002 | 0.003 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | gold-label probe weights | confounder | 0.423 | -0.002 | -0.002 to +0.002 | 0.490 | +0.003 | -0.000 to +0.003 | 0.004 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | rank-4 adjacency subspace | parent | 0.600 | +0.002 | -0.002 to +0.002 | 0.614 | +0.002 | -0.002 to +0.002 | 0.003 | 0.005 |
| Qwen2.5-0.5B | final token, best layer (15/24) | rank-4 adjacency subspace | ancestor | 0.396 | +0.002 | -0.000 to +0.003 | 0.543 | -0.000 | -0.002 to +0.002 | 0.004 | 0.005 |
| Qwen2.5-0.5B | final token, best layer (15/24) | rank-4 adjacency subspace | descendant | 0.462 | +0.000 | -0.001 to +0.006 | 0.567 | -0.005 | -0.008 to -0.003 | 0.004 | 0.005 |
| Qwen2.5-0.5B | final token, best layer (15/24) | rank-4 adjacency subspace | collider | 0.556 | -0.000 | -0.005 to +0.000 | 0.269 | -0.000 | -0.001 to +0.003 | 0.003 | 0.004 |
| Qwen2.5-0.5B | final token, best layer (15/24) | rank-4 adjacency subspace | confounder | 0.423 | -0.001 | -0.002 to +0.002 | 0.490 | -0.000 | -0.001 to +0.003 | 0.004 | 0.005 |

<sub>Source: `results/feature_causal_qwen0.5b_c2c_debiased_adj_final_best.json`</sub>

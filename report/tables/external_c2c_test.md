| Model | n | Gold probe AUROC | TF-IDF AUROC | Type prior AUROC | Model logit-diff AUROC | Probe − TF-IDF | Structural AUROC (post-hoc) | Probe − structural (post-hoc) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 1162 | 0.887 [0.849, 0.925] | 0.847 [0.815, 0.873] | 0.762 [0.729, 0.801] | 0.589 [0.531, 0.660] | +0.040 [+0.001, +0.091] | 0.922 [0.892, 0.949] | -0.035 [-0.078, +0.010] |
| Qwen2.5-1.5B | 1162 | 0.919 [0.898, 0.939] | 0.847 [0.815, 0.873] | 0.762 [0.729, 0.801] | 0.561 [0.503, 0.614] | +0.072 [+0.050, +0.097] | 0.922 [0.892, 0.949] | -0.003 [-0.037, +0.032] |

<sub>Source: `results/confound_qwen0.5b_c2c_probe.json`, `results/confound_qwen1.5b_c2c_probe.json`</sub>

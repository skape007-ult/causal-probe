"""Reference for the owner's part (b). Same definition as v1.0 (scripts/15_answer_bias.py, `residual("quadratic")`
followed by the mean of the within-string AUROCs), written as one function."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def within_string_auroc_length_removed(labels, scores, strings, n_tokens) -> dict:
    y = np.asarray(labels).astype(int); s = np.asarray(scores, float)
    strings = np.asarray(strings); n = np.asarray(n_tokens, float)
    per = {}
    for u in np.unique(strings):
        c = np.where(strings == u)[0]
        n1 = int(y[c].sum()); n0 = len(c) - n1
        if n1 == 0 or n0 == 0:
            continue
        lx = np.log(n[c]); lx = lx - lx.mean()
        X = np.column_stack([np.ones(len(c)), lx, lx ** 2])
        r = s[c] - X @ np.linalg.lstsq(X, s[c], rcond=None)[0]      # labels are not used in the fit
        rk = rankdata(r)                                            # ties get the average rank
        per[str(u)] = float((rk[y[c] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {"mean_auroc": float(np.mean(list(per.values()))) if per else float("nan"), "per_string": per, "n_strings": len(per)}

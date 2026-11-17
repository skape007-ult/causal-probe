"""The one registered test of the fine-tuning study (PREREG.md version 2, Section 3).

A statistic (for example S on POOL-LB) is computed once per trained model. Models of one condition differ by seed.
The test pools the seeds into one number and carries both sources of uncertainty:

    D   = mean_seeds(X) - mean_seeds(Y)            (or mean_seeds(X) - constant; or the mean of within-seed differences)
    SE² = s_X²/n_X + s_Y²/n_Y + v_premise
    t   = D / SE,  Student's t with the Welch-Satterthwaite degrees of freedom of the seed terms, two-sided.

v_premise is the variance of D over premise resamples with the models held fixed (every model rescored on the same
resample). Size and power of exactly this function: finetune/scripts/power.py.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

N_BOOT = 2000


def t_and_df(D, var_x, n_x, var_y, n_y, v_premise):
    """Vectorised. var_y None / n_y 0: one-sample (or paired differences passed as x)."""
    a = np.asarray(var_x, float) / n_x
    b = np.asarray(var_y, float) / n_y if n_y else 0.0
    seed = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        den = a ** 2 / (n_x - 1) + (b ** 2 / (n_y - 1) if n_y else 0.0)
        df = np.where(den > 0, seed ** 2 / den, n_x - 1 + (n_y - 1 if n_y else 0))
    return np.asarray(D, float) / np.sqrt(seed + np.asarray(v_premise, float)), df


def p_two_sided(t, df):
    return 2 * stats.t.sf(np.abs(t), df)


def test(per_seed_x, per_seed_y=None, constant: float = 0.0, boot_D=None) -> dict:
    """per_seed_x, per_seed_y: the statistic of each trained model. boot_D: the values of D on premise resamples
    (models fixed), from which v_premise is taken; required."""
    x = np.asarray(per_seed_x, float)
    yv = None if per_seed_y is None else np.asarray(per_seed_y, float)
    if boot_D is None or len(boot_D) < 200:
        raise ValueError("the premise resamples of D are part of the test")
    D = x.mean() - (yv.mean() if yv is not None else constant)
    v_prem = float(np.var(np.asarray(boot_D, float), ddof=1))
    t, df = t_and_df(D, x.var(ddof=1), len(x), None if yv is None else yv.var(ddof=1), 0 if yv is None else len(yv), v_prem)
    t, df = float(t), float(df)
    se = abs(D / t) if t != 0 else float(np.sqrt(x.var(ddof=1) / len(x) + v_prem))
    half = float(stats.t.ppf(0.975, df)) * se
    return {"D": float(D), "se": float(se), "t": t, "df": df, "p_two_sided": float(p_two_sided(t, df)), "lo": float(D - half), "hi": float(D + half),
            "seed_sd_x": float(x.std(ddof=1)), "seed_sd_y": None if yv is None else float(yv.std(ddof=1)), "premise_se": float(np.sqrt(v_prem)),
            "n_x": len(x), "n_y": 0 if yv is None else len(yv), "per_seed_x": x.tolist(), "per_seed_y": None if yv is None else yv.tolist(),
            "claimed": bool(p_two_sided(t, df) < 0.05)}

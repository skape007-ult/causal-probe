"""Metrics and (cluster) bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .config import N_BOOT, SEED


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=[0, 1], zero_division=0))


def pos_f1(y, p):
    return float(f1_score(y, p, pos_label=1, zero_division=0))


def auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def bootstrap_ci(fn, arrays: tuple, groups=None, n_boot: int = N_BOOT, seed: int = SEED,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for fn(*arrays).

    If `groups` is given, whole groups (premises / stories) are resampled with replacement
    (cluster bootstrap), because items that share a premise or story are not independent.
    Returns {"est", "lo", "hi", "n", "n_groups"}.
    """
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    est = fn(*arrays)
    if n == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_groups": 0}
    if groups is None:
        draws = (rng.integers(0, n, n) for _ in range(n_boot))
        ng = n
    else:
        groups = np.asarray(groups)
        ug, inv = np.unique(groups, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(ug))]
        ng = len(ug)
        draws = (np.concatenate([members[k] for k in rng.integers(0, ng, ng)]) for _ in range(n_boot))
    vals = np.array([fn(*(a[idx] for a in arrays)) for idx in draws], float)
    vals = vals[~np.isnan(vals)]
    lo, hi = (np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (np.nan, np.nan))
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n": int(n), "n_groups": int(ng)}


def baselines(y, n_sim: int = 2000, seed: int = SEED) -> dict:
    """Majority-class and uniform-random baselines on the same items."""
    y = np.asarray(y)
    maj = int(np.mean(y) >= 0.5)
    pm = np.full_like(y, maj)
    rng = np.random.default_rng(seed)
    sims = [(accuracy(y, r), macro_f1(y, r), pos_f1(y, r))
            for r in (rng.integers(0, 2, len(y)) for _ in range(n_sim))]
    ra, rf, rp = np.mean(sims, 0)
    return {"majority": {"label": maj, "accuracy": accuracy(y, pm), "macro_f1": macro_f1(y, pm),
                         "pos_f1": pos_f1(y, pm)},
            "random": {"accuracy": float(ra), "macro_f1": float(rf), "pos_f1": float(rp)}}

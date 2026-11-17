"""The statistics of PREREG.md (version 2, Section 3) as the analysis calls them. Written before any model score exists.

  S(...)            the registered statistic: strings with at least 3 valid and 3 invalid items, then the owner's
                    part (b) (or its reference) on those items
  S_cells(...)      E5a: the same length-removed score inside cells of string x adjacency, pair-weighted
  boot_indices(...) the premise resamples shared by every model of a test
  boot_D(...)       D on each resample, models fixed; eligibility of strings and the length residual are recomputed
                    on every resample (a resample is a new data set)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from . import impl

MIN_CLASS = 3
N_BOOT = 2000


def eligible(labels, strings, min_class: int = MIN_CLASS) -> np.ndarray:
    y, q = np.asarray(labels).astype(int), np.asarray(strings)
    keep = np.zeros(len(y), bool)
    for u in np.unique(q):
        m = q == u
        if y[m].sum() >= min_class and (1 - y[m]).sum() >= min_class:
            keep |= m
    return keep


def S(labels, scores, strings, n_tokens, min_class: int = MIN_CLASS) -> float:
    y, s, q, n = (np.asarray(v) for v in (labels, scores, strings, n_tokens))
    k = eligible(y, q, min_class)
    if not k.any():
        return float("nan")
    fn, _ = impl.within_string_auroc_length_removed()
    return float(fn(y[k], s[k].astype(float), q[k], n[k])["mean_auroc"])


def length_residual(scores, strings, n_tokens) -> np.ndarray:
    """Score minus its least-squares fit on centred log token count and its square, inside each string; no labels."""
    s, q, n = np.asarray(scores, float), np.asarray(strings), np.asarray(n_tokens, float)
    out = np.empty(len(s))
    for u in np.unique(q):
        c = np.where(q == u)[0]
        lx = np.log(n[c]); lx = lx - lx.mean()
        X = np.column_stack([np.ones(len(c)), lx, lx ** 2])
        out[c] = s[c] - X @ np.linalg.lstsq(X, s[c], rcond=None)[0]
    return out


def S_cells(labels, scores, strings, adjacency, n_tokens, min_class: int = MIN_CLASS) -> float:
    """E5a. Residualise within string, then AUROC inside each (string, adjacency) cell with at least min_class items
    per class; mean over cells weighted by valid x invalid pairs (as v1.0 scripts/15_answer_bias.py)."""
    y, q, a = np.asarray(labels).astype(int), np.asarray(strings), np.asarray(adjacency)
    r = length_residual(scores, q, n_tokens)
    key = np.array([f"{u}|{v}" for u, v in zip(q, a)])
    au, w = [], []
    for u in np.unique(key):
        c = np.where(key == u)[0]
        n1 = int(y[c].sum()); n0 = len(c) - n1
        if min(n1, n0) < min_class:
            continue
        rk = rankdata(r[c])
        au.append((rk[y[c] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)); w.append(n1 * n0)
    return float(np.average(au, weights=w)) if au else float("nan")


def boot_indices(groups, n_boot: int = N_BOOT, seed: int = 0) -> list[np.ndarray]:
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]


def boot_D(stat, x_scores: list, y_scores: list | None, indices: list[np.ndarray], constant: float = 0.0) -> np.ndarray:
    """stat(idx, score_vector) -> the statistic on the resample idx. x_scores, y_scores: one score vector per trained
    model (y_scores may hold one fixed reference score; then D = mean_x - stat(reference)). With y_scores None,
    D = mean_x - constant."""
    out = np.empty(len(indices))
    for b, idx in enumerate(indices):
        mx = np.mean([stat(idx, s) for s in x_scores])
        out[b] = mx - (np.mean([stat(idx, s) for s in y_scores]) if y_scores is not None else constant)
    return out

"""Linear probes on cached activations, with controls. CPU only.

All probes are L2-regularised logistic regressions on standardised features, evaluated
out-of-fold with grouped folds (no premise / story on both sides of a split).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import data, prompts
from .config import N_FOLDS, SEED

PROBE_C = 0.01  # fixed a priori (decision log D7); sensitivity reported, never tuned on test


def fit_predict(Xtr, ytr, Xte, C: float = PROBE_C):
    """Standardise on train, fit logistic regression, return test decision scores."""
    sc = StandardScaler().fit(Xtr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def oof_scores(X: np.ndarray, y: np.ndarray, folds, C: float = PROBE_C,
               y_train: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold decision scores. `y_train` (default y) lets controls train on other labels."""
    yt = y if y_train is None else y_train
    s = np.full(len(y), np.nan, np.float32)
    for tr, te in folds:
        if len(np.unique(yt[tr])) < 2:
            s[te] = 0.0
            continue
        s[te] = fit_predict(X[tr], yt[tr], X[te], C)
    return s


def layer_features(resid, layer: int, pos: int) -> np.ndarray:
    return np.asarray(resid[:, layer, pos, :], dtype=np.float32)


def control_task_labels(items: list[dict], seed: int = SEED) -> np.ndarray:
    """Hewitt & Liang (2019) control task, adapted from word types to item 'types'.

    Each surface type gets a fixed random label. Corr2Cause: type = the hypothesis string
    (e.g. "A directly causes B."), which recurs across premises. CLadder: type =
    (graph_id, query_type), which recurs across stories. A probe can only do well on these
    labels by memorising type -> label, so control accuracy measures the probe's capacity
    to memorise; selectivity = real-task accuracy - control-task accuracy.
    """
    rng = np.random.default_rng(seed + 1000)
    def key(it):
        return it["query"] if it["dataset"] == "c2c" else (it["graph_id"], it["query_type"])
    keys = sorted({key(it) for it in items}, key=str)
    lab = dict(zip(map(str, keys), rng.permutation(np.arange(len(keys)) % 2)))
    return np.array([lab[str(key(it))] for it in items])


def shuffled_labels(y: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 2000).permutation(y)


def tfidf_oof(items: list[dict], y: np.ndarray, folds, y_train=None) -> np.ndarray:
    """Bag-of-words baseline: TF-IDF (1-2 grams, case-sensitive so variable names survive)
    on the user text, logistic regression, same folds. Vectoriser fitted on train folds only."""
    yt = y if y_train is None else y_train
    texts = np.array([prompts.user_text(it) for it in items], dtype=object)
    s = np.full(len(y), np.nan, np.float32)
    for tr, te in folds:
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\b\w+\b",
                              min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(texts[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, yt[tr])
        s[te] = clf.decision_function(vec.transform(texts[te]))
    return s


def query_only_oof(items: list[dict], y: np.ndarray, folds) -> np.ndarray:
    """Stronger shortcut baseline: one-hot of the hypothesis / question *type* only
    (Corr2Cause template; CLadder graph_id x query_type). Captures label priors per type."""
    def key(it):
        return it["template"] if it["dataset"] == "c2c" else f"{it['graph_id']}|{it['query_type']}"
    keys = np.array([key(it) for it in items])
    s = np.zeros(len(y), np.float32)
    for tr, te in folds:
        prior = {k: (y[tr][keys[tr] == k].mean() if (keys[tr] == k).any() else y[tr].mean())
                 for k in np.unique(keys)}
        s[te] = [prior[k] - 0.5 for k in keys[te]]
    return s


def folds_for(items, seed: int):
    return data.grouped_folds(items, N_FOLDS, seed)

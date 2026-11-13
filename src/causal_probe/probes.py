"""Linear probes on cached activations, with controls. CPU only.

All probes are L2-regularised logistic regressions on standardised features, evaluated
out-of-fold with grouped folds (no premise / story on both sides of a split).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from . import data, prompts
from .config import N_FOLDS, SEED

PROBE_C = 0.01               # the original fixed value (decision D7); kept as a sensitivity run
PROBE_CS = (0.01, 0.1, 1.0)  # nested-CV grid (decision D17)
TFIDF_CS = (0.1, 1.0, 10.0, 100.0)   # 100 added after the Tier 1 audit (optimum was at the grid edge)
INNER_FOLDS = 3


def _inner_splits(y, groups, seed=0):
    """Grouped, stratified inner folds on the training portion only."""
    from sklearn.model_selection import StratifiedGroupKFold
    sg = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return list(sg.split(np.zeros(len(y)), y, groups))


def _fit(Xtr, ytr, C, groups_tr, Cs, max_iter):
    """Logistic regression. C=None -> choose C by inner grouped CV on the training data
    (never sees the outer test fold), then refit on all training data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if C is not None:
            return LogisticRegression(C=C, max_iter=max_iter).fit(Xtr, ytr)
        try:
            cv = _inner_splits(ytr, groups_tr)
            ok = all(len(np.unique(ytr[tr])) == 2 for tr, _ in cv)
        except ValueError:
            ok = False
        if not ok:
            return LogisticRegression(C=Cs[len(Cs) // 2], max_iter=max_iter).fit(Xtr, ytr)
        return LogisticRegressionCV(Cs=list(Cs), cv=cv, max_iter=max_iter, scoring="accuracy").fit(Xtr, ytr)


def fit_predict(Xtr, ytr, Xte, C: float | None = PROBE_C, groups_tr=None):
    """Standardise on train, fit logistic regression, return test decision scores."""
    sc = StandardScaler().fit(Xtr)
    clf = _fit(sc.transform(Xtr), ytr, C, groups_tr, PROBE_CS, 300)
    return clf.decision_function(sc.transform(Xte))


def oof_scores(X: np.ndarray, y: np.ndarray, folds, C: float | None = PROBE_C,
               y_train: np.ndarray | None = None, groups: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold decision scores. `y_train` (default y) lets controls train on other labels.
    C=None selects C by nested grouped CV inside each training fold (needs `groups`)."""
    yt = y if y_train is None else y_train
    s = np.full(len(y), np.nan, np.float32)
    for tr, te in folds:
        if len(np.unique(yt[tr])) < 2:
            s[te] = 0.0
            continue
        s[te] = fit_predict(X[tr], yt[tr], X[te], C, None if groups is None else groups[tr])
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


def tfidf_oof(items: list[dict], y: np.ndarray, folds, y_train=None, C: float | None = 1.0,
              groups: np.ndarray | None = None) -> np.ndarray:
    """Bag-of-words baseline: TF-IDF (1-2 grams, case-sensitive so variable names survive)
    on the user text, logistic regression, same folds. Vectoriser fitted on train folds only.
    C=None -> same nested selection as the probes (so the baseline is not handicapped)."""
    yt = y if y_train is None else y_train
    texts = np.array([prompts.user_text(it) for it in items], dtype=object)
    s = np.full(len(y), np.nan, np.float32)
    for tr, te in folds:
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\b\w+\b",
                              min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(texts[tr])
        clf = _fit(Xtr, yt[tr], C, None if groups is None else groups[tr], TFIDF_CS, 1000)
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


def _hyp_key(it):
    return it["query"] if it["dataset"] == "c2c" else f"{it['graph_id']}|{it['query_type']}"


def hyp_prior_oof(items: list[dict], y: np.ndarray, folds) -> np.ndarray:
    """Label prior given the exact hypothesis string (Corr2Cause), estimated on training folds.
    Uses no premise and no activations. Added after the Tier 1 audit, which found that the
    hypothesis string alone predicts the label with ~0.86 accuracy in the balanced pool
    (variables are canonically ordered, so e.g. 'A directly causes D.' is usually valid).
    For CLadder the key is graph|query_type (questions do not recur across stories)."""
    keys = np.array([_hyp_key(it) for it in items])
    s = np.zeros(len(y), np.float32)
    for tr, te in folds:
        prior = {k: y[tr][keys[tr] == k].mean() for k in np.unique(keys[tr])}
        s[te] = [prior.get(k, y[tr].mean()) - 0.5 for k in keys[te]]
    return s


def tfidf_query_oof(items: list[dict], y: np.ndarray, folds, C: float = 10.0) -> np.ndarray:
    """TF-IDF on the hypothesis / question text only (1-3 grams). The long premise dilutes the
    hypothesis tokens in a full-text TF-IDF vector; this variant does not have that handicap."""
    texts = np.array([it["query"] for it in items], dtype=object)
    s = np.zeros(len(y), np.float32)
    for tr, te in folds:
        vec = TfidfVectorizer(ngram_range=(1, 3), lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=1, sublinear_tf=True)
        clf = LogisticRegression(C=C, max_iter=1000).fit(vec.fit_transform(texts[tr]), y[tr])
        s[te] = clf.decision_function(vec.transform(texts[te]))
    return s


def folds_for(items, seed: int):
    return data.grouped_folds(items, N_FOLDS, seed)


# ----------------------------------------------------------------------------- structural text baseline
import re as _re

_ST_CORR = _re.compile(r"\b([A-Z]) correlates with ([A-Z])\.")
_ST_IND = _re.compile(r"\b([A-Z]) is independent of ([A-Z])\.")
_ST_CIND = _re.compile(r"\b([A-Z]) and ([A-Z]) are independent given ([^.]*)\.")
_VAR = _re.compile(r"\b([A-Z])\b")
C2C_TEMPLATES = ["parent", "child", "non-parent ancestor", "non-child descendant", "has_collider", "has_confounder"]


def parse_premise(context: str) -> dict:
    """Parse a Corr2Cause premise into its statements: marginal correlations, marginal
    independences and conditional independences {pair: [conditioning sets]}."""
    corr = {frozenset(m) for m in _ST_CORR.findall(context)}
    ind = {frozenset(m) for m in _ST_IND.findall(context)}
    cind = {}
    for a, b, rest in _ST_CIND.findall(context):
        cind.setdefault(frozenset((a, b)), []).append(frozenset(_VAR.findall(rest)))
    return {"corr": corr, "ind": ind, "cind": cind}


_PCT = _re.compile(r"(\d+)%")
_CORRNUM = _re.compile(r"correlation between .*? is (-?\d+\.\d+)")
CL_QTYPES = ["ate", "backadj", "collider_bias", "correlation", "det-counterfactual", "ett", "exp_away", "marginal", "nde", "nie"]
CL_GRAPHS = ["IV", "arrowhead", "chain", "collision", "confounding", "diamond", "diamondcut", "fork", "frontdoor", "mediation"]
_POS_WORDS = ("increase", "more likely", "larger", "positively")
_NEG_WORDS = ("decrease", "less likely", "smaller", "negatively")


def cladder_features(items: list[dict]) -> np.ndarray:
    """Stronger text-only baseline for CLadder: the numbers quoted in the prompt, simple
    differences between them, the polarity of the question's wording, and the query / graph
    type. A gradient-boosted model on these can learn heuristics such as "the second
    probability is larger and the question says 'increase' -> yes" without any causal
    inference (no adjustment formulas are evaluated)."""
    rows = []
    for it in items:
        pct = [float(x) for x in _PCT.findall(it["context"])]
        q = it["query"].lower()
        cm = _CORRNUM.search(it["context"])
        ctx = it["context"]
        f = (pct + [-1.0] * 9)[:9]
        f += [float(len(pct)),
              pct[-1] - pct[-2] if len(pct) >= 2 else 0.0,
              pct[1] - pct[0] if len(pct) >= 2 else 0.0,
              pct[-1] - 50.0 if pct else 0.0,
              float(cm.group(1)) if cm else 0.0, float(cm is not None),
              float(any(w in q for w in _POS_WORDS)), float(any(w in q for w in _NEG_WORDS)),
              float(q.count(" not ")), float("instead" in q), float("overall" in q),
              # backdoor-adjustment questions: which method is the 'case by case' (adjusted) one?
              float(0 <= ctx.find("case by case") < ctx.find("Method 2")) if "Method 2" in ctx else -1.0,
              float("method 1 than method 2" in q), float("method 2 than method 1" in q)]
        f += [float(it["query_type"] == t) for t in CL_QTYPES] + [float(it["graph_id"] == g) for g in CL_GRAPHS]
        rows.append(f)
    return np.array(rows, np.float32)


def structural_features(items: list[dict]) -> np.ndarray:
    """A stronger text-only baseline than bag-of-words (dispatches to `cladder_features` for
    CLadder items). For Corr2Cause: features a shallow parser
    could compute by binding the premise's statements to the two variables named in the
    hypothesis. No causal inference is performed (no orientation rules, no equivalence classes);
    'adjacent' = correlated and never rendered independent by any conditioning set, i.e. the
    skeleton edge, which is step 1 of the PC algorithm.
    """
    if items[0]["dataset"] == "cladder":
        return cladder_features(items)
    rows = []
    for it in items:
        p = parse_premise(it["context"])
        allv = sorted(set(_VAR.findall(it["context"].split("variables,")[1].split(".")[0]))) if "variables," in it["context"] else []
        hv = _VAR.findall(it["query"])
        a, b = (hv + ["?", "?"])[:2]
        pair = frozenset((a, b))

        def adj(x, y):
            q = frozenset((x, y))
            return q in p["corr"] and q not in p["cind"]

        sets = p["cind"].get(pair, [])
        nb_a = {v for v in allv if v != a and adj(a, v)}
        nb_b = {v for v in allv if v != b and adj(b, v)}
        common = nb_a & nb_b
        # unshielded triples a - c - b (a, b not adjacent): is c in some separating set of (a, b)?
        c_in_sep = [any(c in s for s in sets) for c in common] if not adj(a, b) else []
        f = [
            float(pair in p["corr"]), float(pair in p["ind"]), float(adj(a, b)),
            float(len(sets)), float(min((len(s) for s in sets), default=-1)), float(max((len(s) for s in sets), default=-1)),
            float(len(nb_a)), float(len(nb_b)), float(len(common)),
            float(sum(c_in_sep)), float(len(c_in_sep) - sum(c_in_sep)),          # non-colliders / colliders at a-c-b
            float(len(allv)), float(len(p["corr"])), float(len(p["ind"])), float(sum(len(v) for v in p["cind"].values())),
            float(a < b),
        ]
        f += [float(it.get("template") == t) for t in C2C_TEMPLATES]
        rows.append(f)
    return np.array(rows, np.float32)


def _oof_prior(items_tr, y_tr, n_inner: int = 5):
    """Hypothesis-string prior as a feature for the TRAINING rows, computed out-of-fold over
    inner premise-grouped folds, so a row's own label (and its premise-mates') never enters its
    own feature. (A leave-one-out encoding leaks the complement of the label in balanced cells:
    the model learns 'prior slightly below average -> positive' and collapses on test data.)"""
    from sklearn.model_selection import GroupKFold
    keys = np.array([_hyp_key(i) for i in items_tr]); g = np.array([i["group"] for i in items_tr])
    out = np.full(len(y_tr), float(np.mean(y_tr)), np.float32)
    for a, b in GroupKFold(n_splits=min(n_inner, len(set(g)))).split(keys, y_tr, g):
        prior = {k: y_tr[a][keys[a] == k].mean() for k in np.unique(keys[a])}
        out[b] = [prior.get(k, y_tr[a].mean()) for k in keys[b]]
    return out


def _train_prior(items_tr, y_tr, items_te):
    ktr = np.array([_hyp_key(i) for i in items_tr])
    prior = {k: y_tr[ktr == k].mean() for k in np.unique(ktr)}
    return np.array([prior.get(_hyp_key(i), y_tr.mean()) for i in items_te], np.float32)


def structural_fit_predict(items_tr, y_tr, items_te) -> np.ndarray:
    """Gradient boosting on structural_features; for Corr2Cause also the hypothesis-string prior
    (the strongest text-only baseline we use). Returns logit-like scores for the test items.
    CLadder gets no prior feature: its query/graph types are already one-hot features and are
    label-balanced by construction."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, Xte = structural_features(items_tr), structural_features(items_te)
    if items_tr[0]["dataset"] == "c2c":
        Xtr = np.column_stack([Xtr, _oof_prior(items_tr, y_tr)])
        Xte = np.column_stack([Xte, _train_prior(items_tr, y_tr, items_te)])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, y_tr)
    pr = np.clip(clf.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(pr / (1 - pr)).astype(np.float32)


def structural_oof(items: list[dict], y: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold scores of `structural_fit_predict` (non-linear, so template x structure
    interactions are available to it)."""
    s = np.zeros(len(y), np.float32)
    arr = np.array(items, dtype=object)
    for tr, te in folds:
        s[te] = structural_fit_predict(list(arr[tr]), y[tr], list(arr[te]))
    return s

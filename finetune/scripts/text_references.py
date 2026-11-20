"""E5c's text references (PREREG version 2, Section 4): fitted on a condition's TRAINING set only, scored on POOL-LB
(and ID48), before any fine-tuned model's scores are opened. Needs no model; the token counts for S come from the
base model's scores file (identical for every model: same prompts, same tokenizer).

  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/text_references.py
    -> finetune/itemsets/text_references.json (S of every reference) and text_references.npz (their score vectors)

References, as registered:
  lookup            v1.0's "template x statements naming the pair": one gradient-boosting model per template on the
                    six pair features of probes.structural_features (columns 0-5)
  hand_parse        v1.0's hand parse: one gradient-boosting model on all structural features
  tuned_stack       hand parse stacked with a per-hypothesis-string TF-IDF (1-4 grams) logistic model of the premise
                    text; the n-gram C (1, 10, 100) is chosen by premise-grouped 5-fold cross-validation inside the
                    training set (accuracy of the stacked out-of-fold score), never on an evaluation set
  premise_counts_logistic / _boosting   the per-string premise-count classifier (datasets.premise_counts_within_string)
The "best text reference" of a condition is the largest S on POOL-LB among these five, chosen once on the full set.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]
import numpy as np                                                   # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier          # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer          # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from causal_probe import data, probes                                # noqa: E402
from ftstudy import datasets as DS, stat                             # noqa: E402

OUT = ROOT / "finetune" / "itemsets"
NG_CS = (1.0, 10.0, 100.0)
CONDITIONS = ("A", "B", "A48")
EVAL = ("pool_lb", "id48")


def logit(pr):
    pr = np.clip(pr, 1e-4, 1 - 1e-4)
    return np.log(pr / (1 - pr))


def by_key_gbm(keys_tr, X_tr, y_tr, keys_te, X_te):
    s = np.zeros(len(X_te))
    for k in np.unique(keys_te):
        a, b = np.where(keys_tr == k)[0], np.where(keys_te == k)[0]
        if len(a) and len(np.unique(y_tr[a])) > 1:
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0).fit(X_tr[a], y_tr[a])
            s[b] = logit(clf.predict_proba(X_te[b])[:, 1])
    return s


def gbm_all(X_tr, y_tr, X_te):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X_tr, y_tr)
    return logit(clf.predict_proba(X_te)[:, 1])


def by_key_bow(keys_tr, txt_tr, y_tr, keys_te, txt_te, C):
    s = np.zeros(len(txt_te))
    for k in np.unique(keys_te):
        a, b = np.where(keys_tr == k)[0], np.where(keys_te == k)[0]
        if len(a) and len(np.unique(y_tr[a])) > 1:
            vec = TfidfVectorizer(ngram_range=(1, 4), lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=1, sublinear_tf=True)
            clf = LogisticRegression(C=C, max_iter=1000).fit(vec.fit_transform(txt_tr[a]), y_tr[a])
            s[b] = clf.decision_function(vec.transform(txt_te[b]))
    return s


class Side:
    def __init__(self, items):
        self.items = items
        self.y = np.array([i["label"] for i in items])
        self.F = probes.structural_features(items)
        self.tmpl = np.array([i["template"] for i in items])
        self.q = np.array([i["query"] for i in items])
        self.txt = np.array([i["context"] for i in items], dtype=object)

    def sub(self, idx):
        s = Side.__new__(Side)
        s.items = [self.items[i] for i in idx]
        s.y, s.F, s.tmpl, s.q, s.txt = self.y[idx], self.F[idx], self.tmpl[idx], self.q[idx], self.txt[idx]
        return s


def components(tr: Side, te: Side, Cs=NG_CS) -> dict:
    out = {"lookup": by_key_gbm(tr.tmpl, tr.F[:, 0:6], tr.y, te.tmpl, te.F[:, 0:6]),
           "hand_parse": gbm_all(tr.F, tr.y, te.F)}
    for C in Cs:
        out[f"ngram4_C{C:g}"] = by_key_bow(tr.q, tr.txt, tr.y, te.q, te.txt, C)
    return out


def oof_components(tr: Side, k: int = 5, seed: int = 0) -> dict:
    """Out-of-fold component scores inside the training set (premise-grouped folds)."""
    folds = data.grouped_folds(tr.items, k, seed=seed)
    out = None
    for a, b in folds:
        c = components(tr.sub(np.asarray(a)), tr.sub(np.asarray(b)))
        if out is None:
            out = {n: np.zeros(len(tr.y)) for n in c}
        for n, v in c.items():
            out[n][np.asarray(b)] = v
    return out, folds


def fit_stacker(Z, y):
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    clf = LogisticRegression(C=1.0).fit((Z - mu) / sd, y)
    return lambda Zt: clf.decision_function((Zt - mu) / sd)


def tuned_stack(tr: Side, te: Side, comp_te: dict) -> tuple[np.ndarray, float, dict]:
    oof, folds = oof_components(tr)
    acc = {}
    for C in NG_CS:
        Z = np.column_stack([oof["hand_parse"], oof[f"ngram4_C{C:g}"]])
        hit = np.zeros(len(tr.y), bool)
        for a, b in folds:                         # the stacker's own accuracy, also out of fold
            a, b = np.asarray(a), np.asarray(b)
            hit[b] = (fit_stacker(Z[a], tr.y[a])(Z[b]) > 0) == tr.y[b]
        acc[C] = float(hit.mean())
    C = max(NG_CS, key=lambda c: (acc[c], -c))
    st = fit_stacker(np.column_stack([oof["hand_parse"], oof[f"ngram4_C{C:g}"]]), tr.y)
    return st(np.column_stack([comp_te["hand_parse"], comp_te[f"ngram4_C{C:g}"]])), C, acc


def main():
    ev = {k: DS.materialise("eval_" + k) for k in EVAL}
    base = np.load(ROOT / "finetune" / "runs" / "base" / "scores.npz")
    meta = {}
    for k, items in ev.items():
        uid = [str(u) for u in base[f"eval_{k}__uid"]]
        assert uid == [i["uid"] for i in items], f"base scores of eval_{k} are not in the stored order"
        meta[k] = (np.array([i["label"] for i in items]), np.array([i["query"] for i in items]), base[f"eval_{k}__n_tokens"])
    sides = {k: Side(v) for k, v in ev.items()}
    res, vecs = {"statistic": "S as registered, on each evaluation set", "conditions": {}}, {}
    for cond in CONDITIONS:
        tr = Side(DS.materialise("train_" + cond))
        r = {}
        for k in EVAL:
            te = sides[k]
            comp = components(tr, te)
            sc = {"lookup": comp["lookup"], "hand_parse": comp["hand_parse"]}
            sc["tuned_stack"], C, acc = tuned_stack(tr, te, comp)
            sc["premise_counts_logistic"] = DS.premise_counts_within_string(tr.items, te.items)
            sc["premise_counts_boosting"] = DS.premise_counts_within_string(tr.items, te.items, boosting=True)
            y, q, n = meta[k]
            Sv = {name: stat.S(y, v, q, n) for name, v in sc.items()}
            best = max(Sv, key=Sv.get)
            r[k] = {"S": Sv, "best": {"name": best, "S": Sv[best]}, "tuned_stack_C": C, "tuned_stack_cv_accuracy": {f"{c:g}": a for c, a in acc.items()}}
            for name, v in sc.items():
                vecs[f"{cond}|{k}|{name}"] = v
            print(cond, k, {a: round(b, 3) for a, b in Sv.items()}, "best", best, "C", C, flush=True)
        res["conditions"][cond] = r
    (OUT / "text_references.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(OUT / "text_references.npz", **vecs)


if __name__ == "__main__":
    main()

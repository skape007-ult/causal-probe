"""How well do text-only baselines do on Corr2Cause and CLadder? (CPU only, no model, no cache.)

One script reproduces every number in the "text baselines on the benchmarks" section:

  python scripts/download_data.py            # pinned dataset revisions, checksums verified
  python scripts/14_benchmark_baselines.py   # -> results/benchmark_baselines.json  (about 10 minutes)

Data versions (see scripts/download_data.py for revisions and md5 sums):
  Corr2Cause  huggingface.co/datasets/causal-nlp/corr2cause @ 42ba12c7   train.csv / dev.csv / test.csv
  CLadder     huggingface.co/datasets/causal-nlp/CLadder    @ 18d684e4   data/full_v1.5_default.csv

What is measured. A "text-only baseline" is a classifier that sees (part of) the prompt text and no
language model. None of them applies a causal-inference rule.

Corr2Cause, official splits (fit on train, evaluate on dev and test):
  majority            always the more frequent label
  template_prior      P(valid | hypothesis template), 6 templates
  hypothesis_only     P(valid | exact hypothesis string), 120 strings. Every hypothesis-only classifier
                      is a function of that string, so this is the best any of them can do on train.
  premise_only        TF-IDF (1-2 grams) logistic regression on the premise text alone
  parsed_features     gradient boosting on shallow parsed features of premise + hypothesis (are the two
                      named variables correlated / adjacent in the skeleton, how many separating sets,
                      shared neighbours, ...) plus the hypothesis-string prior
Corr2Cause, label-balanced pools (premise-grouped 5-fold cross-validation, 3 seeds):
  the same baselines on (a) the balanced probe pool, (b) the debiased pool, in which every
  hypothesis string and every premise is exactly half valid in every fold, and (c) a first,
  rejected debiasing design that balanced strings over the whole pool only (kept to document why).
CLadder full v1.5 (story-grouped 5-fold cross-validation, 3 seeds):
  type_prior          P(yes | graph type x query type)
  question_only       TF-IDF (1-3 grams) logistic regression on the final question alone
  full_text           TF-IDF (1-2 grams) logistic regression on the whole prompt
  numeric_polarity    gradient boosting on the quoted probabilities, differences between them, polarity
                      words of the question, query type and graph type

Confidence intervals: percentile bootstrap, resampling whole premises (Corr2Cause) or stories (CLadder).
"""
import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from causal_probe import data, probes, stats
from causal_probe.config import PROBE_SEEDS, RESULTS, SEED

N_TRAIN_FIT = 20000       # train items used to fit the two learned Corr2Cause baselines (seeded sample)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def metrics(y, score, groups, thr=0.0):
    """score > thr means 'valid / yes'. Accuracy, macro-F1, F1 of the positive class, AUROC, each
    with a cluster-bootstrap CI."""
    y, score, groups = np.asarray(y), np.asarray(score, float), np.asarray(groups)
    pred = lambda s: (s > thr).astype(int)       # noqa: E731
    out = {"n": int(len(y)), "n_clusters": int(len(set(groups))), "frac_positive_labels": float(y.mean()),
           "frac_predicted_positive": float(pred(score).mean()),
           "accuracy": stats.bootstrap_ci(lambda a, s: stats.accuracy(a, pred(s)), (y, score), groups),
           "macro_f1": stats.bootstrap_ci(lambda a, s: stats.macro_f1(a, pred(s)), (y, score), groups),
           "positive_f1": stats.bootstrap_ci(lambda a, s: stats.pos_f1(a, pred(s)), (y, score), groups),
           "auroc": stats.bootstrap_ci(stats.auroc, (y, score), groups)}
    return out


def premise_tfidf_fit_predict(items_tr, y_tr, items_te, C=1.0):
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=C, max_iter=1000).fit(vec.fit_transform([i["context"] for i in items_tr]), y_tr)
    return clf.decision_function(vec.transform([i["context"] for i in items_te]))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


_PAIR = re.compile(r"\b([A-F])\b")


# ----------------------------------------------------------------------------- Corr2Cause: description
def describe_c2c(splits):
    d = {}
    prem = {s: {i["group"] for i in its} for s, its in splits.items()}
    for s, its in splits.items():
        y = np.array([i["label"] for i in its])
        per_prem = Counter(i["group"] for i in its)
        d[s] = {"n_items": len(its), "n_distinct_premises": len(per_prem),
                "items_per_premise_median": float(np.median(list(per_prem.values()))),
                "frac_valid": float(y.mean()),
                "num_variables": dict(sorted(Counter(i["num_variables"] for i in its).items())),
                "frac_valid_by_template": {t: float(np.mean([i["label"] for i in its if i["template"] == t]))
                                           for t in probes.C2C_TEMPLATES},
                "debiasing": data.c2c_debiased_stats(its)}
    d["premise_overlap"] = {f"{a}&{b}": len(prem[a] & prem[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    d["n_dev_items_with_premise_in_train"] = sum(i["group"] in prem["train"] for i in splits["dev"])
    # Mechanism behind the hypothesis-string shortcut: for directed templates, is the claim ever
    # valid when the cause is named with a LATER letter than the effect?
    order = {}
    for t in ("parent", "child", "non-parent ancestor", "non-child descendant"):
        rows = [(i["label"], _PAIR.findall(i["query"])) for i in splits["train"] if i["template"] == t]
        fw = [lab for lab, v in rows if len(v) >= 2 and v[0] < v[1]]
        bw = [lab for lab, v in rows if len(v) >= 2 and v[0] > v[1]]
        order[t] = {"first_named_variable_earlier_in_alphabet": {"n": len(fw), "frac_valid": float(np.mean(fw)) if fw else None},
                    "first_named_variable_later_in_alphabet": {"n": len(bw), "frac_valid": float(np.mean(bw)) if bw else None},
                    "example_hypothesis": next(i["query"] for i in splits["train"] if i["template"] == t)}
    d["train_validity_by_variable_name_order"] = order
    cells = Counter((i["query"], i["label"]) for i in splits["train"])
    strings = sorted({q for q, _ in cells})
    d["train_hypothesis_strings"] = {
        "n": len(strings),
        "n_never_valid": sum(cells[(q, 1)] == 0 for q in strings),
        "n_always_valid": sum(cells[(q, 0)] == 0 for q in strings),
        "n_both": sum(cells[(q, 0)] > 0 and cells[(q, 1)] > 0 for q in strings),
        "frac_of_train_items_whose_string_is_never_valid": float(sum(cells[(q, 0)] for q in strings if cells[(q, 1)] == 0) / len(splits["train"])),
        "n_dev_or_test_strings_unseen_in_train": len({i["query"] for s in ("dev", "test") for i in splits[s]} - set(strings))}
    return d


# ----------------------------------------------------------------------------- Corr2Cause: official splits
def c2c_official(splits):
    tr = splits["train"]
    ytr = np.array([i["label"] for i in tr])
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(len(tr), N_TRAIN_FIT, replace=False))
    tr_sub, y_sub = [tr[j] for j in sub], ytr[sub]
    keys = np.array([i["query"] for i in tr]); tmpl = np.array([i["template"] for i in tr])
    hyp_prior = {k: float(ytr[keys == k].mean()) for k in np.unique(keys)}
    tpl_prior = {k: float(ytr[tmpl == k].mean()) for k in np.unique(tmpl)}
    base = float(ytr.mean())
    # decision threshold for the hypothesis-only prior that maximises positive-class F1 on train
    ptr = np.array([hyp_prior[k] for k in keys])
    cands = sorted(set(hyp_prior.values()))
    f1s = [stats.pos_f1(ytr, (ptr >= c).astype(int)) for c in cands]
    best_thr = cands[int(np.argmax(f1s))]
    log(f"hypothesis-only: threshold maximising train F1 = P(valid|string) >= {best_thr:.3f} (train F1 {max(f1s):.3f})")
    out = {"fit": {"n_train_items": len(tr), "n_train_items_for_learned_baselines": N_TRAIN_FIT,
                   "train_frac_valid": base, "hypothesis_only_f1_threshold": best_thr,
                   "hypothesis_only_train_positive_f1_at_threshold": float(max(f1s))}}
    for split in ("dev", "test"):
        te = splits[split]
        y = np.array([i["label"] for i in te]); g = np.array([i["group"] for i in te])
        hp = np.array([hyp_prior.get(i["query"], base) for i in te])
        res = {"majority": metrics(y, np.full(len(y), -1.0 if base < 0.5 else 1.0), g),
               "template_prior": metrics(y, logit([tpl_prior[i["template"]] for i in te]), g),
               "hypothesis_only": metrics(y, logit(hp), g),
               "hypothesis_only_f1_threshold": metrics(y, hp, g, thr=best_thr - 1e-9),
               "premise_only": metrics(y, premise_tfidf_fit_predict(tr_sub, y_sub, te), g),
               "parsed_features": metrics(y, probes.structural_fit_predict(tr_sub, y_sub, te), g)}
        if split == "dev":      # dev shares premises with train; repeat without them
            keep = np.array([i["group"] not in {j["group"] for j in tr} for i in te])
            res["parsed_features_excluding_premises_seen_in_train"] = metrics(
                y[keep], probes.structural_fit_predict(tr_sub, y_sub, [i for i, k in zip(te, keep) if k]), g[keep])
        out[split] = res
        log(f"c2c official {split}: " + "  ".join(
            f"{k} acc {v['accuracy']['est']:.3f} F1+ {v['positive_f1']['est']:.3f} auroc {v['auroc']['est']:.3f}" for k, v in res.items()))
    return out


# ----------------------------------------------------------------------------- grouped cross-validation pools
def cv_pool(items, name):
    y = np.array([i["label"] for i in items]); g = np.array([i["group"] for i in items])
    arr = np.array(items, dtype=object)
    seed_folds = [probes.folds_for(items, sd) for sd in PROBE_SEEDS]

    def mean_oof(fn):
        return np.mean([fn(f) for f in seed_folds], 0)

    def premise_oof(folds):
        s = np.zeros(len(y), np.float32)
        for tr, te in folds:
            s[te] = premise_tfidf_fit_predict(list(arr[tr]), y[tr], list(arr[te]))
        return s

    sc = {"hypothesis_only": mean_oof(lambda f: probes.hyp_prior_oof(items, y, f)),
          "hypothesis_only_tfidf": mean_oof(lambda f: probes.tfidf_query_oof(items, y, f)),
          "premise_only": mean_oof(premise_oof),
          "full_text_tfidf": mean_oof(lambda f: probes.tfidf_oof(items, y, f, C=1.0)),
          "parsed_features": mean_oof(lambda f: probes.structural_oof(items, y, f))}
    res = {"n": len(y), "n_premises": int(len(set(g))), "frac_valid": float(y.mean()),
           "n_hypothesis_strings": len({i["query"] for i in items}),
           "baselines": {k: metrics(y, v, g) for k, v in sc.items()}}
    log(f"{name}: " + "  ".join(f"{k} acc {v['accuracy']['est']:.3f} auroc {v['auroc']['est']:.3f}" for k, v in res["baselines"].items()))
    return res


# ----------------------------------------------------------------------------- CLadder
def cladder_full():
    items = data.load_cladder()
    y = np.array([i["label"] for i in items]); g = np.array([i["group"] for i in items])
    seed_folds = [probes.folds_for(items, sd) for sd in PROBE_SEEDS]

    def mean_oof(fn):
        return np.mean([fn(f) for f in seed_folds], 0)

    sc = {"type_prior": mean_oof(lambda f: probes.hyp_prior_oof(items, y, f)),
          "question_only": mean_oof(lambda f: probes.tfidf_query_oof(items, y, f)),
          "full_text": mean_oof(lambda f: probes.tfidf_oof(items, y, f, C=1.0)),
          "numeric_polarity": mean_oof(lambda f: probes.structural_oof(items, y, f))}
    res = {"file": "full_v1.5_default.csv", "n": len(y), "n_stories": int(len(set(g))), "frac_yes": float(y.mean()),
           "frac_yes_by_query_type": {t: float(y[[i["query_type"] == t for i in items]].mean()) for t in probes.CL_QTYPES},
           "baselines": {k: metrics(y, v, g) for k, v in sc.items()}, "numeric_polarity_by": {}}
    s = sc["numeric_polarity"]
    for key in ("query_type", "rung"):
        res["numeric_polarity_by"][key] = {}
        for v in sorted({i[key] for i in items}, key=str):
            m = np.array([i[key] == v for i in items])
            res["numeric_polarity_by"][key][str(v)] = {
                "n": int(m.sum()), "frac_yes": float(y[m].mean()),
                "accuracy": stats.bootstrap_ci(lambda a, b: stats.accuracy(a, (b > 0).astype(int)), (y[m], s[m]), g[m], n_boot=1000),
                "auroc": stats.auroc(y[m], s[m])}
    log("cladder: " + "  ".join(f"{k} acc {v['accuracy']['est']:.3f} auroc {v['auroc']['est']:.3f}" for k, v in res["baselines"].items()))
    log("cladder numeric_polarity by query type: " + " ".join(
        f"{k}:{v['accuracy']['est']:.2f}" for k, v in res["numeric_polarity_by"]["query_type"].items()))
    return res


def main():
    res = {"data": {"corr2cause": "causal-nlp/corr2cause @ 42ba12c769e11ff6427c9f52d7db58e3f9bf3e53 (train.csv, dev.csv, test.csv)",
                    "cladder": "causal-nlp/CLadder @ 18d684e444ed07cc57ae0f3a91ee983a79b8119a (data/full_v1.5_default.csv)"},
           "seed": SEED, "cv": {"folds": 5, "seeds": list(PROBE_SEEDS), "grouping": "premise (Corr2Cause) / story (CLadder)"}}
    splits = {s: data.load_c2c(s) for s in ("train", "dev", "test")}
    res["corr2cause_description"] = describe_c2c(splits)
    log("described Corr2Cause")
    res["corr2cause_official_splits"] = c2c_official(splits)
    naive = data.c2c_string_balanced_pool(splits["train"])
    del splits
    res["corr2cause_string_balanced_only_pool_rejected"] = cv_pool(naive, "c2c string-balanced-only pool (rejected design)")
    res["corr2cause_balanced_pool"] = cv_pool(data.get_dataset("c2c_probe"), "c2c balanced pool")
    res["corr2cause_debiased_pool"] = cv_pool(data.get_dataset("c2c_debiased"), "c2c debiased pool")
    res["cladder_full_v1.5"] = cladder_full()
    res["minutes"] = round((time.time() - T0) / 60, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark_baselines.json").write_text(json.dumps(res, indent=1))
    log("wrote results/benchmark_baselines.json")


if __name__ == "__main__":
    main()

"""What is the gold-label probe reading? (CPU, from the cache; post-hoc follow-up, see notes/log.md 13 Nov.)

For one model and a Corr2Cause item set:
  (a) probes, with the same folds, layer and regularisation as the gold probe, for four properties a
      shallow parser reads off the prompt (probes.structural_features) instead of the gold label:
        correlated   the premise says the two variables named in the hypothesis correlate
        adjacent     ... and no conditioning set makes them independent (skeleton edge)
        separable    at least one set makes them independent
        name_order   the first-named variable comes earlier in the alphabet
      each with its majority rate, and how well the gold label follows from that property alone.
  (b) complementarity of the gold probe and the parsed-feature classifier, from their out-of-fold
      scores on the report half: AUROC of one inside the two strata defined by the other's
      prediction, their correlation, and how often each is right when the other is wrong.
Writes results/probe_content_<model>_<tag>.json.
"""
import argparse
import json

import numpy as np
from joblib import Parallel, delayed

from causal_probe import cache, data, probes, stats
from causal_probe.config import ALL_MODELS, PROBE_SEEDS, RESULTS


def oof_probe(X, target, folds, C):
    s = np.zeros(len(target), np.float32)
    for tr, te in folds:
        if len(np.unique(target[tr])) < 2:
            s[te] = (target[tr][0] * 2 - 1)
        else:
            s[te] = probes.fit_predict(X[tr], target[tr], X[te], C=C)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", default="c2c_debiased", choices=["c2c_debiased", "c2c_probe"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    c = json.loads((RESULTS / f"confound_{a.model}_{a.tag}.json").read_text())
    L, C = c["selected_layer_gold"], c["selected_C_gold"]
    resid, meta, items = cache.load_cache(a.model, a.tag)
    y = np.array(meta["labels"]); groups = np.array([i["group"] for i in items])
    rep = ~data.selection_mask(items)
    F = probes.structural_features(items)
    targets = {"correlated": F[:, 0].astype(int), "adjacent": F[:, 2].astype(int),
               "separable": (F[:, 3] > 0).astype(int), "name_order": F[:, 15].astype(int)}
    seed_folds = [probes.folds_for(items, sd) for sd in PROBE_SEEDS]
    ci = lambda fn, *arrs: stats.bootstrap_ci(fn, tuple(x[rep] for x in arrs), groups[rep], n_boot=1000)     # noqa: E731
    acc = lambda t, s: stats.accuracy(t, (s > 0).astype(int))                                                # noqa: E731
    res = {"model": a.model, "tag": a.tag, "layer": L, "probe_C": C, "n_report": int(rep.sum()), "properties": {}, "positions": list(meta["positions"])}
    pos_ids = {p: i for i, p in enumerate(meta["positions"]) if p in ("final", "content_end")}
    for name, t in targets.items():
        r = {"frac_positive": float(t.mean()), "majority_rate": float(max(t.mean(), 1 - t.mean())),
             # how well the gold label follows from this property alone (better of the two polarities)
             "gold_accuracy_from_property_alone": float(max(stats.accuracy(y[rep], t[rep]), stats.accuracy(y[rep], 1 - t[rep]))),
             "probe": {}}
        for pname, pi in pos_ids.items():
            X = probes.layer_features(resid, L, pi)
            S = np.mean(Parallel(n_jobs=a.jobs)(delayed(oof_probe)(X, t, f, C) for f in seed_folds), 0)
            r["probe"][pname] = {"accuracy": ci(acc, t, S), "auroc": ci(stats.auroc, t, S)}
        res["properties"][name] = r
        print(f"{a.model}/{a.tag} {name:11s} majority {r['majority_rate']:.3f}  gold-from-it {r['gold_accuracy_from_property_alone']:.3f}  "
              + "  ".join(f"probe@{p} {v['accuracy']['est']:.3f}" for p, v in r["probe"].items()), flush=True)
    # (b) complementarity on the report half
    oof = np.load(cache.cache_dir(a.model, a.tag) / f"probe_oof_C{C:g}.npz")
    sp = oof["gold"][:, L].mean(0)
    st = np.mean([probes.structural_oof(items, y, f) for f in seed_folds], 0)
    pp, pt = (sp > 0), (st > 0)
    comp = {"pearson_r_scores": float(np.corrcoef(sp[rep], st[rep])[0, 1]),
            "probe_accuracy": ci(acc, y, sp), "parsed_accuracy": ci(acc, y, st),
            "probe_auroc_where_parsed_says_valid": stats.bootstrap_ci(stats.auroc, (y[rep & pt], sp[rep & pt]), groups[rep & pt], n_boot=1000),
            "probe_auroc_where_parsed_says_invalid": stats.bootstrap_ci(stats.auroc, (y[rep & ~pt], sp[rep & ~pt]), groups[rep & ~pt], n_boot=1000),
            "parsed_auroc_where_probe_says_valid": stats.bootstrap_ci(stats.auroc, (y[rep & pp], st[rep & pp]), groups[rep & pp], n_boot=1000),
            "parsed_auroc_where_probe_says_invalid": stats.bootstrap_ci(stats.auroc, (y[rep & ~pp], st[rep & ~pp]), groups[rep & ~pp], n_boot=1000),
            "probe_right_when_parsed_wrong": float(np.mean((pp == y)[rep & (pt != y)])), "n_parsed_wrong": int((rep & (pt != y)).sum()),
            "parsed_right_when_probe_wrong": float(np.mean((pt == y)[rep & (pp != y)])), "n_probe_wrong": int((rep & (pp != y)).sum()),
            "either_right": float(np.mean(((pp == y) | (pt == y))[rep]))}
    # Does the probe add anything to the parser? Second-level logistic regression on the two out-of-fold
    # scores, cross-validated over the same folds (each item's level-1 scores never saw its own label).
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([sp / sp.std(), st / st.std()])
    stack = np.zeros(len(y), np.float32); w = []
    for tr, te in seed_folds[0]:
        clf = LogisticRegression(C=1.0).fit(Z[tr], y[tr])
        stack[te] = clf.decision_function(Z[te]); w.append(clf.coef_[0])
    diff = lambda t, u, v: stats.accuracy(t, (u > 0).astype(int)) - stats.accuracy(t, (v > 0).astype(int))      # noqa: E731
    comp["stacked_probe_plus_parsed"] = {
        "accuracy": ci(acc, y, stack), "auroc": ci(stats.auroc, y, stack),
        "minus_parsed_alone_accuracy": ci(diff, y, stack, st), "minus_probe_alone_accuracy": ci(diff, y, stack, sp),
        "mean_weights_probe_parsed": np.mean(w, 0).tolist(), "parsed_alone_auroc": ci(stats.auroc, y, st)}
    res["complementarity"] = comp
    (RESULTS / f"probe_content_{a.model}_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"   probe AUROC inside parsed-says-valid {comp['probe_auroc_where_parsed_says_valid']['est']:.3f} / parsed-says-invalid {comp['probe_auroc_where_parsed_says_invalid']['est']:.3f};"
          f" parsed AUROC inside probe-says-valid {comp['parsed_auroc_where_probe_says_valid']['est']:.3f} / invalid {comp['parsed_auroc_where_probe_says_invalid']['est']:.3f};"
          f" r {comp['pearson_r_scores']:.2f}; probe right when parsed wrong {comp['probe_right_when_parsed_wrong']:.2f} (n={comp['n_parsed_wrong']})")
    k = comp["stacked_probe_plus_parsed"]
    print(f"   stacked probe+parsed acc {k['accuracy']['est']:.3f} [{k['accuracy']['lo']:.3f},{k['accuracy']['hi']:.3f}]; minus parsed alone "
          f"{k['minus_parsed_alone_accuracy']['est']:+.3f} [{k['minus_parsed_alone_accuracy']['lo']:+.3f},{k['minus_parsed_alone_accuracy']['hi']:+.3f}]; weights {np.round(k['mean_weights_probe_parsed'], 2)}")


if __name__ == "__main__":
    main()

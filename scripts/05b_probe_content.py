"""What does the gold-label probe add, and what does it read? Debiased Corr2Cause pool, CPU only.
POST-HOC analysis (designed 13 Nov after the debiased-pool probe results; see notes/log.md). It does
not change the pre-registered or the audited criterion; those stay in 05_confound.py.

  python scripts/05b_probe_content.py --model qwen1.5b          # -> results/probe_content_<model>_c2c_debiased.json

A. Baselines in two groups (report half; same items, same folds as the probe):
     shallow confound controls   things a probe could exploit without reading the premise's structure:
                                 bag-of-words of the whole prompt, hypothesis-string prior, premise-only
                                 bag-of-words, a probe on a randomly initialised copy of the model, and
                                 the probe trained on shuffled labels.
     structure-aware reference   the parsed-feature classifier, which binds the premise's statements to
                                 the two queried variables: an approximation of solving the task from the
                                 text, not a confound.
B. Incremental predictive value: gradient boosting on the parsed features alone versus the parsed
   features plus the probe's score as one extra column. The probe score is produced by NESTED
   cross-validation (for training rows: probes fitted inside the training folds only; for test rows:
   a probe fitted on all training rows), so no label of a test row reaches any feature.
C. Which parsed properties are decodable: a probe for each binary property, by layer, with a
   shuffled-label control and the same probe on the random-init model.
D. Stratified analysis: the gold probe's AUROC inside strata of the strongest parsed properties, and
   the stratum-weighted AUROC for each stratification, next to the unstratified AUROC. If the probe's
   signal were carried by one property, its AUROC would fall to 0.5 inside that property's strata.
"""
import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed

from causal_probe import cache, data, probes, stats
from causal_probe.config import ALL_MODELS, PROBE_SEEDS, RESULTS

TAG = "c2c_debiased"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def properties(items):
    F = probes.structural_features(items)
    t = {"pair_correlated": F[:, 0], "pair_marginally_independent": F[:, 1], "pair_adjacent_in_skeleton": F[:, 2],
         "pair_has_separating_set": F[:, 3] > 0, "pair_has_common_neighbour": F[:, 8] > 0,
         "common_neighbour_in_a_separating_set": F[:, 9] > 0, "common_neighbour_in_no_separating_set": F[:, 10] > 0,
         "six_variables": F[:, 11] == 6, "many_correlation_statements": F[:, 12] >= np.median(F[:, 12]),
         "any_marginal_independence_statement": F[:, 13] > 0,
         "many_conditional_independence_statements": F[:, 14] >= np.median(F[:, 14])}
    return F, {k: np.asarray(v).astype(int) for k, v in t.items()}


def oof(X, t, folds, C, shuffle_seed=None):
    s = np.zeros(len(t), np.float32)
    for tr, te in folds:
        ttr = t[tr] if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(t[tr])
        s[te] = probes.fit_predict(X[tr], ttr, X[te], C=C) if len(np.unique(ttr)) > 1 else 0.0
    return s


def layer_job(resid, rres, l, targets, folds, C):
    X = probes.layer_features(resid, l, 0)
    Xr = probes.layer_features(rres, l, 0) if rres is not None else None
    out = {}
    for name, t in targets.items():
        out[name] = {"trained": oof(X, t, folds, C), "shuffled": oof(X, t, folds, C, shuffle_seed=l),
                     "random_init": oof(Xr, t, folds, C) if Xr is not None else None}
    return out


def nested_probe_feature(X, y, items, outer, C):
    """Probe score for every item such that no label of its own outer fold is used anywhere."""
    f = np.zeros(len(y), np.float32)          # as a feature for TEST rows of each outer fold
    ftr = {}                                  # per outer fold: feature for its TRAIN rows (inner out-of-fold)
    for k, (tr, te) in enumerate(outer):
        f[te] = probes.fit_predict(X[tr], y[tr], X[te], C=C)
        inner = data.grouped_folds([items[j] for j in tr], 4, seed=100 + k)
        g = np.zeros(len(tr), np.float32)
        for a_, b_ in inner:
            g[b_] = probes.fit_predict(X[tr][a_], y[tr][a_], X[tr][b_], C=C)
        ftr[k] = g
    return f, ftr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--layer-stride", type=int, default=1)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    c = json.loads((RESULTS / f"confound_{a.model}_{TAG}.json").read_text())
    L, C = c["selected_layer_gold"], c["selected_C_gold"]
    resid, meta, items = cache.load_cache(a.model, TAG)
    has_rand = (cache.cache_dir(a.model, TAG, True) / "meta.json").exists()
    rres = cache.load_cache(a.model, TAG, random_init=True)[0] if has_rand else None
    y = np.array(meta["labels"]); groups = np.array([i["group"] for i in items])
    rep = ~data.selection_mask(items)
    F, props = properties(items)
    props = {k: v for k, v in props.items() if 0.02 <= v.mean() <= 0.98}
    seed_folds = [probes.folds_for(items, sd) for sd in PROBE_SEEDS]
    acc = lambda t, s: stats.accuracy(t, (s > 0).astype(int))                                                   # noqa: E731
    ci = lambda fn, *arrs, m=None: stats.bootstrap_ci(fn, tuple(x[rep if m is None else m] for x in arrs),     # noqa: E731
                                                       groups[rep if m is None else m], n_boot=1000)
    dacc = lambda t, u, v: acc(t, u) - acc(t, v)                                                                # noqa: E731
    dauc = lambda t, u, v: stats.auroc(t, u) - stats.auroc(t, v)                                                # noqa: E731
    res = {"model": a.model, "tag": TAG, "layer": L, "n_layers": int(resid.shape[1] - 1), "probe_C": C,
           "n_report": int(rep.sum()), "status": "post-hoc analysis; does not alter the pre-registered or audited criterion"}

    # ---------------------------------------------------------------- A. baselines in two groups
    po = np.load(cache.cache_dir(a.model, TAG) / f"probe_oof_C{C:g}.npz")
    sp = po["gold"][:, L].mean(0)
    arr = np.array(items, dtype=object)

    def premise_only(folds):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        s = np.zeros(len(y), np.float32)
        for tr, te in folds:
            vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=2, sublinear_tf=True)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(vec.fit_transform([i["context"] for i in arr[tr]]), y[tr])
            s[te] = clf.decision_function(vec.transform([i["context"] for i in arr[te]]))
        return s

    st = np.mean([probes.structural_oof(items, y, f) for f in seed_folds], 0)
    shallow = {"bag_of_words_whole_prompt": po[f"tfidf_C{c['selected_C_tfidf']:g}"].mean(0) if f"tfidf_C{c['selected_C_tfidf']:g}" in po.files else po["tfidf"].mean(0),
               "hypothesis_string_prior": np.mean([probes.hyp_prior_oof(items, y, f) for f in seed_folds], 0),
               "bag_of_words_premise_only": np.mean([premise_only(f) for f in seed_folds], 0),
               "probe_on_shuffled_labels": po["shuffled"][:, L].mean(0)}
    if "random_init" in po.files:      # layer and C for the random-init probe as chosen (selection half) in 05_confound.py
        pr_ = np.load(cache.cache_dir(a.model, TAG) / f"probe_oof_C{c.get('selected_C_random_init', C):g}.npz")
        shallow["probe_on_random_init_model"] = pr_["random_init"][:, c.get("selected_layer_random_init", L)].mean(0)

    def block(s):
        return {"accuracy": ci(acc, y, s), "auroc": ci(stats.auroc, y, s),
                "probe_minus_this_accuracy": ci(dacc, y, sp, s), "probe_minus_this_auroc": ci(dauc, y, sp, s)}
    res["baselines"] = {"gold_probe": {"accuracy": ci(acc, y, sp), "auroc": ci(stats.auroc, y, sp)},
                        "shallow_confound_controls": {k: block(v) for k, v in shallow.items()},
                        "structure_aware_reference": {"parsed_features": block(st)}}
    strongest = max(shallow, key=lambda k: res["baselines"]["shallow_confound_controls"][k]["accuracy"]["est"])
    res["baselines"]["strongest_shallow_control"] = strongest
    res["baselines"]["probe_beats_every_shallow_control"] = bool(all(
        v["probe_minus_this_accuracy"]["lo"] > 0 for v in res["baselines"]["shallow_confound_controls"].values()))
    res["baselines"]["probe_beats_structure_aware_reference"] = bool(
        res["baselines"]["structure_aware_reference"]["parsed_features"]["probe_minus_this_accuracy"]["lo"] > 0)
    log("A: probe %.3f | shallow: %s | parsed %.3f" % (
        res["baselines"]["gold_probe"]["accuracy"]["est"],
        " ".join(f"{k} {v['accuracy']['est']:.3f}" for k, v in res["baselines"]["shallow_confound_controls"].items()),
        res["baselines"]["structure_aware_reference"]["parsed_features"]["accuracy"]["est"]))

    # ---------------------------------------------------------------- B. incremental predictive value (nested)
    X = probes.layer_features(resid, L, 0)
    base_s, plus_s = [], []
    for folds in seed_folds:
        f_te, f_tr = nested_probe_feature(X, y, items, folds, C)
        sb, spl = np.zeros(len(y), np.float32), np.zeros(len(y), np.float32)
        for k, (tr, te) in enumerate(folds):
            for dest, Xtr, Xte in ((sb, F[tr], F[te]), (spl, np.column_stack([F[tr], f_tr[k]]), np.column_stack([F[te], f_te[te]]))):
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, y[tr])
                pr = np.clip(clf.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
                dest[te] = np.log(pr / (1 - pr))
        base_s.append(sb); plus_s.append(spl)
    sb, spl = np.mean(base_s, 0), np.mean(plus_s, 0)
    res["incremental_value"] = {
        "method": "gradient boosting on parsed features vs parsed features + nested out-of-fold probe score; 5 folds x 3 seeds",
        "parsed_features_only": {"accuracy": ci(acc, y, sb), "auroc": ci(stats.auroc, y, sb)},
        "parsed_features_plus_probe": {"accuracy": ci(acc, y, spl), "auroc": ci(stats.auroc, y, spl)},
        "gain_accuracy": ci(dacc, y, spl, sb), "gain_auroc": ci(dauc, y, spl, sb),
        "probe_right_when_parsed_wrong": float(np.mean(((sp > 0) == y)[rep & ((sb > 0) != y)])), "n_parsed_wrong": int((rep & ((sb > 0) != y)).sum()),
        "parsed_right_when_probe_wrong": float(np.mean(((sb > 0) == y)[rep & ((sp > 0) != y)])), "n_probe_wrong": int((rep & ((sp > 0) != y)).sum()),
        "pearson_r_probe_vs_parsed_score": float(np.corrcoef(sp[rep], sb[rep])[0, 1])}
    g = res["incremental_value"]
    log("B: parsed only %.3f, + probe %.3f, gain %+.3f [%+.3f, %+.3f] (AUROC gain %+.3f [%+.3f, %+.3f])" % (
        g["parsed_features_only"]["accuracy"]["est"], g["parsed_features_plus_probe"]["accuracy"]["est"],
        g["gain_accuracy"]["est"], g["gain_accuracy"]["lo"], g["gain_accuracy"]["hi"],
        g["gain_auroc"]["est"], g["gain_auroc"]["lo"], g["gain_auroc"]["hi"]))

    # ---------------------------------------------------------------- C. decodability of parsed properties, by layer
    layers = sorted(set(range(0, resid.shape[1], a.layer_stride)) | {L, resid.shape[1] - 1})
    per = Parallel(n_jobs=a.jobs)(delayed(layer_job)(resid, rres, l, props, seed_folds[0], C) for l in layers)
    res["properties"] = {"layers": layers, "folds": "5 block-wise folds, seed 0", "by_property": {}}
    for name, t in props.items():
        r = {"frac_positive": float(t.mean()), "majority_rate": float(max(t.mean(), 1 - t.mean())),
             "gold_accuracy_from_property_alone": float(max(stats.accuracy(y[rep], t[rep]), stats.accuracy(y[rep], 1 - t[rep]))),
             "by_layer": {k: {"accuracy": [], "auroc": []} for k in ("trained", "shuffled", "random_init")}}
        for li, l in enumerate(layers):
            for k in ("trained", "shuffled", "random_init"):
                s = per[li][name][k]
                r["by_layer"][k]["accuracy"].append(None if s is None else acc(t[rep], s[rep]))
                r["by_layer"][k]["auroc"].append(None if s is None else stats.auroc(t[rep], s[rep]))
        li = layers.index(L)
        tr_s, sh_s, rd_s = per[li][name]["trained"], per[li][name]["shuffled"], per[li][name]["random_init"]
        r["at_gold_probe_layer"] = {
            "trained": {"accuracy": ci(acc, t, tr_s), "auroc": ci(stats.auroc, t, tr_s)},
            "shuffled": {"accuracy": ci(acc, t, sh_s), "auroc": ci(stats.auroc, t, sh_s)},
            "random_init": None if rd_s is None else {"accuracy": ci(acc, t, rd_s), "auroc": ci(stats.auroc, t, rd_s)},
            "trained_minus_random_init_auroc": None if rd_s is None else ci(dauc, t, tr_s, rd_s)}
        best = int(np.nanargmax(r["by_layer"]["trained"]["auroc"]))
        r["best_layer_trained_auroc"] = {"layer": layers[best], "auroc": r["by_layer"]["trained"]["auroc"][best],
                                         "random_init_auroc_same_layer": r["by_layer"]["random_init"]["auroc"][best]}
        res["properties"]["by_property"][name] = r
        x = r["at_gold_probe_layer"]
        log(f"C: {name:42s} maj {r['majority_rate']:.2f} | AUROC trained {x['trained']['auroc']['est']:.3f} shuffled {x['shuffled']['auroc']['est']:.3f} "
            f"random-init {x['random_init']['auroc']['est'] if x['random_init'] else float('nan'):.3f} | acc {x['trained']['accuracy']['est']:.3f}")

    # ---------------------------------------------------------------- D. stratified analysis
    tmpl = np.array([i["template"] for i in items])
    strat = {"template": tmpl, **{k: props[k] for k in ("pair_adjacent_in_skeleton", "pair_has_separating_set", "pair_correlated",
                                                         "pair_has_common_neighbour", "common_neighbour_in_no_separating_set") if k in props},
             "parsed_classifier_prediction": (sb > 0).astype(int)}
    strat["template_x_adjacent_x_common_neighbour"] = np.array([f"{a_}|{b_}|{c_}" for a_, b_, c_ in zip(
        tmpl, props["pair_adjacent_in_skeleton"], props.get("pair_has_common_neighbour", np.zeros(len(y), int)))])
    overall = stats.auroc(y[rep], sp[rep])

    def weighted_auroc(yy, ss, zz):
        num = den = 0.0
        for v in np.unique(zz):
            m = zz == v
            n1, n0 = int(yy[m].sum()), int((1 - yy[m]).sum())
            if n1 and n0:
                num += n1 * n0 * stats.auroc(yy[m], ss[m]); den += n1 * n0
        return num / den if den else float("nan")

    res["stratified"] = {"overall_probe_auroc": ci(stats.auroc, y, sp), "by_stratifier": {}}
    for name, z in strat.items():
        rows = {}
        for v in np.unique(z):
            m = rep & (z == v)
            if m.sum() < 30 or len(np.unique(y[m])) < 2:
                continue
            rows[str(v)] = {"n": int(m.sum()), "frac_gold_valid": float(y[m].mean()),
                            "probe_auroc": stats.bootstrap_ci(stats.auroc, (y[m], sp[m]), groups[m], n_boot=1000),
                            "parsed_auroc": stats.auroc(y[m], sb[m])}
        res["stratified"]["by_stratifier"][name] = {
            "strata": rows,
            "pair_weighted_within_stratum_probe_auroc": stats.bootstrap_ci(weighted_auroc, (y[rep], sp[rep], z[rep]), groups[rep], n_boot=1000),
            # accuracy of predicting the gold label from the stratum alone (majority label per stratum)
            "gold_accuracy_from_stratifier_alone": float(sum(max(y[rep & (z == v)].sum(), (1 - y[rep & (z == v)]).sum())
                                                             for v in np.unique(z)) / rep.sum())}
        w = res["stratified"]["by_stratifier"][name]
        log(f"D: {name:40s} gold-from-strata {w['gold_accuracy_from_stratifier_alone']:.3f} | within-stratum probe AUROC "
            f"{w['pair_weighted_within_stratum_probe_auroc']['est']:.3f} (overall {overall:.3f}) | "
            + " ".join(f"{k}:{v['probe_auroc']['est']:.2f}(n={v['n']})" for k, v in list(rows.items())[:6]))
    res["minutes"] = round((time.time() - T0) / 60, 1)
    (RESULTS / f"probe_content_{a.model}_{TAG}.json").write_text(json.dumps(res, indent=1))
    log("wrote results")


if __name__ == "__main__":
    main()

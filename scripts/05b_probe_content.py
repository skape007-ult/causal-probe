"""What does the gold-label probe add, and what does it read? Debiased Corr2Cause pool, CPU only.

POST-HOC analysis (13 Nov; requested after the debiased-pool probe results, then revised after the
third independent audit, notes/audit_debiased_raw.md). It does not change the pre-registered or the
audited criterion, which stay in 05_confound.py.

  python scripts/05b_probe_content.py --model qwen1.5b      # -> results/probe_content_<model>_c2c_debiased.json

In this pool every hypothesis string and every premise is exactly half valid in every fold. So a
classifier that sees only one of them, and any model that is ADDITIVE in premise and hypothesis
features (a linear bag-of-words of the whole prompt), is at 0.50 by construction. Label information
can only come from an INTERACTION between hypothesis and premise. The baselines are therefore a
ladder, ordered by how much they bind the premise to the hypothesis:

  A0 single-input controls (0.50 by construction; listed to show the construction works)
       hypothesis-string prior · premise-only bag-of-words · whole-prompt bag-of-words (linear)
       · the probe trained on shuffled labels (null distribution over 20 shuffles)
  A  interaction controls that bind nothing to the queried variables ("shallow confound controls")
       template x premise-level counts · template x premise bag-of-words · probe on the random-init model
  B  text references that bind premise statements to the queried variables ("structure-aware")
       template x per-variable statement counts · template x statements naming the pair
       · per-hypothesis-string bag of premise n-grams (no parser) · hand-written parse (all features)
       · strongest text-only model here (hand parse stacked with the per-string n-gram model)

  B'. Incremental value of the probe over (i) the hand parse and (ii) the strongest text-only model:
      nested logistic stacking (every score a row is trained or tested on was produced without that
      row's label and without any label of its outer test fold). The earlier "add one column to
      gradient boosting" test is kept for the record; the audit showed it cannot detect a 2-3 point gain.
  C.  Probes for parsed properties of the item, with a shuffled-label null (10 shuffles), the
      random-init model, text baselines for the same property, and the end-of-premise position.
  D.  Stratified analysis: gold-probe AUROC inside strata of single properties, of the parse score
      (quintiles) and of two text scores jointly (25 cells).
  E.  The model's own answers by hypothesis template: does the answer logit track validity, and
      the probe's score, within items that share the hypothesis string?
  F.  Robustness of the headline probe number: single seeds, shuffled-label null, training on the
      selection half only.
"""
import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from causal_probe import cache, data, probes, stats
from causal_probe.config import ALL_MODELS, PROBE_SEEDS, RESULTS

TAG = "c2c_debiased"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- text features
def per_variable_counts(items):
    """For each of the two variables named in the hypothesis SEPARATELY: how many premise statements of
    each kind mention it. Binds statements to single variables, not to the pair."""
    rows = []
    for it in items:
        p = probes.parse_premise(it["context"])
        hv = probes._VAR.findall(it["query"])
        f = []
        for v in (hv + ["?", "?"])[:2]:
            f += [sum(v in q for q in p["corr"]), sum(v in q for q in p["ind"]),
                  sum(len(sets) for q, sets in p["cind"].items() if v in q),
                  sum(v in s for sets in p["cind"].values() for s in sets)]
        rows.append(f)
    return np.array(rows, np.float32)


def logit(pr):
    pr = np.clip(pr, 1e-4, 1 - 1e-4)
    return np.log(pr / (1 - pr)).astype(np.float32)


def by_key_gbm(keys, X):
    """Component: a separate gradient-boosting model for every value of `keys` (template), i.e. a full
    interaction between that key and the features. (One model with the key as a feature finds nothing:
    every first split has zero gain in this pool.)"""
    def comp(tr, te, y):
        s = np.zeros(len(te), np.float32)
        for k in np.unique(keys[te]):
            a, b = tr[keys[tr] == k], np.where(keys[te] == k)[0]
            if len(a) and len(np.unique(y[a])) > 1:
                clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0).fit(X[a], y[a])
                s[b] = logit(clf.predict_proba(X[te[b]])[:, 1])
        return s
    return comp


def by_key_bow(keys, texts, ngram=(1, 2), C=1.0):
    """Component: a separate TF-IDF + logistic regression on the premise text for every value of `keys`."""
    def comp(tr, te, y):
        s = np.zeros(len(te), np.float32)
        for k in np.unique(keys[te]):
            a, b = tr[keys[tr] == k], np.where(keys[te] == k)[0]
            if len(a) and len(np.unique(y[a])) > 1:
                vec = TfidfVectorizer(ngram_range=ngram, lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=1, sublinear_tf=True)
                clf = LogisticRegression(C=C, max_iter=1000).fit(vec.fit_transform(texts[a]), y[a])
                s[b] = clf.decision_function(vec.transform(texts[te[b]]))
        return s
    return comp


def gbm_all(X):
    def comp(tr, te, y):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr], y[tr])
        return logit(clf.predict_proba(X[te])[:, 1])
    return comp


def probe_comp(X, C):
    def comp(tr, te, y):
        return probes.fit_predict(X[tr], y[tr], X[te], C=C) if len(np.unique(y[tr])) > 1 else np.zeros(len(te), np.float32)
    return comp


def oof(comp, folds, y):
    s = np.zeros(len(y), np.float32)
    for tr, te in folds:
        s[te] = comp(tr, te, y)
    return s


def nested(comp, outer, y, items):
    """Scores for stacking. For each outer fold: test-row scores from a model fitted on all training
    rows, and training-row scores from an inner 4-fold cross-validation inside the training rows."""
    te_s = np.zeros(len(y), np.float32); tr_s = []
    for k, (tr, te) in enumerate(outer):
        te_s[te] = comp(tr, te, y)
        g = np.zeros(len(tr), np.float32)
        for a_, b_ in data.grouped_folds([items[j] for j in tr], 4, seed=100 + k):
            g[b_] = comp(tr[a_], tr[b_], y)
        tr_s.append(g)
    return te_s, tr_s


def stack(parts, outer, y):
    """Logistic regression on the listed components' scores, fitted per outer fold on inner out-of-fold
    training scores and applied to the test rows."""
    s = np.zeros(len(y), np.float32)
    for k, (tr, te) in enumerate(outer):
        Ztr = np.column_stack([p[1][k] for p in parts]); Zte = np.column_stack([p[0][te] for p in parts])
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        clf = LogisticRegression(C=1.0).fit((Ztr - mu) / sd, y[tr])
        s[te] = clf.decision_function((Zte - mu) / sd)
    return s


def property_layer_job(resid, rres, l, props, folds, C):
    """Property probes at one layer for the trained and the random-init model (top-level so that joblib
    passes the memory-mapped caches by reference)."""
    Xl = probes.layer_features(resid, l, 0)
    Xr = probes.layer_features(rres, l, 0) if rres is not None else None
    return {n: (oof(probe_comp(Xl, C), folds, t), oof(probe_comp(Xr, C), folds, t) if Xr is not None else None) for n, t in props.items()}


def weighted_auroc(yy, ss, zz):
    """AUROC computed only over valid/invalid pairs that share a stratum (pair-weighted mean of the
    within-stratum AUROCs)."""
    num = den = 0.0
    for v in np.unique(zz):
        m = zz == v
        n1, n0 = int(yy[m].sum()), int((1 - yy[m]).sum())
        if n1 and n0:
            num += n1 * n0 * stats.auroc(yy[m], ss[m]); den += n1 * n0
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--layer-stride", type=int, default=2)
    a = ap.parse_args()
    c = json.loads((RESULTS / f"confound_{a.model}_{TAG}.json").read_text())
    L, C = c["selected_layer_gold"], c["selected_C_gold"]
    resid, meta, items = cache.load_cache(a.model, TAG)
    has_rand = (cache.cache_dir(a.model, TAG, True) / "meta.json").exists()
    rres, rmeta = (cache.load_cache(a.model, TAG, random_init=True)[:2] if has_rand else (None, None))
    y = np.array(meta["labels"]); groups = np.array([i["group"] for i in items]); ld = np.array(meta["logit_diff"])
    rep = ~data.selection_mask(items); sel = ~rep
    tmpl = np.array([i["template"] for i in items]); strs = np.array([i["query"] for i in items])
    prem = np.array([i["context"] for i in items], dtype=object)
    F = probes.structural_features(items)
    Fcounts, Fpair, Fvar = F[:, 11:15], F[:, 0:6], np.column_stack([per_variable_counts(items), F[:, 11:15]])
    seed_folds = [probes.folds_for(items, sd) for sd in PROBE_SEEDS]
    acc = lambda t, s: stats.accuracy(t, (s > 0).astype(int))                                               # noqa: E731
    ci = lambda fn, *arrs, m=None: stats.bootstrap_ci(fn, tuple(x[rep if m is None else m] for x in arrs),  # noqa: E731
                                                       groups[rep if m is None else m], n_boot=1000)
    dacc = lambda t, u, v: acc(t, u) - acc(t, v)                                                            # noqa: E731
    dauc = lambda t, u, v: stats.auroc(t, u) - stats.auroc(t, v)                                            # noqa: E731
    mean_oof = lambda comp: np.mean([oof(comp, f, y) for f in seed_folds], 0)                               # noqa: E731
    res = {"model": a.model, "tag": TAG, "layer": L, "n_layers": int(resid.shape[1] - 1), "probe_C": C, "n_report": int(rep.sum()),
           "status": "post-hoc analysis (13 Nov), revised after the third audit; does not alter the pre-registered or audited criterion"}

    # ---------------------------------------------------------------- A. baseline ladder
    po = np.load(cache.cache_dir(a.model, TAG) / f"probe_oof_C{C:g}.npz")
    sp_seeds = po["gold"][:, L]; sp = sp_seeds.mean(0)
    X = probes.layer_features(resid, L, 0)
    ladder = {"A0_single_input": {
        "hypothesis_string_prior": np.mean([probes.hyp_prior_oof(items, y, f) for f in seed_folds], 0),
        "bag_of_words_premise_only": mean_oof(by_key_bow(np.zeros(len(y), int), prem)),
        "bag_of_words_whole_prompt_linear": np.mean([probes.tfidf_oof(items, y, f, C=1.0) for f in seed_folds], 0)},
        "A_no_binding_interaction": {
        "template_x_premise_counts": mean_oof(by_key_gbm(tmpl, Fcounts)),
        "template_x_premise_bag_of_words": mean_oof(by_key_bow(tmpl, prem))},
        "B_binding_text_reference": {
        "template_x_per_variable_statement_counts": mean_oof(by_key_gbm(tmpl, Fvar)),
        "template_x_statements_naming_the_pair": mean_oof(by_key_gbm(tmpl, Fpair)),
        "per_string_premise_bigrams": mean_oof(by_key_bow(strs, prem, (1, 2))),
        "per_string_premise_4grams": mean_oof(by_key_bow(strs, prem, (1, 4))),
        "hand_parse_all_features": mean_oof(gbm_all(F))}}
    if "random_init" in po.files:
        pr_ = np.load(cache.cache_dir(a.model, TAG) / f"probe_oof_C{c.get('selected_C_random_init', C):g}.npz")
        ladder["A_no_binding_interaction"]["probe_on_random_init_model"] = pr_["random_init"][:, c.get("selected_layer_random_init", L)].mean(0)
    log("A: component scores done")

    # nested scores for stacking (parse, per-string 4-grams, probe), per seed
    comps = {"parse": gbm_all(F), "ngram4": by_key_bow(strs, prem, (1, 4)), "probe": probe_comp(X, C)}
    nest = {k: Parallel(n_jobs=a.jobs)(delayed(nested)(fn, f, y, items) for f in seed_folds) for k, fn in comps.items()}
    stk = lambda names: np.mean([stack([nest[n][si] for n in names], f, y) for si, f in enumerate(seed_folds)], 0)   # noqa: E731
    s_parse, s_parse_probe = stk(["parse"]), stk(["parse", "probe"])
    s_text, s_text_probe = stk(["parse", "ngram4"]), stk(["parse", "ngram4", "probe"])
    ladder["B_binding_text_reference"]["strongest_text_only_parse_plus_4grams"] = s_text
    log("A/B': nested stacks done")

    def block(s):
        return {"accuracy": ci(acc, y, s), "auroc": ci(stats.auroc, y, s),
                "probe_minus_this_accuracy": ci(dacc, y, sp, s), "probe_minus_this_auroc": ci(dauc, y, sp, s)}
    res["baselines"] = {"gold_probe": {"accuracy": ci(acc, y, sp), "auroc": ci(stats.auroc, y, sp)},
                        **{g: {k: block(v) for k, v in d.items()} for g, d in ladder.items()}}
    for g, d in res["baselines"].items():
        if g != "gold_probe":
            log(f"A {g}: " + "  ".join(f"{k} {v['accuracy']['est']:.3f}" for k, v in d.items()))
    A_ = res["baselines"]["A_no_binding_interaction"]; B_ = res["baselines"]["B_binding_text_reference"]
    res["baselines"]["summary"] = {
        "probe_above_every_no_binding_control": bool(all(v["probe_minus_this_accuracy"]["lo"] > 0 for v in A_.values())),
        "strongest_no_binding_control": max(A_, key=lambda k: A_[k]["accuracy"]["est"]),
        "binding_references_the_probe_exceeds": [k for k, v in B_.items() if v["probe_minus_this_accuracy"]["lo"] > 0],
        "binding_references_not_distinguishable_from_probe": [k for k, v in B_.items() if v["probe_minus_this_accuracy"]["lo"] <= 0 <= v["probe_minus_this_accuracy"]["hi"]],
        "binding_references_above_the_probe": [k for k, v in B_.items() if v["probe_minus_this_accuracy"]["hi"] < 0]}

    # ---------------------------------------------------------------- B'. incremental value
    # the earlier, insensitive test, kept for the record: one extra column in gradient boosting
    sb_l, spl_l = [], []
    for si, folds in enumerate(seed_folds):
        f_te, f_tr = nest["probe"][si]
        sb, spl = np.zeros(len(y), np.float32), np.zeros(len(y), np.float32)
        for k, (tr, te) in enumerate(folds):
            for dest, Xtr, Xte in ((sb, F[tr], F[te]), (spl, np.column_stack([F[tr], f_tr[k]]), np.column_stack([F[te], f_te[te]]))):
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, y[tr])
                dest[te] = logit(clf.predict_proba(Xte)[:, 1])
        sb_l.append(sb); spl_l.append(spl)
    sb, spl = np.mean(sb_l, 0), np.mean(spl_l, 0)

    def gain(base, plus):
        return {"base": {"accuracy": ci(acc, y, base), "auroc": ci(stats.auroc, y, base)},
                "with_probe": {"accuracy": ci(acc, y, plus), "auroc": ci(stats.auroc, y, plus)},
                "gain_accuracy": ci(dacc, y, plus, base), "gain_auroc": ci(dauc, y, plus, base),
                "gain_accuracy_all_items": stats.bootstrap_ci(dacc, (y, plus, base), groups, n_boot=1000)}
    res["incremental_value"] = {
        "method": "nested logistic stacking of out-of-fold scores (5 outer x 4 inner folds, 3 seeds)",
        "over_hand_parse": gain(s_parse, s_parse_probe),
        "over_strongest_text_only": gain(s_text, s_text_probe),
        "gbm_plus_one_column_insensitive": {"gain_accuracy": ci(dacc, y, spl, sb), "gain_auroc": ci(dauc, y, spl, sb),
                                            "note": "third audit: this test returns about zero for a synthetic column worth +2 points"},
        "probe_right_when_hand_parse_wrong": float(np.mean(((sp > 0) == y)[rep & ((s_parse > 0) != y)])),
        "n_hand_parse_wrong": int((rep & ((s_parse > 0) != y)).sum()),
        "pearson_r_probe_vs_hand_parse_score": float(np.corrcoef(sp[rep], s_parse[rep])[0, 1])}
    g = res["incremental_value"]
    log("B': over hand parse %+.3f [%+.3f, %+.3f]; over strongest text %+.3f [%+.3f, %+.3f]; (gbm+column %+.3f)" % (
        g["over_hand_parse"]["gain_accuracy"]["est"], g["over_hand_parse"]["gain_accuracy"]["lo"], g["over_hand_parse"]["gain_accuracy"]["hi"],
        g["over_strongest_text_only"]["gain_accuracy"]["est"], g["over_strongest_text_only"]["gain_accuracy"]["lo"],
        g["over_strongest_text_only"]["gain_accuracy"]["hi"], g["gbm_plus_one_column_insensitive"]["gain_accuracy"]["est"]))

    # ---------------------------------------------------------------- C. properties
    props = {"pair_adjacent_in_skeleton": F[:, 2], "pair_correlated": F[:, 0], "pair_has_common_neighbour": F[:, 8] > 0,
             "common_neighbour_in_a_separating_set": F[:, 9] > 0, "common_neighbour_in_no_separating_set": F[:, 10] > 0,
             "six_variables": F[:, 11] == 6, "many_correlation_statements": F[:, 12] >= np.median(F[:, 12]),
             "any_marginal_independence_statement": F[:, 13] > 0,
             "many_conditional_independence_statements": F[:, 14] >= np.median(F[:, 14])}
    props = {k: np.asarray(v).astype(int) for k, v in props.items()}
    props = {k: v for k, v in props.items() if 0.02 <= v.mean() <= 0.98}
    f0 = seed_folds[0]
    str_id = np.unique(strs, return_inverse=True)[1].astype(np.float32)
    layers = sorted(set(range(0, resid.shape[1], a.layer_stride)) | {L, resid.shape[1] - 1})
    per = Parallel(n_jobs=a.jobs)(delayed(property_layer_job)(resid, rres, l, props, f0, C) for l in layers)
    pe = meta["positions"].index("premise_end") if "premise_end" in meta["positions"] else None
    Xpe = probes.layer_features(resid, L, pe) if pe is not None else None
    res["properties"] = {"layers": layers, "folds": "5 block-wise folds, seed 0", "n_shuffles": 10, "by_property": {}}
    for name, t in props.items():
        li = layers.index(L)
        tr_s, rd_s = per[li][name]
        tr_auc = [stats.auroc(t[rep], per[i][name][0][rep]) for i in range(len(layers))]
        rd_auc = [stats.auroc(t[rep], per[i][name][1][rep]) if per[i][name][1] is not None else None for i in range(len(layers))]
        null = []
        for k in range(10):
            ts = t.copy()
            for tr, te in f0:
                ts[tr] = np.random.default_rng(1000 + k).permutation(t[tr])
            sh = np.zeros(len(t), np.float32)
            for tr, te in f0:
                sh[te] = probes.fit_predict(X[tr], ts[tr], X[te], C=C) if len(np.unique(ts[tr])) > 1 else 0.0
            null.append(stats.auroc(t[rep], sh[rep]))
        text_nb = oof(by_key_gbm(tmpl, np.column_stack([Fcounts, str_id])), f0, t)                          # premise counts + string id, no binding
        text_var = oof(by_key_gbm(tmpl, Fvar), f0, t)                                                        # per-variable counts, no pair lookup
        r = {"frac_positive": float(t.mean()), "majority_rate": float(max(t.mean(), 1 - t.mean())),
             "gold_accuracy_from_property_alone": float(max(stats.accuracy(y[rep], t[rep]), stats.accuracy(y[rep], 1 - t[rep]))),
             "by_layer": {"trained_auroc": tr_auc, "random_init_auroc": rd_auc},
             "at_gold_probe_layer": {
                 "trained": {"auroc": ci(stats.auroc, t, tr_s), "accuracy": ci(acc, t, tr_s)},
                 "shuffled_null_auroc": {"mean": float(np.mean(null)), "sd": float(np.std(null)), "max": float(np.max(null))},
                 "random_init_same_layer_auroc": None if rd_s is None else ci(stats.auroc, t, rd_s),
                 "random_init_best_layer_auroc": None if rd_s is None else float(np.nanmax([x for x in rd_auc if x is not None])),
                 "trained_minus_random_init_auroc": None if rd_s is None else ci(dauc, t, tr_s, rd_s),
                 "text_no_binding_auroc": ci(stats.auroc, t, text_nb), "text_per_variable_counts_auroc": ci(stats.auroc, t, text_var),
                 "trained_minus_text_per_variable_auroc": ci(dauc, t, tr_s, text_var),
                 "trained_at_end_of_premise_auroc": None if Xpe is None else ci(stats.auroc, t, oof(probe_comp(Xpe, C), f0, t))}}
        res["properties"]["by_property"][name] = r
        x = r["at_gold_probe_layer"]
        log(f"C: {name:40s} trained {x['trained']['auroc']['est']:.3f} | null {x['shuffled_null_auroc']['mean']:.3f}±{x['shuffled_null_auroc']['sd']:.3f} | "
            f"rand {x['random_init_same_layer_auroc']['est'] if x['random_init_same_layer_auroc'] else float('nan'):.3f} | text no-binding "
            f"{x['text_no_binding_auroc']['est']:.3f} per-variable {x['text_per_variable_counts_auroc']['est']:.3f} | end of premise "
            f"{x['trained_at_end_of_premise_auroc']['est'] if x['trained_at_end_of_premise_auroc'] else float('nan'):.3f}")

    # ---------------------------------------------------------------- D. strata
    def quint(s):
        return np.digitize(s, np.quantile(s[rep], [0.2, 0.4, 0.6, 0.8]))
    s_ng = ladder["B_binding_text_reference"]["per_string_premise_4grams"]
    strat = {"template": tmpl, "pair_adjacent_in_skeleton": props["pair_adjacent_in_skeleton"], "pair_correlated": props["pair_correlated"],
             "pair_has_common_neighbour": props.get("pair_has_common_neighbour", np.zeros(len(y), int)),
             "template_x_adjacent_x_common_neighbour": np.array([f"{a_}|{b_}|{c_}" for a_, b_, c_ in zip(
                 tmpl, props["pair_adjacent_in_skeleton"], props.get("pair_has_common_neighbour", np.zeros(len(y), int)))]),
             "hand_parse_prediction": (s_parse > 0).astype(int), "hand_parse_score_quintiles": quint(s_parse),
             "parse_quintile_x_4gram_quintile": quint(s_parse) * 5 + quint(s_ng)}
    res["stratified"] = {"overall_probe_auroc": ci(stats.auroc, y, sp), "by_stratifier": {}}
    for name, z in strat.items():
        res["stratified"]["by_stratifier"][name] = {
            "n_strata": int(len(np.unique(z[rep]))),
            "gold_accuracy_from_strata_alone": float(sum(max(y[rep & (z == v)].sum(), (1 - y[rep & (z == v)]).sum()) for v in np.unique(z)) / rep.sum()),
            "probe_auroc_within_strata": stats.bootstrap_ci(weighted_auroc, (y[rep], sp[rep], z[rep]), groups[rep], n_boot=1000)}
        w = res["stratified"]["by_stratifier"][name]
        log(f"D: {name:40s} strata {w['n_strata']:3d} gold-from-strata {w['gold_accuracy_from_strata_alone']:.3f} probe AUROC within {w['probe_auroc_within_strata']['est']:.3f}")

    # ---------------------------------------------------------------- E. the model's answers, by template (all items)
    def resid_by(v, z):
        out = v.astype(np.float64).copy()
        for q in np.unique(z):
            out[z == q] -= out[z == q].mean()
        return out
    ld_r, sp_r = resid_by(ld, strs), resid_by(sp, strs)
    adj = props["pair_adjacent_in_skeleton"]
    boot = lambda fn, *arrs, m: stats.bootstrap_ci(fn, tuple(x[m] for x in arrs), groups[m], n_boot=1000)     # noqa: E731
    allm = np.ones(len(y), bool)
    res["answers"] = {"scope": "all 2,400 items (answers involve no selection)", "pooled_auroc_for_gold": boot(stats.auroc, y, ld, m=allm),
                      "frac_variance_of_answer_logit_explained_by_hypothesis_string": float(1 - ld_r.var() / ld.var()),
                      "within_string_auroc_for_gold": boot(weighted_auroc, y, ld, strs, m=allm),
                      "corr_answer_logit_with_probe_score_string_means_removed": boot(lambda u, v: float(np.corrcoef(u, v)[0, 1]), ld_r, sp_r, m=allm),
                      "by_template": {}}
    for t_ in np.unique(tmpl):
        m = tmpl == t_
        res["answers"]["by_template"][t_] = {
            "n": int(m.sum()), "example": str(strs[m][0]),
            "within_string_auroc_for_gold": boot(weighted_auroc, y, ld, strs, m=m),
            "within_string_auroc_for_gold_report_half": boot(weighted_auroc, y, ld, strs, m=m & rep),
            "within_string_auroc_for_gold_selection_half": boot(weighted_auroc, y, ld, strs, m=m & sel),
            "within_string_auroc_for_adjacency": boot(weighted_auroc, adj, ld, strs, m=m),
            "corr_answer_logit_with_probe_score": boot(lambda u, v: float(np.corrcoef(u, v)[0, 1]), ld_r, sp_r, m=m),
            "probe_within_string_auroc_for_gold": boot(weighted_auroc, y, sp, strs, m=m)}
        w = res["answers"]["by_template"][t_]
        log(f"E: {t_:22s} answer AUROC for gold {w['within_string_auroc_for_gold']['est']:.3f} [{w['within_string_auroc_for_gold']['lo']:.3f},"
            f"{w['within_string_auroc_for_gold']['hi']:.3f}]  r(answer, probe) {w['corr_answer_logit_with_probe_score']['est']:+.3f}  probe AUROC {w['probe_within_string_auroc_for_gold']['est']:.3f}")

    # ---------------------------------------------------------------- F. robustness of the headline number
    null = []
    for k in range(20):
        s = np.zeros(len(y), np.float32)
        for tr, te in f0:
            s[te] = probes.fit_predict(X[tr], np.random.default_rng(2000 + k).permutation(y[tr]), X[te], C=C)
        null.append(acc(y[rep], s[rep]))
    half = probes.fit_predict(X[sel], y[sel], X[rep], C=C)
    res["robustness"] = {"single_seed_accuracy": [acc(y[rep], s[rep]) for s in sp_seeds],
                         "three_seed_ensemble_accuracy": acc(y[rep], sp[rep]),
                         "shuffled_label_null_accuracy": {"n": 20, "mean": float(np.mean(null)), "sd": float(np.std(null)), "max": float(np.max(null))},
                         "trained_on_selection_half_only_accuracy": stats.bootstrap_ci(acc, (y[rep], half), groups[rep], n_boot=1000)}
    res["baselines"]["A0_single_input"]["probe_on_shuffled_labels_null"] = res["robustness"]["shuffled_label_null_accuracy"]
    log("F: single seeds %s; shuffled null %.3f±%.3f; selection-half-only %.3f" % (
        np.round(res["robustness"]["single_seed_accuracy"], 3), np.mean(null), np.std(null), res["robustness"]["trained_on_selection_half_only_accuracy"]["est"]))
    res["minutes"] = round((time.time() - T0) / 60, 1)
    (RESULTS / f"probe_content_{a.model}_{TAG}.json").write_text(json.dumps(res, indent=1))
    log("wrote results")


if __name__ == "__main__":
    main()

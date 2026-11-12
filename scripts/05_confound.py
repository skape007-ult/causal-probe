"""'Represents but doesn't say?' - the gold-vs-model-output confound analysis (CPU, from cache).

Writes results/confound_<model>_<tag>.json.

A probe for the gold label could simply be reading the answer the model is about to give.
So we (1) pick the probe layer on one half of the premises/stories and report on the other
half; (2) look at items the model gets WRONG, where reading the model's answer gives < 50%;
(3) look within each model-answer stratum (model said Yes / model said No), where the
binarised model answer carries no information at all; (4) compare against what is linearly
available from the input text (TF-IDF, type prior) and from a random-init model, and against
the model's own graded confidence (answer logit difference).

The headline claim is licensed only if `verdict.represents_but_doesnt_say` is true.
"""
import argparse
import json

import numpy as np

from causal_probe import cache, probes, stats
from causal_probe.config import MODELS, RESULTS, SEED


def ci(fn, arrays, groups):
    return stats.bootstrap_ci(fn, arrays, groups)


def acc_fn(y, s):
    return stats.accuracy(y, (s > 0).astype(int))


def diff_acc(y, a, b):
    return acc_fn(y, a) - acc_fn(y, b)


def diff_auc(y, a, b):
    return stats.auroc(y, a) - stats.auroc(y, b)


def block(y, scores: dict, groups, ref="gold_probe"):
    """Accuracy and AUROC for every scorer on these items + paired differences vs the probe."""
    out = {"n": int(len(y)), "n_groups": int(len(set(groups))), "frac_gold_yes": float(np.mean(y)) if len(y) else None}
    if len(y) < 20:
        out["skipped"] = "fewer than 20 items"
        return out
    for k, s in scores.items():
        out[k] = {"accuracy": ci(acc_fn, (y, s), groups), "auroc": ci(stats.auroc, (y, s), groups)}
    for k, s in scores.items():
        if k != ref:
            out[f"{ref}_minus_{k}"] = {"accuracy": ci(diff_acc, (y, scores[ref], s), groups),
                                       "auroc": ci(diff_auc, (y, scores[ref], s), groups)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tag", required=True, choices=["c2c_probe", "cladder"])
    a = ap.parse_args()
    d = cache.cache_dir(a.model, a.tag)
    oof = np.load(d / "probe_oof.npz")
    _, meta, items = cache.load_cache(a.model, a.tag)
    y, m = oof["y"], oof["model_pred"]
    ld = np.array(meta["logit_diff"], np.float32)
    groups = np.array([i["group"] for i in items])

    # split-half over groups: select layer on `sel`, report on `rep`
    ug = np.unique(groups)
    rng = np.random.default_rng(SEED + 7)
    sel_groups = set(rng.permutation(ug)[: len(ug) // 2])
    sel = np.array([g in sel_groups for g in groups]); rep = ~sel

    def pick(name):
        S = oof[name]                                           # [seeds, L, n]
        tgt = m if name == "model_out" else y
        acc = ((S[:, :, sel] > 0) == tgt[sel][None, None]).mean((0, 2))
        return int(acc.argmax()), acc.tolist()

    res = {"model": MODELS[a.model], "tag": a.tag, "n": int(len(y)),
           "n_select": int(sel.sum()), "n_report": int(rep.sum()),
           "model_accuracy_report_half": stats.accuracy(y[rep], m[rep]),
           "model_pred_yes_frac": float(m.mean())}
    L_gold, sel_curve = pick("gold")
    res["selected_layer_gold"] = L_gold
    res["selection_half_accuracy_by_layer"] = sel_curve
    scores = {"gold_probe": oof["gold"][:, L_gold].mean(0),
              "tfidf": oof["tfidf"].mean(0), "type_prior": oof["type_prior"].mean(0),
              "model_logit_diff": ld}
    if "random_init" in oof.files:
        L_r, _ = pick("random_init")
        res["selected_layer_random_init"] = L_r
        scores["random_init_probe"] = oof["random_init"][:, L_r].mean(0)
    if "model_out" in oof.files:
        L_m, _ = pick("model_out")
        res["selected_layer_model_out"] = L_m
        mo = oof["model_out"][:, L_m].mean(0)
        res["model_out_probe"] = {
            "accuracy_predicting_model_answer": ci(acc_fn, (m[rep], mo[rep]), groups[rep]),
            "auroc_predicting_model_answer": ci(stats.auroc, (m[rep], mo[rep]), groups[rep]),
            "accuracy_if_scored_against_gold": ci(acc_fn, (y[rep], mo[rep]), groups[rep]),
            "tfidf_accuracy_predicting_model_answer": ci(acc_fn, (m[rep], oof["tfidf_model_out"].mean(0)[rep]), groups[rep]),
        }
        res["gold_probe_scored_against_model_answer"] = ci(acc_fn, (m[rep], scores["gold_probe"][rep]), groups[rep])
        res["gold_vs_model_out_probe_prediction_agreement"] = float(np.mean((scores["gold_probe"][rep] > 0) == (mo[rep] > 0)))
        res["corr_gold_probe_score_vs_model_out_probe_score"] = float(np.corrcoef(scores["gold_probe"][rep], mo[rep])[0, 1])
    res["corr_gold_probe_score_vs_model_logit_diff"] = float(np.corrcoef(scores["gold_probe"][rep], ld[rep])[0, 1])

    def sub(mask):
        mask = mask & rep
        return block(y[mask], {k: s[mask] for k, s in scores.items()}, groups[mask])

    res["all_items"] = sub(np.ones(len(y), bool))
    res["model_wrong"] = sub(m != y)
    res["model_right"] = sub(m == y)
    res["model_said_yes"] = sub(m == 1)     # within stratum: gold-yes = model right, gold-no = model wrong
    res["model_said_no"] = sub(m == 0)

    def lo(path):
        cur = res
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur["lo"]

    w = "model_wrong"
    checks = {
        "probe_beats_chance_on_wrong_items": lo([w, "gold_probe", "accuracy"]),
        "probe_beats_tfidf_on_wrong_items": lo([w, "gold_probe_minus_tfidf", "accuracy"]),
        "probe_beats_type_prior_on_wrong_items": lo([w, "gold_probe_minus_type_prior", "accuracy"]),
    }
    thresholds = {"probe_beats_chance_on_wrong_items": 0.5}
    passed = {k: (v is not None and v > thresholds.get(k, 0.0)) for k, v in checks.items()}
    strat = {}
    for sname in ("model_said_yes", "model_said_no"):
        for base, thr in (("gold_probe", 0.5), ("gold_probe_minus_tfidf", 0.0),
                          ("gold_probe_minus_type_prior", 0.0), ("gold_probe_minus_model_logit_diff", 0.0)):
            v = lo([sname, base, "auroc"])
            strat[f"{sname}/{base}_auroc_ci_lo"] = v
            passed[f"{sname}/{base}"] = v is not None and v > thr
    res["verdict"] = {
        "ci_lower_bounds_wrong_items": checks, "ci_lower_bounds_within_strata": strat, "passed": passed,
        # the brief's criterion: beats chance on wrong items AND beats the text baselines there
        "represents_but_doesnt_say": bool(passed["probe_beats_chance_on_wrong_items"]
                                          and passed["probe_beats_tfidf_on_wrong_items"]
                                          and passed["probe_beats_type_prior_on_wrong_items"]),
        "definition": "95% cluster-bootstrap CI lower bound of gold-probe accuracy on model-wrong items "
                      "(report half, layer chosen on the other half) > 0.5, and CI lower bounds of the paired "
                      "accuracy differences vs TF-IDF and vs the type-prior baseline > 0.",
    }

    # ---- external check for Corr2Cause: train on the whole pool, test on the natural-distribution test set
    if a.tag == "c2c_probe":
        pres, _, _ = cache.load_cache(a.model, "c2c_probe")
        tres, tmeta, titems = cache.load_cache(a.model, "c2c_test")
        ty = np.array(tmeta["labels"]); tld = np.array(tmeta["logit_diff"], np.float32)
        tm = (tld > 0).astype(int); tg = np.array([i["group"] for i in titems])
        s_probe = probes.fit_predict(probes.layer_features(pres, L_gold, 0), y, probes.layer_features(tres, L_gold, 0))
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from causal_probe import prompts
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=2, sublinear_tf=True)
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(vec.fit_transform([prompts.user_text(i) for i in items]), y)
        s_tfidf = clf.decision_function(vec.transform([prompts.user_text(i) for i in titems]))
        tmpl = np.array([i["template"] for i in items]); ttmpl = np.array([i["template"] for i in titems])
        prior = {k: y[tmpl == k].mean() - 0.5 for k in np.unique(tmpl)}
        s_prior = np.array([prior.get(k, 0.0) for k in ttmpl], np.float32)
        tscores = {"gold_probe": s_probe, "tfidf": s_tfidf, "type_prior": s_prior, "model_logit_diff": tld}
        ext = {"note": "probe/TF-IDF trained on the balanced train pool at the selected layer; evaluated on test.csv "
                       "(15.5% positive, 21 premises, 2-6 variables vs 4-6 in the pool: a distribution shift). "
                       "Use AUROC; accuracy at threshold 0 reflects the balanced training prior.",
               "model_accuracy": stats.accuracy(ty, tm), "model_pred_yes_frac": float(tm.mean())}
        for name, mask in (("all_items", np.ones(len(ty), bool)), ("model_wrong", tm != ty), ("model_right", tm == ty),
                           ("model_said_yes", tm == 1), ("model_said_no", tm == 0)):
            ext[name] = block(ty[mask], {k: s[mask] for k, s in tscores.items()}, tg[mask])
        res["external_c2c_test"] = ext

    (RESULTS / f"confound_{a.model}_{a.tag}.json").write_text(json.dumps(res, indent=1))
    mw = res["model_wrong"]
    if "gold_probe" in mw:
        print(f"{a.model}/{a.tag} layer {L_gold}: on model-wrong items (n={mw['n']}): "
              f"probe acc {mw['gold_probe']['accuracy']['est']:.3f} [{mw['gold_probe']['accuracy']['lo']:.3f},{mw['gold_probe']['accuracy']['hi']:.3f}] "
              f"tfidf {mw['tfidf']['accuracy']['est']:.3f} prior {mw['type_prior']['accuracy']['est']:.3f} "
              f"-> represents_but_doesnt_say={res['verdict']['represents_but_doesnt_say']}")
    for sname in ("model_said_yes", "model_said_no"):
        b = res[sname]
        if "gold_probe" in b:
            print(f"   {sname} (n={b['n']}): AUROC probe {b['gold_probe']['auroc']['est']:.3f} "
                  f"tfidf {b['tfidf']['auroc']['est']:.3f} prior {b['type_prior']['auroc']['est']:.3f} "
                  f"model-logit {b['model_logit_diff']['auroc']['est']:.3f}")


if __name__ == "__main__":
    main()

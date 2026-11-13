"""Tier 2c: cross-model comparison, assembled from existing results (no new computation
except bootstrap CIs on already-saved out-of-fold probe scores). Writes results/cross_model.json.

Question: do decodability and the depth at which it peaks scale with model size?
"""
import json

import numpy as np

from causal_probe import cache, stats
from causal_probe.config import MODELS, RESULTS, SEED

PARAMS_B = {"qwen0.5b": 0.494, "qwen1.5b": 1.54, "gemma2b": 2.61}   # parameter counts (billions), from model cards


def main():
    out = {"models": {}, "note": "selected layer = argmax of gold-probe accuracy on the selection half of "
                                 "premises/stories; all metrics are on the report half (see 05_confound.py)."}
    for m in MODELS:
        row = {"params_billion": PARAMS_B[m], "datasets": {}}
        for tag in ("c2c_probe", "cladder"):
            pc = RESULTS / f"confound_{m}_{tag}.json"
            if not pc.exists():
                continue
            c = json.loads(pc.read_text())
            p = json.loads((RESULTS / f"probes_{m}_{tag}_C{c['selected_C_gold']:g}.json").read_text())
            L = p["n_layers_plus_embed"] - 1
            gold = np.array([l["accuracy"]["est"] for l in p["layers"]["gold"]])
            ctrl = np.array([l["accuracy"]["est"] for l in p["layers"]["control"]])
            a = c["all_items"]
            # first layer whose gold accuracy is within 1 point of the maximum ("how early is it there?")
            first_near_max = int(np.argmax(gold >= gold.max() - 0.01))
            # paired CI: probe minus random-init probe (report half)
            d = {"n_layers": L, "selected_layer": c["selected_layer_gold"], "selected_C": c["selected_C_gold"],
                 "selected_layer_relative_depth": c["selected_layer_gold"] / L,
                 "first_layer_within_1pt_of_max": first_near_max,
                 "first_layer_within_1pt_relative_depth": first_near_max / L,
                 "model_accuracy": c["model_accuracy_report_half"],
                 "gold_probe_accuracy": a["gold_probe"]["accuracy"], "gold_probe_auroc": a["gold_probe"]["auroc"],
                 "tfidf_accuracy": a["tfidf"]["accuracy"], "type_prior_accuracy": a["type_prior"]["accuracy"],
                 "probe_minus_tfidf_accuracy": a["gold_probe_minus_tfidf"]["accuracy"],
                 "probe_minus_tfidf_auroc": a["gold_probe_minus_tfidf"]["auroc"],
                 "probe_minus_type_prior_accuracy": a["gold_probe_minus_type_prior"]["accuracy"],
                 "selectivity_at_selected_layer": float(gold[c["selected_layer_gold"]] - ctrl[c["selected_layer_gold"]]),
                 "max_selectivity": float((gold - ctrl).max()),
                 "represents_but_doesnt_say": c["verdict"]["represents_but_doesnt_say"],
                 "represents_but_doesnt_say_calibrated": c["verdict"]["calibrated"]["represents_but_doesnt_say"]}
            if "random_init_probe" in a:
                d["random_init_accuracy"] = a["random_init_probe"]["accuracy"]
                d["probe_minus_random_init_accuracy"] = a["gold_probe_minus_random_init_probe"]["accuracy"]
            row["datasets"][tag] = d
        if row["datasets"]:
            out["models"][m] = row
    # pairwise model differences in gold-probe accuracy (paired over items, cluster bootstrap), all-layer max excluded:
    # uses each model's split-half-selected layer on the common report half.
    out["pairwise"] = {}
    ms = list(out["models"])
    for tag in ("c2c_probe", "cladder"):
        sc = {}
        for m in ms:
            if tag not in out["models"][m]["datasets"]:
                continue
            oof = np.load(cache.cache_dir(m, tag) / f"probe_oof_C{out['models'][m]['datasets'][tag]['selected_C']:g}.npz")
            _, meta, items = cache.load_cache(m, tag)
            groups = np.array([i["group"] for i in items])
            ug = np.unique(groups)
            rng = np.random.default_rng(SEED + 7)
            selg = set(rng.permutation(ug)[: len(ug) // 2])
            rep = np.array([g not in selg for g in groups])
            sc[m] = (oof["y"][rep], oof["gold"][:, out["models"][m]["datasets"][tag]["selected_layer"]].mean(0)[rep], groups[rep])
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if ms[i] in sc and ms[j] in sc:
                    y, a_, g = sc[ms[i]]; _, b_, _ = sc[ms[j]]
                    out["pairwise"][f"{tag}:{ms[j]}_minus_{ms[i]}"] = stats.bootstrap_ci(
                        lambda yy, s1, s2: stats.accuracy(yy, (s2 > 0).astype(int)) - stats.accuracy(yy, (s1 > 0).astype(int)),
                        (y, a_, b_), g)
    (RESULTS / "cross_model.json").write_text(json.dumps(out, indent=1))
    for m, row in out["models"].items():
        for tag, d in row["datasets"].items():
            print(f"{m:9s} {tag:10s} probe {d['gold_probe_accuracy']['est']:.3f} tfidf {d['tfidf_accuracy']['est']:.3f} "
                  f"Δ {d['probe_minus_tfidf_accuracy']['est']:+.3f} [{d['probe_minus_tfidf_accuracy']['lo']:+.3f},{d['probe_minus_tfidf_accuracy']['hi']:+.3f}] "
                  f"layer {d['selected_layer']}/{d['n_layers']} ({d['selected_layer_relative_depth']:.2f}) sel {d['selectivity_at_selected_layer']:+.3f}")
    for k, v in out["pairwise"].items():
        print(k, f"{v['est']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")


if __name__ == "__main__":
    main()

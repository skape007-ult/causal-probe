"""Sensitivity of the gold probe to its one hyperparameter (L2 strength C), from the cache.
Writes results/probe_sensitivity_<model>.json. Reported, never used to pick C (decision D7).
Also reports a difference-of-means "probe" (no fitting beyond two class means) as a
capacity-free reference."""
import argparse
import json

import numpy as np
from joblib import Parallel, delayed

from causal_probe import cache, probes, stats
from causal_probe.config import ALL_MODELS, MODELS, RESULTS

CS = [0.0001, 0.001, 0.01, 0.1, 1.0]


def one(X, y, folds, C):
    return stats.accuracy(y, (probes.oof_scores(X, y, folds, C=C) > 0).astype(int))


def diffmeans_acc(X, y, folds):
    s = np.zeros(len(y))
    for tr, te in folds:
        m1, m0 = X[tr][y[tr] == 1].mean(0), X[tr][y[tr] == 0].mean(0)
        w = m1 - m0
        s[te] = (X[te] - (m1 + m0) / 2) @ w
    return stats.accuracy(y, (s > 0).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    a = ap.parse_args()
    res = {"model": a.model, "Cs": CS, "default_C": probes.PROBE_C}
    for tag in ("c2c_probe", "cladder"):
        resid, meta, items = cache.load_cache(a.model, tag)
        y = np.array(meta["labels"])
        folds = probes.folds_for(items, 0)
        L = resid.shape[1]
        layers = sorted({int(round(x)) for x in np.linspace(1, L - 1, 6)})
        Xs = {l: probes.layer_features(resid, l, 0) for l in layers}
        accs = Parallel(n_jobs=4)(delayed(one)(Xs[l], y, folds, C) for l in layers for C in CS)
        grid = np.array(accs).reshape(len(layers), len(CS))
        res[tag] = {"layers": layers, "accuracy_layer_by_C": grid.tolist(),
                    "diff_of_means_accuracy": [diffmeans_acc(Xs[l], y, folds) for l in layers],
                    "max_spread_across_C_at_any_layer": float((grid.max(1) - grid.min(1)).max()),
                    "best_C_per_layer": [CS[i] for i in grid.argmax(1)]}
        print(a.model, tag, "layers", layers)
        for l, row, dm in zip(layers, grid, res[tag]["diff_of_means_accuracy"]):
            print(f"  L{l:2d} " + " ".join(f"{v:.3f}" for v in row) + f" | diff-of-means {dm:.3f}")
    (RESULTS / f"probe_sensitivity_{a.model}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

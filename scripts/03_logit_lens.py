"""Logit lens from the cache (CPU). Writes results/lens_<model>.json.

For every layer, apply the final norm + unembedding to the residual stream at the final
prompt position and take logit(Yes) - logit(No). Reported as the *gold-signed* difference
(positive = leaning towards the correct answer), split by whether the model's final answer
is correct, and additionally split by gold label (because a constant Yes/No bias would
otherwise masquerade as 'knowing' the answer on one class).
"""
import argparse
import json

import numpy as np

from causal_probe import cache, lens, stats
from causal_probe.config import ALL_MODELS, MODELS, RESULTS


def curve(v, groups):
    """v [n, L] -> per-layer mean with cluster-bootstrap CI."""
    if len(v) == 0:
        return None
    L = v.shape[1]
    rng = np.random.default_rng(0)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(ug), L)); np.add.at(sums, inv, v)
    cnt = np.bincount(inv, minlength=len(ug))
    boots = []
    for _ in range(1000):
        k = rng.integers(0, len(ug), len(ug))
        boots.append(sums[k].sum(0) / cnt[k].sum())
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    return {"mean": v.mean(0).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    a = ap.parse_args()
    res = {"model": a.model}
    for tag in ["c2c_test", "c2c_probe", "cladder"]:
        resid, meta, items = cache.load_cache(a.model, tag)
        ro = lens.load_readout(cache.cache_dir(a.model, tag) / "readout.npz")
        ld = lens.lens_by_layer(resid[:, :, 0, :], ro)            # [n, L+1]
        y = np.array(meta["labels"]); final = np.array(meta["logit_diff"])
        groups = np.array([i["group"] for i in items])
        pred = (final > 0).astype(int); correct = pred == y
        signed = ld * np.where(y == 1, 1, -1)[:, None]
        r = {"n": len(y), "n_correct": int(correct.sum()), "n_incorrect": int((~correct).sum()),
             # sanity: the lens at the last layer must reproduce the model's real answer logits
             "last_layer_vs_model_max_abs_err": float(np.abs(ld[:, -1] - final).max()),
             "last_layer_vs_model_sign_agree": float(np.mean((ld[:, -1] > 0) == (final > 0))),
             "raw_logit_diff_all": curve(ld, groups),
             "gold_signed": {"correct": curve(signed[correct], groups[correct]),
                             "incorrect": curve(signed[~correct], groups[~correct])},
             "raw_by_gold_and_correct": {
                 f"gold={'yes' if g else 'no'}/{'correct' if c else 'incorrect'}":
                     curve(ld[(y == g) & (correct == c)], groups[(y == g) & (correct == c)])
                 for g in (1, 0) for c in (True, False)},
             # threshold-free: does the lens logit diff at each layer rank gold-yes above gold-no?
             "auroc_gold_by_layer": [stats.auroc(y, ld[:, l]) for l in range(ld.shape[1])],
             "lens_accuracy_by_layer": [stats.accuracy(y, (ld[:, l] > 0).astype(int)) for l in range(ld.shape[1])],
             }
        res[tag] = r
        print(f"{a.model}/{tag}: lens-vs-model max err {r['last_layer_vs_model_max_abs_err']:.3f}, "
              f"sign agree {r['last_layer_vs_model_sign_agree']:.3f}, "
              f"best-layer lens AUROC {np.nanmax(r['auroc_gold_by_layer']):.3f} @ {int(np.nanargmax(r['auroc_gold_by_layer']))}")
    (RESULTS / f"lens_{a.model}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

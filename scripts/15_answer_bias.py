"""Answer bias versus chance: is a model 'at chance' because its answers carry no information, or
only because it leans towards one answer? (CPU, from the cached per-item answer logits.)

For every cached model, item set and prompting mode:
  yes_rate               share of items answered Yes at the natural threshold (logit difference > 0)
  accuracy               accuracy at that threshold
  auroc                  does the Yes-No logit difference rank valid above invalid items? Independent
                         of any threshold, so a constant bias towards Yes or No does not affect it.
  best_threshold_acc     accuracy when the threshold is moved to wherever it helps most. Chosen on one
                         half of the premises/stories and scored on the other half, both ways
                         (2-fold cross-fitting), so it is not flattered by tuning on the test items.
  best_threshold_acc_in_sample   the same with the threshold tuned on the scored items (upper bound).
  majority               accuracy of always giving the more frequent label.
Reading: auroc near 0.5 and best_threshold_acc near majority = no usable information ("at chance").
auroc clearly above 0.5 with accuracy at or below majority = information is there but hidden by bias.

Writes results/answer_bias.json.
"""
import json

import numpy as np

from causal_probe import cache, data, stats
from causal_probe.config import ALL_MODELS, RESULTS, SEED

TAGS = [("c2c_test", "zero-shot"), ("c2c_probe", "zero-shot"), ("c2c_debiased", "zero-shot"), ("cladder", "zero-shot"),
        ("c2c_test_renamed", "zero-shot, variables renamed"),
        ("c2c_test_sub_fewshot", "4-shot"), ("cladder_sub_fewshot", "4-shot")]


def best_thr(y, s):
    """Threshold maximising accuracy of (s > thr) on these items."""
    o = np.argsort(s); ys, ss = y[o], s[o]
    # predict No for the k lowest scores: correct = (# gold-No among them) + (# gold-Yes among the rest)
    correct = np.concatenate([[ys.sum()], np.cumsum(1 - ys) + (ys.sum() - np.cumsum(ys))])
    k = int(correct.argmax())
    if k == 0:
        return -np.inf
    return float(ss[k - 1]) if k == len(ss) else float((ss[k - 1] + ss[k]) / 2)


def crossfit_acc(y, s, g):
    ug = np.unique(g)
    half = set(np.random.default_rng(SEED + 3).permutation(ug)[: len(ug) // 2])
    a = np.array([x in half for x in g])
    pred = np.zeros(len(y), int)
    for tr, te in ((a, ~a), (~a, a)):
        pred[te] = (s[te] > best_thr(y[tr], s[tr])).astype(int)
    return pred


def main():
    out = {}
    for m in ALL_MODELS:
        for tag, mode in TAGS:
            d = cache.cache_dir(m, tag)
            if not (d / "meta.json").exists():
                continue
            meta = json.loads((d / "meta.json").read_text())
            items = cache.items_for(tag, meta)
            y = np.array(meta["labels"]); s = np.array(meta["logit_diff"]); g = np.array([i["group"] for i in items])
            cf = crossfit_acc(y, s, g)
            acc = lambda a_, b_: stats.accuracy(a_, b_)        # noqa: E731
            r = {"mode": mode, "n": int(len(y)), "n_clusters": int(len(set(g))), "frac_gold_yes": float(y.mean()),
                 "majority": float(max(y.mean(), 1 - y.mean())),
                 "yes_rate": float((s > 0).mean()), "mean_logit_diff": float(s.mean()),
                 "accuracy": stats.bootstrap_ci(lambda a_, b_: acc(a_, (b_ > 0).astype(int)), (y, s), g),
                 "macro_f1": stats.bootstrap_ci(lambda a_, b_: stats.macro_f1(a_, (b_ > 0).astype(int)), (y, s), g),
                 "auroc": stats.bootstrap_ci(stats.auroc, (y, s), g),
                 "best_threshold_acc": stats.bootstrap_ci(acc, (y, cf), g),
                 "best_threshold_acc_in_sample": float(acc(y, (s > best_thr(y, s)).astype(int))),
                 "best_threshold_in_sample": best_thr(y, s)}
            lo = r["auroc"]["lo"]
            r["reading"] = ("informative (AUROC CI above 0.5)" if lo > 0.5 else
                            "below chance (AUROC CI below 0.5)" if r["auroc"]["hi"] < 0.5 else "at chance (AUROC CI includes 0.5)")
            out.setdefault(m, {})[tag] = r
            print(f"{m:9s} {tag:22s} yes {r['yes_rate']:.2f} acc {r['accuracy']['est']:.3f} (maj {r['majority']:.3f}) "
                  f"auroc {r['auroc']['est']:.3f} [{r['auroc']['lo']:.3f},{r['auroc']['hi']:.3f}] "
                  f"best-thr acc {r['best_threshold_acc']['est']:.3f} -> {r['reading']}")
    (RESULTS / "answer_bias.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

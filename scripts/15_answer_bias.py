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

Debiased pool only, added after the eleventh audit (notes/audit_runs_b_d_raw.md, F1): the same question WITHIN hypothesis
string. On that pool the answer logit differs between hypothesis strings by far more than within them, and the label is
balanced inside every string, so the pooled AUROC is dominated by differences that carry no label information and can sit
at 0.5 while the answers track the label inside every string (or track it with opposite signs in different hypothesis
types, which cancel). Reported per model under "within_hypothesis_string":
  mean_auroc                     mean over the hypothesis strings of the AUROC within the string
  by_type                        the same per hypothesis type
  within_string_and_adjacency    AUROC within (string, queried pair adjacent or not) cells, weighted by the number of
                                 valid-invalid pairs: does the answer follow the label beyond following adjacency?
Intervals: premises resampled. Permutation p (two-sided): the two labels of a premise swapped at random (each premise has
one valid and one invalid item). These statistics were chosen after the pooled ones had been seen.

Writes results/answer_bias.json.
"""
import json

import numpy as np

from scipy.stats import rankdata

from causal_probe import cache, data, probes, stats
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


def _cell_aurocs(y, s, cells):
    """AUROC of s for y inside each cell (list of index arrays); (aurocs, weights = valid x invalid pairs). One-class cells are skipped."""
    au, w = [], []
    for idx in cells:
        yy = y[idx]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(s[idx])
        au.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)); w.append(n1 * n0)
    return np.array(au), np.array(w, float)


def within_string(items, y, s, n_boot=1000, n_perm=2000):
    strs = np.array([i["query"] for i in items]); grp = np.array([i["group"] for i in items]); tm = np.array([i["template"] for i in items])
    adj = probes.structural_features(items)[:, 2].astype(int)
    cellkey = np.array([f"{a_}|{b_}" for a_, b_ in zip(strs, adj)])
    types = sorted(set(tm))
    order = np.argsort(grp, kind="stable").reshape(-1, 2)                     # the two items of each premise
    assert (grp[order[:, 0]] == grp[order[:, 1]]).all() and (y[order[:, 0]] != y[order[:, 1]]).all()

    def stat(yy, ss, keys_s, keys_c, tmv):
        res = {}
        us = np.unique(keys_s)
        cells_s = [np.where(keys_s == u)[0] for u in us]
        a, _ = _cell_aurocs(yy, ss, cells_s)
        res["all"] = a.mean()
        for t in types:
            a_t, _ = _cell_aurocs(yy, ss, [c for c in cells_s if tmv[c[0]] == t])
            res[t] = a_t.mean() if len(a_t) else np.nan
        cells_c = [np.where(keys_c == u)[0] for u in np.unique(keys_c)]
        a, w = _cell_aurocs(yy, ss, cells_c)
        res["cell"] = np.average(a, weights=w)
        for t in types:
            a_t, w_t = _cell_aurocs(yy, ss, [c for c in cells_c if tmv[c[0]] == t])
            res["cell_" + t] = np.average(a_t, weights=w_t) if len(a_t) else np.nan
        return res

    obs = stat(y, s, strs, cellkey, tm)
    rng = np.random.default_rng(SEED + 15)
    boot = []
    for _ in range(n_boot):
        ix = order[rng.integers(0, len(order), len(order))].ravel()
        boot.append(stat(y[ix], s[ix], strs[ix], cellkey[ix], tm[ix]))
    null = []
    for _ in range(n_perm):
        yp = y.copy(); fl = order[rng.random(len(order)) < 0.5]
        yp[fl[:, 0]], yp[fl[:, 1]] = y[fl[:, 1]], y[fl[:, 0]]
        null.append(stat(yp, s, strs, cellkey, tm))

    def pack(k):
        b = np.array([x[k] for x in boot], float); n = np.array([x[k] for x in null], float)
        return {"est": float(obs[k]), "lo": float(np.nanpercentile(b, 2.5)), "hi": float(np.nanpercentile(b, 97.5)),
                "permutation_p_two_sided": float((1 + (np.abs(n - 0.5) >= abs(obs[k] - 0.5) - 1e-12).sum()) / (n_perm + 1)),
                "null_mean": float(np.nanmean(n)), "null_sd": float(np.nanstd(n))}
    a_all, _ = _cell_aurocs(y, s, [np.where(strs == u)[0] for u in np.unique(strs)])
    m_ = pack("all")
    return {"mean_auroc": m_, "n_strings": int(len(a_all)), "n_strings_with_auroc_above_half": int((a_all > 0.5).sum()),
            "by_type": {t: {**pack(t), "n_strings": int(len(set(strs[tm == t])))} for t in types},
            "within_string_and_adjacency": {**pack("cell"), "by_type": {t: pack("cell_" + t) for t in types},
                                            "share_of_items_in_cells_with_both_labels": float(np.mean([
                                                0 < y[cellkey == c].mean() < 1 for c in cellkey]))},
            "valid_item_above_invalid_item_of_the_same_premise": float(np.mean(np.where(y[order[:, 0]] == 1, s[order[:, 0]] > s[order[:, 1]], s[order[:, 1]] > s[order[:, 0]]))),
            "reading": ("above chance within hypothesis string" if m_["lo"] > 0.5 else "below chance within hypothesis string" if m_["hi"] < 0.5
                        else "at chance within hypothesis string"),
            "note": "chosen after the pooled statistics had been seen (eleventh audit, F1); n_boot %d premise resamples, %d permutations" % (n_boot, n_perm)}


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
            if tag == "c2c_debiased":
                r["within_hypothesis_string"] = w = within_string(items, y, s)
                c_ = w["within_string_and_adjacency"]
                print(f"{m:9s} {tag:22s} within string: mean AUROC {w['mean_auroc']['est']:.3f} [{w['mean_auroc']['lo']:.3f},{w['mean_auroc']['hi']:.3f}] "
                      f"p {w['mean_auroc']['permutation_p_two_sided']:.4f}; within string and adjacency {c_['est']:.3f} [{c_['lo']:.3f},{c_['hi']:.3f}] "
                      f"p {c_['permutation_p_two_sided']:.4f} -> {w['reading']}")
            out.setdefault(m, {})[tag] = r
            print(f"{m:9s} {tag:22s} yes {r['yes_rate']:.2f} acc {r['accuracy']['est']:.3f} (maj {r['majority']:.3f}) "
                  f"auroc {r['auroc']['est']:.3f} [{r['auroc']['lo']:.3f},{r['auroc']['hi']:.3f}] "
                  f"best-thr acc {r['best_threshold_acc']['est']:.3f} -> {r['reading']}")
    (RESULTS / "answer_bias.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

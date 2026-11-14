"""How many gold-label steering comparisons were made, and how many top ranks does chance give? CPU only.

Added after the eighth audit (notes/audit_round2_runs_raw.md, F3 and Q1 b, c). Every row of the steering-controls
table compares one real direction with 20 matched controls. Under "the real direction is one more draw from its control
family" its rank among the 21 is uniform, so 1 in 21 comparisons shows "larger than all 20 controls" by chance. This
script lists every such comparison for the gold-label directions from the raw per-item outputs
(cache/<model>/<tag>/steering_controls[_hyp].npz), with the signed and absolute rank and a z value against the
controls, and counts the top ranks against that expectation, per item set (the controls are shuffled within hypothesis
string on the debiased pool only).

Added after the eleventh audit (notes/audit_runs_b_d_raw.md, F3 and F4): (1) whether the site responds at all, i.e. whether
the positive control (the model-answer direction) moves the answer more than all 20 shuffled-label probe directions there;
a comparison at a site where it does not cannot show anything, and the counts are also given without such sites;
(2) the mean rank of the gold directions among their controls, which uses all comparisons instead of the first ranks only
(null: every rank from 1 to 21 equally likely, mean 11; the probability is from 200,000 simulated sets of independent ranks).

Writes results/steering_controls_multiplicity.json.
"""
import json
from math import comb

import numpy as np

from causal_probe import cache
from causal_probe.config import MODELS, REMOTE_MODELS, RESULTS

MATCH = {"gold_probe": "shuffled_probe", "gold_diffmeans": "shuffled_diffmeans"}


def p_at_least(k, n, p):
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def main():
    arms = []
    for m in {**MODELS, **REMOTE_MODELS}:
        for tag in ("c2c_debiased", "c2c_probe"):
            for hyp in ("", "_hyp"):
                f = cache.cache_dir(m, tag) / f"steering_controls{hyp}.npz"
                j = RESULTS / f"steering_controls_{m}_{tag}{hyp}.json"
                if not (f.exists() and j.exists()):
                    continue
                z = np.load(f, allow_pickle=True); meta = json.loads(j.read_text())
                fam = np.array([str(x) for x in z["families"]]); out = z["out"].astype(np.float64); k = len(fam)
                slopes = ((out[:, 1:1 + k] - out[:, 1 + k:]) / (2 * float(z["alpha"]))).mean(0)
                own = z["own_sd_over_s"]
                pc = float(slopes[fam == "model_out"][0]); pcc = slopes[fam == "shuffled_probe"]
                pc_rank = int(1 + (np.abs(pcc) > abs(pc)).sum())
                iso_med = float(np.median(np.abs(slopes[fam == "isotropic"])))
                for d, cf in MATCH.items():
                    r = float(slopes[fam == d][0]); c = slopes[fam == cf]
                    assert abs(r - meta["real"][d]["slope_per_s"]["est"]) < 1e-6, (m, tag, hyp, d)       # the raw file is the one the JSON came from
                    arms.append({"model": m, "tag": tag, "token": "end of hypothesis" if hyp else "final", "layer": int(z["layer"]), "direction": d,
                                 "control_labels_shuffled": meta.get("control_labels_shuffled", "over the whole item set"),
                                 "positive_control_slope_per_s": pc, "positive_control_abs_rank_among_shuffled_probes": pc_rank,
                                 "site_responds_positive_control_beyond_all_controls": pc_rank == 1, "isotropic_median_abs_slope": iso_med,
                                 "slope_per_s": r, "n_controls": int(len(c)), "control_mean": float(c.mean()), "control_sd": float(c.std(ddof=1)),
                                 "z_vs_controls": float((r - c.mean()) / c.std(ddof=1)),
                                 "abs_rank_1_is_largest": int(1 + (np.abs(c) > abs(r)).sum()),
                                 "signed_rank_1_is_most_positive": int(1 + (c > r).sum()),
                                 "abs_rank_per_natural_sd": int(1 + (np.abs(c * own[fam == cf]) > abs(r * own[fam == d][0])).sum()),
                                 "natural_sd_over_s": float(own[fam == d][0]), "natural_sd_controls_median": float(np.median(own[fam == cf]))})
    by_arm = {f"{a['model']}.{a['tag']}.{'hyp' if a['token'] != 'final' else 'final'}.{a['direction']}": a for a in arms}
    res = {"arms": arms, "by_arm": by_arm, "note": "rank among the real direction and its 20 controls; under no difference from the control family every rank has probability 1/21. "
                                 "The two tokens of one model share the steered items' pool and the shuffled labellings' generator, so arms are not all independent; "
                                 "the binomial probabilities are a guide, not an exact test.", "summary": {}}
    rng = np.random.default_rng(0)

    def mean_rank(A):
        if not A:
            return None
        obs = float(np.mean([a["abs_rank_1_is_largest"] for a in A]))
        sim = rng.integers(1, 22, size=(200_000, len(A))).mean(1)
        return {"n": len(A), "mean_abs_rank": obs, "null_mean": 11.0, "p_mean_rank_this_small_or_smaller": float((1 + (sim <= obs + 1e-12).sum()) / (len(sim) + 1))}

    alive = lambda a: a["site_responds_positive_control_beyond_all_controls"]      # noqa: E731
    for name, sel in (("all", lambda a: True), ("debiased_pool", lambda a: a["tag"] == "c2c_debiased"), ("original_pool", lambda a: a["tag"] == "c2c_probe"),
                      ("responding_sites", alive), ("responding_sites_debiased_pool", lambda a: alive(a) and a["tag"] == "c2c_debiased")):
        A = [a for a in arms if sel(a)]
        n = len(A); top = [a for a in A if a["abs_rank_1_is_largest"] == 1]
        res["summary"][name] = {
            "n_comparisons": n, "n_rank_1_in_absolute_slope": len(top), "expected_rank_1_by_chance": n / 21,
            "p_at_least_this_many_rank_1": p_at_least(len(top), n, 1 / 21), "p_at_least_one_rank_1": p_at_least(1, n, 1 / 21),
            "single_arm_p": 1 / 21, "bonferroni_p_for_one_rank_1_arm": min(1.0, n / 21),
            "rank_1_arms": [f"{a['model']} {a['tag']} {a['token']} {a['direction']} ({a['slope_per_s']:+.3f})" for a in top],
            "gold_diffmeans_slopes_positive": f"{sum(a['slope_per_s'] > 0 for a in A if a['direction'] == 'gold_diffmeans')}/{sum(a['direction'] == 'gold_diffmeans' for a in A)}",
            "gold_probe_slopes_negative": f"{sum(a['slope_per_s'] < 0 for a in A if a['direction'] == 'gold_probe')}/{sum(a['direction'] == 'gold_probe' for a in A)}",
            "sites_not_responding": sorted({f"{a['model']} {a['tag']} {a['token']} (layer {a['layer']}; positive control rank {a['positive_control_abs_rank_among_shuffled_probes']}/21)"
                                            for a in A if not alive(a)}),
            "mean_rank": {"both_directions": mean_rank(A), "gold_diffmeans": mean_rank([a for a in A if a["direction"] == "gold_diffmeans"]),
                          "gold_probe": mean_rank([a for a in A if a["direction"] == "gold_probe"])},
            "gold_diffmeans_largest_signed_rank": max((a["signed_rank_1_is_most_positive"] for a in A if a["direction"] == "gold_diffmeans"), default=None)}
    (RESULTS / "steering_controls_multiplicity.json").write_text(json.dumps(res, indent=1))
    for name, s in res["summary"].items():
        print(f"{name:14s} {s['n_comparisons']:2d} comparisons, {s['n_rank_1_in_absolute_slope']} at rank 1 (expected {s['expected_rank_1_by_chance']:.2f}; "
              f"P(at least this many) {s['p_at_least_this_many_rank_1']:.3f}); gold difference of means positive in {s['gold_diffmeans_slopes_positive']}, "
              f"gold probe negative in {s['gold_probe_slopes_negative']}; mean |rank| difference of means {s['mean_rank']['gold_diffmeans']['mean_abs_rank']:.1f} "
              f"(P {s['mean_rank']['gold_diffmeans']['p_mean_rank_this_small_or_smaller']:.4f}), probe {s['mean_rank']['gold_probe']['mean_abs_rank']:.1f} "
              f"(P {s['mean_rank']['gold_probe']['p_mean_rank_this_small_or_smaller']:.4f}); not responding: {s['sites_not_responding']}")
    for a in arms:
        if a["tag"] == "c2c_debiased":
            print(f"  {a['model']:9s} {a['token']:17s} {a['direction']:15s} slope {a['slope_per_s']:+.3f}  z {a['z_vs_controls']:+.2f}  |rank| {a['abs_rank_1_is_largest']:2d}/21  signed rank {a['signed_rank_1_is_most_positive']:2d}/21")


if __name__ == "__main__":
    main()

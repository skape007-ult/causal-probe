"""Re-analysis of the raw steering outputs, written after the Tier 2 audit (notes/audit_tier2_raw.md).
CPU only: reads cache/<model>/<tag>/steering<suffix>.npz and the cached activations.

Why. The first summary (07_steering.py) reported flip rates for the + sign only, "accuracy toward
gold" and a mean over three random directions. The audit showed these hide a sign-consistent
effect of the gold-probe direction in one model, and that the step sizes are hundreds of times the
natural spread along that direction. This script reports, for every steered direction:

  slope            mean over items of (LD(+a) - LD(-a)) / (2a), in logits per unit of s, at each a,
                   with a cluster-bootstrap CI, and the share of items with a positive slope.
                   LD = Yes-No answer logit difference; s = RMS distance of an activation from the
                   mean activation (a norm over all dimensions, NOT a spread along one direction).
  natural scale    SD of the activations along the direction, and the gap between the gold classes
                   along it, both in units of s (selection half). `alpha_1_in_own_sd` says how many
                   natural SDs one unit of alpha is. `effect_of_natural_class_gap` = slope x gap:
                   the change in LD if an item were moved from one class mean to the other.
  information      AUROC of the projection for the gold label and for the model's own (median-split)
                   answer, on the report half; cosine between the direction fitted on the selection
                   half and the same kind of direction fitted on the report half.
  flips            flip rate for +, for - and for the better sign, each with a CI.
  controllability  share of items whose answer equals the gold label when pushed "towards gold".
                   NOT a measure of knowledge: any direction with a consistent slope reaches 1 or 0.
and paired comparisons of |slope| between directions, and for the ablations the observed effect
next to a noise floor (below which nothing can be said).

Writes results/steering_analysis_<model>_<tag><suffix>.json.
"""
import argparse
import json

import numpy as np

from causal_probe import cache, data, probes, stats, steer
from causal_probe.config import ALL_MODELS, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", required=True, choices=["c2c_probe", "cladder", "c2c_debiased"])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    d = cache.cache_dir(a.model, a.tag)
    z = np.load(d / f"steering{a.suffix}.npz", allow_pickle=True)
    sj = json.loads((RESULTS / f"steering_{a.model}_{a.tag}{a.suffix}.json").read_text())
    idx, out, names, alphas = z["idx"], z["out"].astype(np.float64), [str(n) for n in z["names"]], [float(x) for x in z["alphas"]]
    layer, s_unit, C = int(z["layer"]), float(z["s_unit"]), sj["probe_C"]
    resid, meta, items = cache.load_cache(a.model, a.tag)
    X = probes.layer_features(resid, layer, 0).astype(np.float64)
    y = np.array(meta["labels"]); ld = np.array(meta["logit_diff"])
    groups = np.array([i["group"] for i in items])
    sel = data.selection_mask(items); rep = ~sel
    mcal = (ld > np.median(ld)).astype(int)
    assert rep[idx].all(), "evaluation items must be in the report half"
    nd, na = len(names), len(alphas)
    base = out[:, 0]
    A = out[:, 1:1 + nd * na * 2].reshape(len(idx), nd, na, 2)       # [..., 0] = +, [..., 1] = -
    ABL = out[:, 1 + nd * na * 2:]
    yy, gg = y[idx], groups[idx]

    def ci(fn, *arrs):
        return stats.bootstrap_ci(fn, arrs, gg, n_boot=1000)

    # the directions exactly as 07_steering.py fitted them (selection half)
    mu = X[sel].mean(0)
    dirs = {"gold_probe": steer.probe_direction(X[sel], y[sel], C), "gold_diffmeans": steer.diff_of_means(X[sel], y[sel]),
            "model_out": steer.probe_direction(X[sel], mcal[sel], C)}
    dirs_rep = {"gold_probe": steer.probe_direction(X[rep], y[rep], C), "gold_diffmeans": steer.diff_of_means(X[rep], y[rep]),
                "model_out": steer.probe_direction(X[rep], mcal[rep], C)}
    r2 = np.random.default_rng(0 + 11)
    for k in range(nd - 3):
        dirs[f"random_{k}"] = steer.unit(r2.standard_normal(X.shape[1]))
    s_check = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
    Xc = X[sel] - mu
    # top principal directions of the selection-half activations
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    top10 = Vt[:10]

    res = {"model": a.model, "tag": a.tag, "layer": layer, "n_layers": int(resid.shape[1] - 1), "probe_C": C,
           "n_items": int(len(idx)), "n_clusters": int(len(set(gg))), "s_unit": s_unit,
           "s_unit_recomputed": s_check, "mean_resid_norm_over_s": float(np.linalg.norm(X[sel], axis=1).mean() / s_unit),
           "baseline": {"frac_yes": float((base > 0).mean()), "logit_diff_sd": float(base.std()),
                        "max_abs_diff_vs_cache": float(np.abs(base - ld[idx]).max())},
           "alphas": alphas, "directions": {}}
    slope_items = {}
    for k, n in enumerate(names):
        v = dirs[n]
        proj_sel = Xc @ v
        proj_rep = (X[rep] - mu) @ v
        own_sd = float(proj_sel.std())
        gap = float(proj_sel[y[sel] == 1].mean() - proj_sel[y[sel] == 0].mean())
        sl = (A[:, k, :, 0] - A[:, k, :, 1]) / (2 * np.array(alphas))[None]      # [items, alphas], logits per s
        a1 = alphas.index(1.0)
        slope_items[n] = sl[:, a1]
        plus, minus = A[:, k, :, 0], A[:, k, :, 1]
        flips = lambda x: float(np.mean((x > 0) != (base > 0)))                 # noqa: E731
        r = {"slope_per_s": [dict(alpha=al, **ci(lambda t: float(t.mean()), sl[:, ai])) for ai, al in enumerate(alphas)],
             "slope_per_s_at_alpha_1": ci(lambda t: float(t.mean()), sl[:, a1]),
             "frac_items_with_positive_slope_at_alpha_1": float((sl[:, a1] > 0).mean()),
             "projection_sd_over_s": own_sd / s_unit, "alpha_1_in_own_sd": s_unit / own_sd,
             "gold_class_gap_over_s": gap / s_unit,
             "slope_per_own_sd_at_alpha_1": float(sl[:, a1].mean() * own_sd / s_unit),
             "effect_of_natural_class_gap_on_logit_diff": float(sl[:, a1].mean() * gap / s_unit),
             "auroc_gold_report_half": stats.auroc(y[rep], proj_rep),
             "auroc_model_answer_report_half": stats.auroc(mcal[rep], proj_rep),
             "frac_sq_norm_in_top10_pcs": float(((top10 @ v) ** 2).sum()),
             "flip_rate": [{"alpha": al, "plus": ci(lambda b, x: float(np.mean((x > 0) != (b > 0))), base, plus[:, ai]),
                            "minus": ci(lambda b, x: float(np.mean((x > 0) != (b > 0))), base, minus[:, ai]),
                            "best_sign": max(flips(plus[:, ai]), flips(minus[:, ai])),
                            "frac_yes_plus": float((plus[:, ai] > 0).mean()), "frac_yes_minus": float((minus[:, ai] > 0).mean()),
                            "controllability_toward_gold": ci(lambda t, p, m: stats.accuracy(t, (np.where(t == 1, p, m) > 0).astype(int)), yy, plus[:, ai], minus[:, ai]),
                            "controllability_away_from_gold": ci(lambda t, p, m: stats.accuracy(t, (np.where(t == 1, m, p) > 0).astype(int)), yy, plus[:, ai], minus[:, ai])}
                           for ai, al in enumerate(alphas)]}
        if n in dirs_rep:
            r["split_half_cosine"] = float(v @ dirs_rep[n])
        # ablation: observed effect vs what the linear response predicts, and the unexplained residual (noise floor)
        abl = ABL[:, k] - base
        pred = -((X[idx] - mu) @ v) / s_unit * sl[:, a1]
        r["ablation"] = {"mean_abs_delta_logit_diff": float(np.abs(abl).mean()),
                         "mean_abs_predicted_from_slope": float(np.abs(pred).mean()),
                         "residual_sd_after_linear_prediction": float((abl - pred).std()),
                         "pearson_r_observed_vs_predicted": float(np.corrcoef(abl, pred)[0, 1]) if pred.std() > 0 else None}
        res["directions"][n] = r
    floor = float(np.median([res["directions"][n]["ablation"]["residual_sd_after_linear_prediction"] for n in names]))
    res["ablation_noise_floor_logit_diff"] = floor
    for n in names:
        ab = res["directions"][n]["ablation"]
        ab["above_noise_floor"] = bool(ab["mean_abs_delta_logit_diff"] > 3 * floor)
    # geometry
    Sig_w = Xc.T @ (Xc @ dirs["gold_probe"])
    res["geometry"] = {"cos_gold_probe_vs_diffmeans": float(dirs["gold_probe"] @ dirs["gold_diffmeans"]),
                       "cos_gold_probe_vs_model_out": float(dirs["gold_probe"] @ dirs["model_out"]),
                       "cos_diffmeans_vs_model_out": float(dirs["gold_diffmeans"] @ dirs["model_out"]),
                       "cos_covariance_times_probe_vs_diffmeans": float(steer.unit(Sig_w) @ dirs["gold_diffmeans"]),
                       "cos_diffmeans_vs_pc1": float(Vt[0] @ dirs["gold_diffmeans"])}
    # paired comparisons of |slope| at alpha = 1
    rnd = [n for n in names if n.startswith("random")]
    ab_ = {n: np.abs(slope_items[n]) for n in names}
    res["paired_abs_slope_differences"] = {
        **{f"gold_probe_minus_{n}": ci(lambda p, q: float(abs(p.mean()) - abs(q.mean())), slope_items["gold_probe"], slope_items[n]) for n in rnd},
        "gold_probe_minus_mean_abs_random": ci(lambda p, *q: float(abs(p.mean()) - np.mean([abs(t.mean()) for t in q])), slope_items["gold_probe"], *[slope_items[n] for n in rnd]),
        "gold_diffmeans_minus_gold_probe": ci(lambda p, q: float(abs(p.mean()) - abs(q.mean())), slope_items["gold_diffmeans"], slope_items["gold_probe"]),
        "model_out_minus_gold_probe": ci(lambda p, q: float(abs(p.mean()) - abs(q.mean())), slope_items["model_out"], slope_items["gold_probe"])}
    rs = np.array([slope_items[n].mean() for n in rnd])
    res["random_directions"] = {"n": len(rnd), "slopes_at_alpha_1": rs.tolist(), "rms_slope": float(np.sqrt((rs ** 2).mean())),
                                "gold_probe_slope_over_rms_random": float(slope_items["gold_probe"].mean() / np.sqrt((rs ** 2).mean())),
                                "note": "too few random directions to place the probe direction in their distribution; see steering_controls_*.json"}
    (RESULTS / f"steering_analysis_{a.model}_{a.tag}{a.suffix}.json").write_text(json.dumps(res, indent=1))
    g = res["directions"]
    print(f"{a.model}/{a.tag}{a.suffix} layer {layer} C {C:g}: slope/s  " + "  ".join(
        f"{n} {g[n]['slope_per_s_at_alpha_1']['est']:+.3f}" for n in names))
    print("   gold_probe: alpha=1 is %.0f own SDs; natural class gap moves LD by %+.4f (LD sd %.2f); AUROC gold %.3f; split-half cos %.2f; "
          "DoM AUROC gold %.3f / model answer %.3f" % (
              g["gold_probe"]["alpha_1_in_own_sd"], g["gold_probe"]["effect_of_natural_class_gap_on_logit_diff"], res["baseline"]["logit_diff_sd"],
              g["gold_probe"]["auroc_gold_report_half"], g["gold_probe"]["split_half_cosine"],
              g["gold_diffmeans"]["auroc_gold_report_half"], g["gold_diffmeans"]["auroc_model_answer_report_half"]))


if __name__ == "__main__":
    main()

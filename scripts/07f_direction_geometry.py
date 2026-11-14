"""What is the gold-label direction that was steered at the end of the hypothesis? CPU only.

Added after the eighth audit (notes/audit_round2_runs_raw.md, Q1 e), which computed these quantities independently;
this script puts them into a result file so that the write-up quotes results/, not the audit. For a model whose local
debiased-pool activations are the ones the GPU run used (Qwen 3B; Qwen 0.5B, which ran locally), at the site of the
end-of-hypothesis steering-controls run:

  - cosine of the gold difference of class means with the model-answer probe direction, with the difference of class
    means of the model's own answer, with the five per-template class-mean differences for adjacency that the
    adjacency test used at the same site (stored in its raw file), and with 20 within-string shuffled-label controls;
  - the share of the gold direction's squared norm inside the span of those five adjacency differences (and the same
    for the controls);
  - whether the gold label is related to the model's answer within hypothesis string on the selection half;
  - the size of the steering effect at natural scale: slope times the distance between the class means along the
    direction, next to the mean and spread of the unsteered answer logit difference on the steered items.

Writes results/direction_geometry_<model>_c2c_debiased_hyp.json.
"""
import argparse
import json

import numpy as np

from causal_probe import cache, data, probes, stats, steer
from causal_probe.config import ALL_MODELS, RESULTS, SEED

TAG = "c2c_debiased"
TEMPLATES = ["parent", "non-parent ancestor", "non-child descendant", "has_collider", "has_confounder"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    a = ap.parse_args()
    b = json.loads((RESULTS / f"steering_controls_{a.model}_{TAG}_hyp.json").read_text())
    layer, C = int(b["layer"]), float(b["probe_C"])
    resid, meta, items = cache.load_cache(a.model, TAG)
    X = probes.layer_features(resid, layer, meta["positions"].index("content_end")).astype(np.float64)
    y = np.array(meta["labels"]); ld = np.array(meta["logit_diff"])
    sel = data.selection_mask(items); rep = ~sel
    strs = np.array([i["query"] for i in items])
    z = np.load(cache.cache_dir(a.model, TAG) / "steering_controls_hyp.npz", allow_pickle=True)
    fam = np.array([str(f) for f in z["families"]])
    mu = X[sel].mean(0); s_unit = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
    assert abs(s_unit - float(z["s_unit"])) < 1e-3 * s_unit, "local activations are not the ones the steering run used"

    gold = steer.diff_of_means(X[sel], y[sel])
    own = ((X[sel] - mu) @ gold).std() / s_unit
    assert abs(own - float(z["own_sd_over_s"][fam == "gold_diffmeans"][0])) < 1e-4, "the regenerated gold direction differs from the run's"
    # the run's 20 within-string control labellings, regenerated with its generator (07c_steering_controls.py)
    rng = np.random.default_rng(SEED + 101)
    for _ in range(100):
        rng.standard_normal(X.shape[1])
    ctrl, strs_sel, ysel = [], strs[sel], y[sel]
    for _ in range(20):
        ys = ysel.copy()
        for s_ in np.unique(strs_sel):
            m_ = np.where(strs_sel == s_)[0]
            ys[m_] = ysel[rng.permutation(m_)]
        ctrl.append(steer.diff_of_means(X[sel], ys))
    ctrl = np.stack(ctrl)
    own_c = ((X[sel] - mu) @ ctrl.T).std(0) / s_unit
    assert np.abs(own_c / z["own_sd_over_s"][fam == "shuffled_diffmeans"] - 1).max() < 1e-3, "regenerated controls differ from the run's"

    mcal = (ld > np.median(ld)).astype(int)
    model_out = steer.probe_direction(X[sel], mcal[sel], C)
    ans_dom = steer.diff_of_means(X[sel], mcal[sel])
    res = {"model": a.model, "layer": layer, "n_layers": int(resid.shape[1] - 1), "position": "content_end",
           "regenerated_directions_match_the_run": True, "cosine_of_gold_difference_of_means_with": {}}
    cg = res["cosine_of_gold_difference_of_means_with"]

    def with_controls(v):
        c = ctrl @ v
        return {"gold": float(gold @ v), "controls_mean": float(c.mean()), "controls_sd": float(c.std(ddof=1)), "controls_max_abs": float(np.abs(c).max())}
    cg["model_answer_probe_direction"] = with_controls(model_out)
    cg["difference_of_class_means_of_the_models_answer"] = with_controls(ans_dom)
    cc = ctrl @ gold
    cg["the_20_shuffled_label_controls"] = {"mean": float(cc.mean()), "sd": float(cc.std(ddof=1)), "min": float(cc.min()), "max": float(cc.max())}
    adj_path = cache.cache_dir(a.model, TAG) / "steering_adjacency_hyp_best.npz"
    if adj_path.exists() and int(np.load(adj_path)["layer"]) == layer:
        delta = np.load(adj_path)["delta"].astype(np.float64)                      # [5 templates, d], class-mean differences for adjacency
        U = np.stack([steer.unit(d) for d in delta])
        cg["adjacency_class_mean_difference_by_template"] = {g: with_controls(U[i]) for i, g in enumerate(TEMPLATES)}
        Q, _ = np.linalg.qr(U.T)
        res["share_of_squared_norm_in_the_span_of_the_five_adjacency_differences"] = {
            "gold": float(((Q.T @ gold) ** 2).sum()), "controls_mean": float(((ctrl @ Q) ** 2).sum(1).mean()),
            "controls_max": float(((ctrl @ Q) ** 2).sum(1).max())}
    # is the gold label related to the model's answer within hypothesis string, on the half the direction was fitted on?
    au = [stats.auroc(y[sel & (strs == s_)], ld[sel & (strs == s_)]) for s_ in np.unique(strs)]
    res["gold_label_vs_models_answer_within_string_selection_half"] = {"mean_auroc": float(np.mean(au)), "n_strings": int(len(au))}
    # natural scale
    k = len(fam); out = z["out"].astype(np.float64); idx = z["idx"]
    slopes = ((out[:, 1:1 + k] - out[:, 1 + k:]) / (2 * float(z["alpha"]))).mean(0)
    j = int(np.where(fam == "gold_diffmeans")[0][0])
    gap = float(((X[rep & (y == 1)].mean(0) - X[rep & (y == 0)].mean(0)) @ gold) / s_unit)
    res["natural_scale"] = {
        "slope_per_s": float(slopes[j]), "natural_sd_over_s": float(own), "one_s_in_natural_sds": float(1 / own),
        "class_gap_along_direction_over_s_report_half": gap, "effect_of_moving_across_the_class_gap_logits": float(slopes[j] * gap),
        "unsteered_logit_diff_on_steered_items": {"mean": float(out[:, 0].mean()), "sd": float(out[:, 0].std(ddof=1)), "share_yes": float((out[:, 0] > 0).mean())},
        "effect_of_plus_one_s_logits": float((out[:, 1 + j] - out[:, 0]).mean()), "effect_of_minus_one_s_logits": float((out[:, 1 + k + j] - out[:, 0]).mean()),
        "share_of_items_with_positive_slope": float((((out[:, 1 + j] - out[:, 1 + k + j]) / 2) > 0).mean()),
        "share_of_answers_flipped_by_plus_one_s": float(((out[:, 1 + j] > 0) != (out[:, 0] > 0)).mean()),
        "share_of_answers_flipped_by_minus_one_s": float(((out[:, 1 + k + j] > 0) != (out[:, 0] > 0)).mean())}
    (RESULTS / f"direction_geometry_{a.model}_{TAG}_hyp.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:2600])


if __name__ == "__main__":
    main()

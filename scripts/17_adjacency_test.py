"""Round 2, item 3 (second version, after the independent code review of 16_feature_causal.py):
is the adjacency information causally used for the answer?

Adjacency = the two variables named in the hypothesis are correlated in the premise and no
conditioning set separates them. It is linearly decodable from the activations, and the models'
answers depend on it with a sign that differs by hypothesis template (the observational pattern).
This script intervenes on the activations' encoding of adjacency at one site (a layer and a token)
and asks whether the answer moves the way the observational pattern predicts, against the same
operations built from adjacency labels shuffled within hypothesis string.

What changed from the first version (notes/audit_round2_code_raw.md; notes/log.md 13 Nov 18:00):
  * everything is centred by hypothesis string, and control labels are shuffled within string, so a
    direction cannot encode "which hypothesis is asked" (finding F9);
  * the class-mean difference is taken per hypothesis template: the pooled one did not separate
    adjacency within "X directly causes Y" hypotheses (F4);
  * removal uses LEACE (least-squares concept erasure, per template) instead of a rank-4 INLP
    subspace, and the adjacency that a fresh probe can still read after removal is measured and
    saved (F3: the first version's ablation left the property readable at AUROC 0.84);
  * two step sizes per direction, so that a small real slope can be told from float16 noise and
    linearity can be checked; every applied vector's norm is saved, so effects can be expressed per
    natural SD or per unit norm afterwards (F5);
  * the readouts are effects with intervals, control distributions (not control means) and one
    pre-specified contrast across templates (F6, F7);
  * the second site is the first layer at 50% of the best layer's excess AUROC, not 90% (M1);
  * a self-check against a full forward pass in the model's own dtype at stage start, atomic
    checkpoints, and consistency checks between raw file and analysis (F2, M4).

Needs the model (GPU) for the intervention pass; --analyse-only recomputes the JSON on CPU from
cache/<model>/<tag>/steering_adjacency_<site>.npz. Writes results/adjacency_<model>_<tag>_<site>.json,
site = <final|hyp>_<best|first50|layer index>.

Directions, all fitted on the selection half of the pool's blocks, on string-centred activations:
  probe       weight vector of the logistic probe for adjacency (one direction for all templates)
  dom         per template: mean(adjacent) - mean(not adjacent); an item uses its own template's
  gold_probe  probe for the gold label;  model_out  probe for the model's median-split answer (raw
              activations; the positive control: it is fitted to the quantity that is measured)
Controls: the same constructions for N_SHUF shuffles of the adjacency labels within string.
Interventions on evaluation items (report half, --per-template items per template):
  step     +/- 2 natural SDs of the direction, and +/- 0.5 s (s = RMS distance of an activation from
           the mean activation), for every real and control direction
  translate  add (not-adjacent items) or subtract (adjacent items) the template's class-mean
           difference: on average this puts an item where the other class sits
  erase    LEACE per template: remove the component that carries adjacency, v = -delta (w.x)/(w.delta)
           with w = (Sigma + lambda I)^-1 delta
  transfer translate by the class-mean difference of EACH template (five vectors per item), to ask
           whether the encoding is shared across hypothesis types. If one adjacency feature is read
           in a template-specific way, the effect on an item follows the item's own template (the
           target); if each template's class difference is just that template's answer shift, the
           effect follows the template the vector came from (the source). Added after the first
           full-size result (notes/log.md 13 Nov 18:20): at Qwen 0.5B's final token the "parent"
           class difference is anti-parallel to the other four and all align with the answer direction.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from causal_probe import cache, data, models, probes, prompts, stats, steer
from causal_probe.config import ALL_MODELS, RESULTS, SEED

SMALL_SD, LARGE_S = 2.0, 0.5
TEMPLATES = ["parent", "non-parent ancestor", "non-child descendant", "has_collider", "has_confounder"]
REAL = ["probe", "dom", "gold_probe", "model_out"]


# ----------------------------------------------------------------------------- fitting (CPU)
def string_centre(X, strs, fit_mask):
    out = X.copy()
    for s_ in np.unique(strs):
        m = strs == s_
        assert (fit_mask & m).any(), f"hypothesis string absent from the selection half: {s_}"
        out[m] -= X[fit_mask & m].mean(0)
    return out


def shuffle_within(t, keys, rng):
    out = t.copy()
    for k in np.unique(keys):
        m = np.where(keys == k)[0]
        out[m] = t[rng.permutation(m)]
    return out


def template_deltas(Xc, t, tm, mask):
    """[G, d] class-mean differences (adjacent minus not adjacent) per template."""
    return np.stack([Xc[mask & (tm == g) & (t == 1)].mean(0) - Xc[mask & (tm == g) & (t == 0)].mean(0) for g in TEMPLATES])


def leace_w(Sinv, deltas):
    """Read-out directions w_g = (Sigma + lambda I)^-1 delta_g for every template, [G, d]."""
    return deltas @ Sinv


def erase_vectors(Xc, tm_idx, deltas, W):
    """Per-item LEACE edit v_i = -delta_g (w_g . x_i) / (w_g . delta_g), g = the item's template."""
    d, w = deltas[tm_idx], W[tm_idx]
    return -d * ((w * Xc).sum(1) / (w * d).sum(1))[:, None]


def residual_auroc(Xc_tr, t_tr, tm_tr, Xc_te, t_te, tm_te, C):
    """AUROC for adjacency of a fresh probe fitted on (already edited) training activations."""
    s = probes.fit_predict(Xc_tr, t_tr, Xc_te, C=C)
    out = {"pooled": stats.auroc(t_te, s)}
    for g in TEMPLATES:
        out[g] = stats.auroc(t_te[tm_te == g], s[tm_te == g])
    return out


def fit(X, sel, t, y, mcal, tm, strs, items, C, n_shuf, log):
    rng = np.random.default_rng(SEED + 311)
    rep = ~sel
    tm_idx = np.array([TEMPLATES.index(g) for g in tm])
    Xc = string_centre(X, strs, sel)
    mu = X[sel].mean(0)
    s_unit = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
    Xs, ts, tms, ss = Xc[sel], t[sel], tm[sel], strs[sel]
    # covariance of the string-centred selection-half activations, for LEACE
    Sig = np.cov(Xs.T.astype(np.float64))
    lam_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    mean_eig = float(np.trace(Sig) / len(Sig))
    # choose lambda on the selection half: the value that leaves a fresh probe closest to chance (3 grouped folds)
    idx_sel = np.where(sel)[0]
    folds = data.grouped_folds([items[j] for j in idx_sel], 3, SEED)
    score = {}
    for lam in lam_grid:
        dev = []
        for tr, te in folds:
            m_tr = np.zeros(len(Xs), bool); m_tr[tr] = True
            Sinv = np.linalg.inv(np.cov(Xs[tr].T.astype(np.float64)) + lam * mean_eig * np.eye(len(Sig)))
            dl = template_deltas(Xs, ts, tms, m_tr).astype(np.float64)
            W = leace_w(Sinv, dl)
            E = Xs + erase_vectors(Xs.astype(np.float64), tm_idx[sel], dl, W).astype(np.float32)
            r = residual_auroc(E[tr], ts[tr], tms[tr], E[te], ts[te], tms[te], C)
            dev.append(abs(r["pooled"] - 0.5))
        score[lam] = float(np.mean(dev))
        log(f"[leace] lambda {lam:g} x mean eigenvalue: held-out |AUROC - 0.5| after erasure {score[lam]:.3f}")
    lam = min(score, key=score.get)
    Sinv = np.linalg.inv(Sig + lam * mean_eig * np.eye(len(Sig)))

    def bundle(labels):
        dl = template_deltas(Xc, labels_full(labels), tm, sel).astype(np.float64)
        return dl, leace_w(Sinv, dl)

    def labels_full(lab_sel):            # labels defined on the selection half, placed in a full-length array
        full = np.zeros(len(X), int); full[sel] = lab_sel
        return full

    out = {"mu": mu, "s_unit": s_unit, "Xc": Xc, "tm_idx": tm_idx, "leace_lambda": lam, "leace_lambda_cv": score}
    out["probe"] = steer.probe_direction(Xs, ts, C)
    out["gold_probe"] = steer.probe_direction(Xs, y[sel], C)
    out["model_out"] = steer.probe_direction(X[sel], mcal[sel], C)
    out["delta"], out["W"] = bundle(ts)
    # how much adjacency can still be read after the real erasure (fresh probe; selection -> report half)
    E = Xc + erase_vectors(Xc.astype(np.float64), tm_idx, out["delta"], out["W"]).astype(np.float32)
    out["auroc_before"] = residual_auroc(Xc[sel], ts, tms, Xc[rep], t[rep], tm[rep], C)
    out["auroc_after_erasure"] = residual_auroc(E[sel], ts, tms, E[rep], t[rep], tm[rep], C)
    Pp = Xc - np.outer(Xc @ out["probe"], out["probe"])
    out["auroc_after_removing_probe_direction"] = residual_auroc(Pp[sel], ts, tms, Pp[rep], t[rep], tm[rep], C)
    log(f"adjacency AUROC of a fresh probe on the report half: {out['auroc_before']['pooled']:.3f} before, "
        f"{out['auroc_after_erasure']['pooled']:.3f} after LEACE, {out['auroc_after_removing_probe_direction']['pooled']:.3f} "
        f"after removing the probe direction only")
    out["shuf_probe"], out["shuf_delta"], out["shuf_W"], after = [], [], [], []
    for k in range(n_shuf):
        zk = shuffle_within(ts, ss, rng)
        out["shuf_probe"].append(steer.probe_direction(Xs, zk, C))
        dl, W = bundle(zk)
        out["shuf_delta"].append(dl); out["shuf_W"].append(W)
        if k < 5:      # the real labels' decodability after a control erasure (should stay high)
            Ek = Xc + erase_vectors(Xc.astype(np.float64), tm_idx, dl, W).astype(np.float32)
            after.append(residual_auroc(Ek[sel], ts, tms, Ek[rep], t[rep], tm[rep], C)["pooled"])
    out["auroc_after_control_erasure"] = after
    for k_ in ("shuf_probe", "shuf_delta", "shuf_W"):
        out[k_] = np.stack(out[k_])
    # natural SDs (selection half): probe-type directions pooled; class-mean differences within their template
    proj = lambda d: float((Xs @ d).std())                                     # noqa: E731
    out["sd_probe"] = proj(out["probe"]); out["sd_gold_probe"] = proj(out["gold_probe"])
    out["sd_model_out"] = float(((X[sel] - mu) @ out["model_out"]).std())
    out["sd_shuf_probe"] = np.array([proj(d) for d in out["shuf_probe"]])
    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)   # noqa: E731

    def sd_dom(dl):
        return np.array([float((Xs[tms == g] @ unit(dl[gi])).std()) for gi, g in enumerate(TEMPLATES)])
    out["sd_dom"] = sd_dom(out["delta"]); out["sd_shuf_dom"] = np.stack([sd_dom(dl) for dl in out["shuf_delta"]])
    # class gap along the probe direction, per template (selection half), in activation units
    out["gap_probe"] = np.array([float((Xs[(tms == g) & (ts == 1)] @ out["probe"]).mean() - (Xs[(tms == g) & (ts == 0)] @ out["probe"]).mean())
                                 for g in TEMPLATES])
    out["auroc_probe_direction_by_template"] = [stats.auroc(t[rep & (tm == g)], Xc[rep & (tm == g)] @ out["probe"]) for g in TEMPLATES]
    out["auroc_dom_direction_by_template"] = [stats.auroc(t[rep & (tm == g)], Xc[rep & (tm == g)] @ out["delta"][gi]) for gi, g in enumerate(TEMPLATES)]
    return out


def item_vectors(F, i, t_i, n_shuf):
    """All intervention vectors for item i, and the norm of each. Layout (K = 1 + 4n + 2(1 + n_shuf),
    n = 4 + 2 n_shuf directions in the order probe, dom, gold_probe, model_out, shuffled probes,
    shuffled doms; + 5 transfer vectors):  [0] none | +small x n | -small x n | +large x n | -large x n |
    translate: real, shuffles | erase: real, shuffles | transfer: the class-mean difference of each template"""
    g = F["tm_idx"][i]
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)                            # noqa: E731
    dirs = [F["probe"], unit(F["delta"][g]), F["gold_probe"], F["model_out"]] + list(F["shuf_probe"]) + [unit(dl[g]) for dl in F["shuf_delta"]]
    sds = [F["sd_probe"], F["sd_dom"][g], F["sd_gold_probe"], F["sd_model_out"]] + list(F["sd_shuf_probe"]) + [sd[g] for sd in F["sd_shuf_dom"]]
    D = np.stack(dirs); sds = np.array(sds)
    small = (SMALL_SD * sds)[:, None] * D
    large = LARGE_S * F["s_unit"] * D
    sgn = 1.0 if t_i == 0 else -1.0
    trans = np.stack([sgn * F["delta"][g]] + [sgn * dl[g] for dl in F["shuf_delta"]])
    x = F["Xc"][i].astype(np.float64)

    def er(dl, W):
        return -dl[g] * float(W[g] @ x) / float(W[g] @ dl[g])
    erase = np.stack([er(F["delta"], F["W"])] + [er(dl, W) for dl, W in zip(F["shuf_delta"], F["shuf_W"])])
    xfer = sgn * F["delta"]                                              # [5, d]: every template's class-mean difference
    V = np.concatenate([np.zeros((1, D.shape[1])), small, -small, large, -large, trans, erase, xfer]).astype(np.float32)
    return V, np.linalg.norm(V, axis=1)


# ----------------------------------------------------------------------------- analysis (CPU)
def two_sided(real, ctrl):
    """Place of a real value among control values: share of controls with a smaller absolute value,
    share of controls below it (signed), and the controls' range."""
    c = np.asarray(ctrl, float)
    return {"value": float(real), "n_controls": int(len(c)), "share_of_controls_smaller_in_absolute_value": float((np.abs(c) < abs(real)).mean()),
            "share_of_controls_below": float((c < real).mean()), "control_min": float(c.min()), "control_max": float(c.max()),
            "control_mean": float(c.mean()), "control_sd": float(c.std()),
            "outside_control_range": bool(real > c.max() or real < c.min())}


def obs_gaps(ld, t, tm, strs, blocks, n_boot=1000):
    """Per template: mean over hypothesis strings of [mean LD of adjacent items - mean LD of non-adjacent
    items], with a bootstrap over the pool's blocks."""
    est = obs_gaps_point(ld, t, tm, strs)
    rng = np.random.default_rng(SEED)
    ub = np.unique(blocks)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub))
        # weighted version: repeat blocks by replication count
        reps = np.concatenate([np.where(blocks == ub[b])[0] for b in range(len(ub)) for _ in range(w[b])])
        boots.append(obs_gaps_point(ld[reps], t[reps], tm[reps], strs[reps]))
    lo, hi = np.nanquantile(np.array(boots), [0.025, 0.975], axis=0)
    return est, lo, hi


def obs_gaps_point(ld, t, tm, strs):
    out = []
    for g in TEMPLATES:
        m = tm == g
        v = []
        for s_ in np.unique(strs[m]):
            a_, b_ = ld[m & (strs == s_) & (t == 1)], ld[m & (strs == s_) & (t == 0)]
            if len(a_) and len(b_):
                v.append(a_.mean() - b_.mean())
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


def analyse(a, z, meta, items, t_all, layer, layer_info):
    assert int(z["layer"]) == layer, f"raw file is for layer {int(z['layer'])}, the selection now gives {layer}: stale file?"
    n_shuf = int(z["n_shuf"]); n = 4 + 2 * n_shuf
    idx, out, norms = z["idx"], z["out"].astype(np.float64), z["norms"].astype(np.float64)
    K0 = 1 + 4 * n + 2 * (1 + n_shuf)
    assert out.shape == norms.shape and out.shape[0] == len(idx) and out.shape[1] in (K0, K0 + len(TEMPLATES)), out.shape
    has_xfer = out.shape[1] == K0 + len(TEMPLATES)
    assert len(idx) >= int(z["min_items"]), f"only {len(idx)} items in the raw file"
    y_all = np.array(meta["labels"]); ld_all = np.array(meta["logit_diff"])
    tm_all = np.array([i["template"] for i in items]); str_all = np.array([i["query"] for i in items])
    grp_all = np.array([i["group"] for i in items]); blk_all = np.array([i["block"] for i in items])
    t, tm, y, grp = t_all[idx], tm_all[idx], y_all[idx], grp_all[idx]
    base = out[:, 0]
    blk = lambda k: out[:, 1 + k * n: 1 + (k + 1) * n]                         # noqa: E731
    nrm = lambda k: norms[:, 1 + k * n: 1 + (k + 1) * n]                       # noqa: E731
    ps, ms_, pl, ml = blk(0), blk(1), blk(2), blk(3)
    slope_small = (ps - ms_) / (2 * nrm(0))           # logits per unit of activation norm
    slope_large = (pl - ml) / (2 * nrm(2))
    sd = nrm(0) / SMALL_SD                            # natural SD of each direction (per item: dom is per template)
    noise_small = np.abs((ps + ms_) / 2 - base[:, None])       # symmetric part: zero for a linear response
    tr = out[:, 1 + 4 * n: 1 + 4 * n + 1 + n_shuf]; tr_n = norms[:, 1 + 4 * n: 1 + 4 * n + 1 + n_shuf]
    er = out[:, 1 + 4 * n + 1 + n_shuf: K0]; er_n = norms[:, 1 + 4 * n + 1 + n_shuf: K0]
    xf = out[:, K0:]; xf_n = norms[:, K0:]
    sgn = np.where(t == 0, 1.0, -1.0)
    iP, iD, iG, iM = 0, 1, 2, 3
    cP = np.arange(4, 4 + n_shuf); cD = np.arange(4 + n_shuf, n)
    gm = [tm == g for g in TEMPLATES]

    def ci(v, m=None):
        m = np.ones(len(idx), bool) if m is None else m
        return stats.bootstrap_ci(lambda x: float(x.mean()), (v[m],), grp[m], n_boot=1000)

    G, Glo, Ghi = obs_gaps(ld_all, t_all, tm_all, str_all, blk_all)
    sig = np.where(Glo > 0, 1.0, np.where(Ghi < 0, -1.0, 0.0))                 # predicted sign where the gap's CI excludes zero
    res = {"model": a.model, "tag": a.tag, "position": a.position, "layer_spec": a.layer_spec, "layer": layer,
           "n_layers": int(z["n_layers"]), "probe_C": float(z["C"]), **layer_info,
           "n_items": int(len(idx)), "n_items_by_template": {g: int(m.sum()) for g, m in zip(TEMPLATES, gm)},
           "n_shuffles": n_shuf, "small_step_in_own_sd": SMALL_SD, "large_step_in_s": LARGE_S, "s_unit": float(z["s_unit"]),
           "self_check_shortcut_vs_full_forward_max_abs_diff": float(z["self_check"]),
           "baseline_max_abs_diff_vs_cache": float(np.abs(base - ld_all[idx]).max()), "baseline_logit_diff_sd": float(base.std()),
           "leace": {"lambda_times_mean_eigenvalue": float(z["leace_lambda"]),
                     "adjacency_auroc_of_fresh_probe_report_half": {
                         "before": json.loads(str(z["auroc_before"])), "after_erasure": json.loads(str(z["auroc_after_erasure"])),
                         "after_removing_probe_direction_only": json.loads(str(z["auroc_after_probe_dir"])),
                         "after_control_erasures_pooled": [float(v) for v in z["auroc_after_control"]]}},
           "directions": {"auroc_for_adjacency_report_half_by_template": {
               "probe": dict(zip(TEMPLATES, map(float, z["auroc_probe_dir"]))), "dom_own_template": dict(zip(TEMPLATES, map(float, z["auroc_dom_dir"])))},
               "natural_sd_over_s": {"probe": float(z["sd_probe"]) / float(z["s_unit"]),
                                     "shuffled_probes_range": [float(z["sd_shuf_probe"].min()) / float(z["s_unit"]), float(z["sd_shuf_probe"].max()) / float(z["s_unit"])],
                                     "model_out": float(z["sd_model_out"]) / float(z["s_unit"])},
               "class_mean_difference_norm_over_s": {g: float(np.linalg.norm(z["delta"][gi])) / float(z["s_unit"]) for gi, g in enumerate(TEMPLATES)},
               "shuffled_class_mean_difference_norm_over_s_range": [float(z["shuf_delta_norm"].min()) / float(z["s_unit"]),
                                                                    float(z["shuf_delta_norm"].max()) / float(z["s_unit"])]},
           "observational": {g: {"gap_adjacent_minus_not": {"est": float(G[gi]), "lo": float(Glo[gi]), "hi": float(Ghi[gi])},
                                 "predicted_sign": int(sig[gi])} for gi, g in enumerate(TEMPLATES)},
           "note_observational": "all 2400 pool items, cached answer logits; within hypothesis string, unweighted mean over strings; bootstrap over the 25 blocks"}
    used = sig != 0
    res["contrast_definition"] = ("mean over templates whose observational gap excludes zero of predicted_sign x effect; "
                                  "full mediation would give 'full_mediation_benchmark' = mean |gap| over those templates")
    bench = float(np.abs(G[used]).mean()) if used.any() else float("nan")
    res["full_mediation_benchmark"] = bench
    res["templates_in_contrast"] = [g for g, u in zip(TEMPLATES, used) if u]

    def contrast(per_template):                      # per_template: [G] or [G, k]
        v = np.asarray(per_template, float)
        return (sig[used, None] * v[used].reshape(used.sum(), -1)).mean(0) if used.any() else np.full(v.reshape(len(TEMPLATES), -1).shape[1], np.nan)

    def contrast_ci(eff):                            # eff: per-item signed effect
        codes = np.array([TEMPLATES.index(g) for g in tm])
        def fn(e, c):
            return float(np.mean([sig[gi] * e[c == gi].mean() for gi in range(len(TEMPLATES)) if used[gi] and (c == gi).any()])) if used.any() else float("nan")
        return stats.bootstrap_ci(fn, (eff, codes), grp, n_boot=1000)

    # ---- 1. steps: slope of the answer logit difference
    def slope_block(j, ctrl, name):
        r = {"by_template": {}}
        for conv, sl, scale in (("per_unit_norm_from_large_step", slope_large, None), ("per_unit_norm_from_small_step", slope_small, None),
                                ("per_natural_sd_from_small_step", slope_small * sd, None)):
            r[conv] = {**ci(sl[:, j]), "vs_controls": two_sided(sl[:, j].mean(), sl[:, ctrl].mean(0))}
        for gi, g in enumerate(TEMPLATES):
            m = gm[gi]
            r["by_template"][g] = {
                "per_unit_norm_from_large_step": {**ci(slope_large[:, j], m), "vs_controls": two_sided(slope_large[m, j].mean(), slope_large[m][:, ctrl].mean(0))},
                "per_natural_sd_from_small_step": {**ci((slope_small * sd)[:, j], m),
                                                   "vs_controls": two_sided((slope_small * sd)[m, j].mean(), (slope_small * sd)[m][:, ctrl].mean(0))}}
        r["linearity_ratio_small_over_large_slope"] = float(slope_small[:, j].mean() / slope_large[:, j].mean()) if abs(slope_large[:, j].mean()) > 1e-9 else None
        r["noise_floor_mean_abs_symmetric_part_small_step"] = float(noise_small[:, j].mean())
        r["mean_abs_effect_of_small_step"] = float(np.abs(ps[:, j] - base).mean())
        return r

    res["steps"] = {"probe": slope_block(iP, cP, "probe"), "dom_own_template": slope_block(iD, cD, "dom"),
                    "gold_probe": slope_block(iG, cP, "gold_probe"), "model_out": slope_block(iM, cP, "model_out")}
    # effect of moving across the natural class gap along the probe direction, per template
    gap_p = z["gap_probe"].astype(np.float64)
    codes = np.array([TEMPLATES.index(g) for g in tm])
    eff_gap = slope_large[:, iP] * gap_p[codes]
    eff_gap_c = slope_large[:, cP] * gap_p[codes][:, None]                     # controls moved by the same distance
    res["probe_direction_across_class_gap"] = {
        "by_template": {g: {**ci(eff_gap, gm[gi]), "class_gap_in_natural_sd": float(gap_p[gi] / float(z["sd_probe"])),
                            "vs_controls": two_sided(eff_gap[gm[gi]].mean(), eff_gap_c[gm[gi]].mean(0)),
                            "observational_gap": float(G[gi])} for gi, g in enumerate(TEMPLATES)},
        "contrast": {**contrast_ci(eff_gap),
                     "vs_controls": two_sided(float(contrast([eff_gap[m].mean() for m in gm])[0]), contrast(np.stack([eff_gap_c[m].mean(0) for m in gm])))}}

    # ---- 2. translation by the template's class-mean difference
    e_tr = sgn * (tr[:, 0] - base)                                              # > 0: towards Yes when moved towards "adjacent"
    e_tr_c = sgn[:, None] * (tr[:, 1:] - base[:, None])
    scale = tr_n[:, :1] / tr_n[:, 1:]                                           # norm of the real translation / norm of each control's
    e_tr_cm = e_tr_c * scale                                                    # controls rescaled to the real translation's norm (linear response assumed)
    res["translation"] = {"by_template": {}, "note": "effect towards Yes of moving an item to where the other adjacency class sits on average"}
    for gi, g in enumerate(TEMPLATES):
        m = gm[gi]
        res["translation"]["by_template"][g] = {
            **ci(e_tr, m), "observational_gap": float(G[gi]), "norm_over_s": float(tr_n[m, 0].mean() / float(z["s_unit"])),
            "vs_controls_as_fitted": two_sided(e_tr[m].mean(), e_tr_c[m].mean(0)),
            "vs_controls_norm_matched": two_sided(e_tr[m].mean(), e_tr_cm[m].mean(0))}
    c_real = float(contrast([e_tr[m].mean() for m in gm])[0])
    res["translation"]["contrast"] = {**contrast_ci(e_tr),
                                      "vs_controls_as_fitted": two_sided(c_real, contrast(np.stack([e_tr_c[m].mean(0) for m in gm]))),
                                      "vs_controls_norm_matched": two_sided(c_real, contrast(np.stack([e_tr_cm[m].mean(0) for m in gm]))),
                                      "share_of_full_mediation": c_real / bench if bench == bench and bench > 0 else None}

    # ---- 3. erasure (LEACE): does the answer's dependence on adjacency shrink?
    def gap_eval(ld, m):                               # difference of mean answer logit, adjacent minus not, on evaluation items
        return float(ld[m & (t == 1)].mean() - ld[m & (t == 0)].mean())

    res["erasure"] = {"by_template": {}, "mean_abs_delta_logit_diff": two_sided(float(np.abs(er[:, 0] - base).mean()), np.abs(er[:, 1:] - base[:, None]).mean(0)),
                      "norm_of_edit_over_s": {"real": float(er_n[:, 0].mean() / float(z["s_unit"])),
                                              "controls_range": [float(er_n[:, 1:].mean(0).min() / float(z["s_unit"])), float(er_n[:, 1:].mean(0).max() / float(z["s_unit"]))]}}
    # control erasures are smaller edits than the real one (the real labels' read-out has more variance to remove).
    # Under a linear response a control edit scaled to the real edit's mean norm changes the answer by the same factor:
    er_scale = er_n[:, 0].mean() / er_n[:, 1:].mean(0)                           # [n_shuf]
    er_cm = base[:, None] + (er[:, 1:] - base[:, None]) * er_scale[None, :]     # control-erased answers, rescaled
    ch, chc, chcm = [], [], []
    for gi, g in enumerate(TEMPLATES):
        m = gm[gi]
        g0 = gap_eval(base, m); g1 = gap_eval(er[:, 0], m); gc = np.array([gap_eval(er[:, 1 + k], m) for k in range(n_shuf)])
        gcm = np.array([gap_eval(er_cm[:, k], m) for k in range(n_shuf)])
        ch.append(g1 - g0); chc.append(gc - g0); chcm.append(gcm - g0)
        bt = stats.bootstrap_ci(lambda b_, e_, tt: float((e_[tt == 1].mean() - e_[tt == 0].mean()) - (b_[tt == 1].mean() - b_[tt == 0].mean())),
                                (base[m], er[m, 0], t[m]), grp[m], n_boot=1000)
        res["erasure"]["by_template"][g] = {
            "gap_on_evaluation_items_before": g0, "gap_after_erasure": g1,
            "change_in_gap": {**bt, "vs_controls": two_sided(g1 - g0, gc - g0), "vs_controls_norm_matched": two_sided(g1 - g0, gcm - g0)},
            "auroc_answer_for_adjacency_before": stats.auroc(t[m], base[m]), "auroc_answer_for_adjacency_after": stats.auroc(t[m], er[m, 0]),
            "auroc_after_control_erasures_range": [float(min(stats.auroc(t[m], er[m, 1 + k]) for k in range(n_shuf))),
                                                   float(max(stats.auroc(t[m], er[m, 1 + k]) for k in range(n_shuf)))],
            "mean_abs_delta_logit_diff": two_sided(float(np.abs(er[m, 0] - base[m]).mean()), np.abs(er[m, 1:] - base[m, None]).mean(0))}
    # contrast: predicted_sign x (gap before - gap after) > 0 means the erasure shrinks the dependence in the predicted direction
    res["erasure"]["contrast_reduction_of_dependence"] = {
        "value": float(contrast(-np.array(ch))[0]), "vs_controls": two_sided(float(contrast(-np.array(ch))[0]), contrast(-np.stack(chc))),
        "vs_controls_norm_matched": two_sided(float(contrast(-np.array(ch))[0]), contrast(-np.stack(chcm))),
        "control_edit_norm_scale_factors_range": [float(er_scale.min()), float(er_scale.max())],
        "benchmark_full_removal_on_evaluation_items": float(contrast([gap_eval(base, m) for m in gm])[0])}
    # ---- 4. transfer: the class-mean difference of template h applied to items of template g
    if has_xfer:
        e_x = sgn[:, None] * (xf - base[:, None])                               # [items, source template]
        M = np.array([[e_x[gm[gi], hi].mean() for hi in range(len(TEMPLATES))] for gi in range(len(TEMPLATES))])   # [target, source]
        cells = {}
        for gi, g in enumerate(TEMPLATES):
            for hi, h in enumerate(TEMPLATES):
                m = gm[gi]
                ctrl = e_tr_c[m] * (xf_n[m, hi][:, None] / tr_n[m, 1:])         # target's own controls, rescaled to this vector's norm
                cells[f"{h} -> {g}"] = {**ci(e_x[:, hi], m), "vs_controls_norm_matched": two_sided(M[gi, hi], ctrl.mean(0))}
        off = [(gi, hi) for gi in range(len(TEMPLATES)) for hi in range(len(TEMPLATES)) if gi != hi and used[gi] and used[hi]]
        D_ = z["delta"].astype(np.float64); U_ = D_ / np.linalg.norm(D_, axis=1, keepdims=True)

        def xcontrast(which):                                                    # per-item effects -> mean over off-diagonal cells
            codes_ = np.array([TEMPLATES.index(g) for g in tm])
            def fn(*cols):
                e, c = np.stack(cols[:-1], 1), cols[-1]
                return float(np.mean([sig[gi if which == "target" else hi] * e[c == gi, hi].mean() for gi, hi in off])) if off else float("nan")
            return stats.bootstrap_ci(fn, tuple(e_x[:, hi] for hi in range(len(TEMPLATES))) + (codes_,), grp, n_boot=1000)

        res["transfer"] = {
            "templates": TEMPLATES, "effect_matrix_rows_target_cols_source": M.tolist(), "cells": cells,
            "cosine_between_class_mean_differences": (U_ @ U_.T).tolist(),
            "off_diagonal_contrast_with_the_TARGET_template_sign": xcontrast("target"),
            "off_diagonal_contrast_with_the_SOURCE_template_sign": xcontrast("source"),
            "note": "effect towards Yes, per move towards adjacent, of translating an item of the row's template by the class-mean "
                    "difference of the column's template. A shared adjacency feature read in a template-specific way predicts a "
                    "positive target-sign contrast; class differences that are each template's own answer shift predict a positive "
                    "source-sign contrast."}
    res["minutes_gpu"] = float(z["minutes"])
    (RESULTS / f"adjacency_{a.model}_{a.tag}_{a.site}.json").write_text(json.dumps(res, indent=1))

    print(f"{a.model}/{a.tag} {a.site}: layer {layer}/{res['n_layers']}; fresh-probe adjacency AUROC "
          f"{res['leace']['adjacency_auroc_of_fresh_probe_report_half']['before']['pooled']:.3f} -> "
          f"{res['leace']['adjacency_auroc_of_fresh_probe_report_half']['after_erasure']['pooled']:.3f} after erasure; "
          f"self-check {res['self_check_shortcut_vs_full_forward_max_abs_diff']:.4f}")
    print("  observational gap : " + " ".join(f"{g[:10]} {G[gi]:+.3f}[{Glo[gi]:+.2f},{Ghi[gi]:+.2f}]" for gi, g in enumerate(TEMPLATES)))
    print("  translation effect: " + " ".join(f"{g[:10]} {res['translation']['by_template'][g]['est']:+.3f}" for g in TEMPLATES)
          + f" | contrast {res['translation']['contrast']['est']:+.4f} (benchmark {bench:.3f}; controls below: "
            f"{res['translation']['contrast']['vs_controls_norm_matched']['share_of_controls_below']:.2f})")
    print("  probe across gap  : " + " ".join(f"{g[:10]} {res['probe_direction_across_class_gap']['by_template'][g]['est']:+.4f}" for g in TEMPLATES)
          + f" | contrast {res['probe_direction_across_class_gap']['contrast']['est']:+.4f}")
    print("  erasure, gap change: " + " ".join(f"{g[:10]} {res['erasure']['by_template'][g]['change_in_gap']['est']:+.3f}" for g in TEMPLATES)
          + f" | reduction contrast {res['erasure']['contrast_reduction_of_dependence']['value']:+.4f} (full removal "
            f"{res['erasure']['contrast_reduction_of_dependence']['benchmark_full_removal_on_evaluation_items']:+.3f})")
    if has_xfer:
        x_ = res["transfer"]
        print("  transfer (rows = item's template, columns = template the vector came from):")
        for gi, g in enumerate(TEMPLATES):
            print("    %-10s " % g[:10] + " ".join(f"{v:+.3f}" for v in x_["effect_matrix_rows_target_cols_source"][gi]))
        print(f"    off-diagonal contrast, target sign {x_['off_diagonal_contrast_with_the_TARGET_template_sign']['est']:+.4f} "
              f"[{x_['off_diagonal_contrast_with_the_TARGET_template_sign']['lo']:+.4f}, {x_['off_diagonal_contrast_with_the_TARGET_template_sign']['hi']:+.4f}]; "
              f"source sign {x_['off_diagonal_contrast_with_the_SOURCE_template_sign']['est']:+.4f} "
              f"[{x_['off_diagonal_contrast_with_the_SOURCE_template_sign']['lo']:+.4f}, {x_['off_diagonal_contrast_with_the_SOURCE_template_sign']['hi']:+.4f}]")
    m_ = res["steps"]["model_out"]["per_unit_norm_from_large_step"]
    print(f"  positive control (model-answer direction): slope {m_['est']:+.4f} per unit norm, larger in absolute value than "
          f"{100 * m_['vs_controls']['share_of_controls_smaller_in_absolute_value']:.0f}% of shuffled-label probe directions")


def choose_layer(a, resid, items, sel, t, pi, log):
    path = RESULTS / f"feature_select_{a.model}_{a.tag}_adj_{a.position}.json"       # shared with the first version
    if not path.exists():
        l, c, grid = probes.quick_select(resid, t, items, sel, a.select_stride, log, pos=pi, metric="auroc")
        path.write_text(json.dumps({"model": a.model, "tag": a.tag, "feature": "pair_adjacent_in_skeleton", "position": a.position,
                                    "best_layer": l, "best_C": c, "stride": a.select_stride, "selection_half_cv_auroc": grid}, indent=1))
    s = json.loads(path.read_text())
    grid = {(int(k.split("|")[0]), float(k.split("|")[1])): v for k, v in s["selection_half_cv_auroc"].items()}
    layers = sorted({l for l, _ in grid})
    best = {l: max((v, c) for (ll, c), v in grid.items() if ll == l) for l in layers}
    top = best[s["best_layer"]][0]
    if a.layer_spec == "best":
        layer = s["best_layer"]
    elif a.layer_spec.startswith("first"):
        frac = int(a.layer_spec[5:]) / 100
        layer = next(l for l in layers if best[l][0] - 0.5 >= frac * (top - 0.5))
    else:
        layer = min(layers, key=lambda l: abs(l - int(a.layer_spec)))
    return int(layer), float(best[layer][1]), {"selection_auroc_at_layer": float(best[layer][0]), "best_layer": int(s["best_layer"]),
                                               "selection_auroc_at_best_layer": float(top), "selection_stride": int(s["stride"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", default="c2c_debiased", choices=["c2c_debiased"])
    ap.add_argument("--position", default="final", choices=["final", "content_end"])
    ap.add_argument("--layer-spec", default="best", help="best | first50 (first layer at 50%% of the best layer's excess AUROC) | <cache layer index>")
    ap.add_argument("--per-template", type=int, default=100)
    ap.add_argument("--n-shuf", type=int, default=20)
    ap.add_argument("--sub-batch", type=int, default=16)
    ap.add_argument("--select-stride", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    a.site = f"{'hyp' if a.position == 'content_end' else 'final'}_{a.layer_spec}"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    resid, meta, items = cache.load_cache(a.model, a.tag)
    pi = meta["positions"].index(a.position)
    t = probes.structural_features(items)[:, 2].astype(int)
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"])
    tm = np.array([i["template"] for i in items]); strs = np.array([i["query"] for i in items])
    assert set(tm) == set(TEMPLATES), set(tm)
    sel = data.selection_mask(items); rep = ~sel
    layer, C, layer_info = choose_layer(a, resid, items, sel, t, pi, log)
    raw_path = cache.cache_dir(a.model, a.tag) / f"steering_adjacency_{a.site}.npz"
    if a.layer_spec != "best" and layer == layer_info["best_layer"]:
        log(f"site {a.site}: the selected layer equals the best layer ({layer}); nothing to add, skipping")
        return
    if a.analyse_only:
        analyse(a, np.load(raw_path), meta, items, t, layer, layer_info)
        return
    log(f"site {a.site}: layer {layer}, C {C:g}, selection-half AUROC {layer_info['selection_auroc_at_layer']:.3f}")
    X = probes.layer_features(resid, layer, pi)
    mcal = (ld0[sel] > np.median(ld0[sel])).astype(int)
    mcal_full = np.zeros(len(X), int); mcal_full[sel] = mcal
    F = fit(X, sel, t, y, mcal_full, tm, strs, items, C, a.n_shuf, log)
    log(f"directions fitted ({4 + 2 * a.n_shuf} step directions, {1 + a.n_shuf} translations, {1 + a.n_shuf} erasers)")

    rng = np.random.default_rng(SEED + 17)
    per = [rng.permutation(np.where(rep & (tm == g))[0])[: a.per_template] for g in TEMPLATES]
    idx = np.array([p[r] for r in range(max(len(p) for p in per)) for p in per if r < len(p)])   # round-robin over templates
    V0, N0 = item_vectors(F, idx[0], int(t[idx[0]]), a.n_shuf)
    K = len(V0)
    sig_ = float(np.abs(V0).sum())
    if raw_path.exists():
        zr = np.load(raw_path)
        if int(zr["layer"]) == layer and abs(float(zr["signature"]) - sig_) < 1e-3 * sig_ and np.array_equal(zr["idx"], idx[: len(zr["idx"])]):
            log("a raw file for exactly this configuration exists; skipping the GPU pass")
            analyse(a, zr, meta, items, t, layer, layer_info)
            return
    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    # self-check in the model's own dtype and device: shortcut vs full forward pass, three vectors at the steered token
    enc0 = prompts.encode_with_positions(tok, items[idx[0]])
    n_dir = 4 + 2 * a.n_shuf
    pick = torch.tensor(V0[[0, 2 * n_dir + 2, 1 + 4 * n_dir + 1 + a.n_shuf]])        # none, a large step, the real erasure
    fast = steer.steered_logit_diffs(model, enc0["input_ids"], layer, pick, ans, sub_batch=3, pos=enc0["pos"][a.position])
    slow = steer.steered_logit_diffs_full(model, enc0["input_ids"], layer, pick, ans, pos=enc0["pos"][a.position])
    self_check = float(np.abs(fast - slow).max())
    log(f"self-check, shortcut vs full forward pass: max |difference| {self_check:.4f} logits; unsteered vs cache {abs(fast[0] - ld0[idx[0]]):.4f}")
    if self_check > 0.25:
        raise SystemExit("self-check failed: the key-value shortcut disagrees with a full forward pass")

    out = np.zeros((len(idx), K), np.float32); norms = np.zeros((len(idx), K), np.float32)
    ckpt = cache.cache_dir(a.model, a.tag) / f"steering_adjacency_{a.site}_partial.npz"
    start = 0
    if ckpt.exists():
        try:
            zc = np.load(ckpt)
            m_ = len(zc["idx"])
            if zc["out"].shape[1] == K and np.array_equal(zc["idx"], idx[:m_]) and abs(float(zc["signature"]) - sig_) < 1e-3 * sig_ \
                    and (m_ == len(idx) or bool(zc["shortened"])):
                idx, out, norms = idx[:m_], out[:m_], norms[:m_]
                start = int(zc["done"]); out[:start] = zc["out"][:start]; norms[:start] = zc["norms"][:start]
                log(f"resuming at item {start}/{len(idx)}")
        except Exception as e:      # an interrupted write: start again
            log(f"checkpoint unreadable ({type(e).__name__}); starting from the first item")
    shortened = len(idx) < sum(len(p) for p in per)

    def save_ckpt(done):
        tmp = ckpt.with_name(ckpt.name + ".tmp.npz")
        np.savez(tmp, idx=idx, out=out, norms=norms, done=done, signature=sig_, shortened=shortened)
        os.replace(tmp, ckpt)

    t_loop = time.time()
    n_ = start
    while n_ < len(idx):
        i = idx[n_]
        enc = prompts.encode_with_positions(tok, items[i])
        V, N = item_vectors(F, i, int(t[i]), a.n_shuf)
        out[n_] = steer.steered_logit_diffs(model, enc["input_ids"], layer, torch.tensor(V), ans, sub_batch=a.sub_batch, pos=enc["pos"][a.position])
        norms[n_] = N
        n_ += 1
        if n_ - start == 6 and start == 0:              # timing on items 2-6 (the first includes warm-up)
            per_item = (time.time() - t_loop) / 6
            budget = a.max_minutes * 60 - (time.time() - t0)
            proj = per_item * (len(idx) - n_)
            log(f"[timing] {per_item:.2f} s/item ({K} vectors) -> {proj / 60:.1f} min for the remaining {len(idx) - n_} items; "
                f"{budget / 60:.1f} min left of {a.max_minutes:.0f}")
            if proj > budget:
                keep = n_ + int(budget / per_item)
                if keep < 100:
                    raise SystemExit("fewer than 100 items would fit the time limit")
                log(f"[timing] shortening the run to the first {keep} items (balanced over templates)")
                idx, out, norms, shortened = idx[:keep], out[:keep], norms[:keep], True
        if n_ % 50 == 0:
            save_ckpt(n_)
            log(f"{n_}/{len(idx)}")
        if model.device.type == "mps" and n_ % 5 == 0:
            torch.mps.empty_cache()
    np.savez(raw_path, idx=idx, out=out, norms=norms, layer=layer, n_layers=resid.shape[1] - 1, C=C, n_shuf=a.n_shuf,
             s_unit=F["s_unit"], signature=sig_, self_check=self_check, min_items=min(100, len(idx)), per_template=a.per_template,
             shortened=shortened, leace_lambda=F["leace_lambda"], sd_probe=F["sd_probe"], sd_model_out=F["sd_model_out"],
             sd_shuf_probe=F["sd_shuf_probe"], gap_probe=F["gap_probe"],
             delta=F["delta"], shuf_delta_norm=np.linalg.norm(F["shuf_delta"], axis=-1),
             probe=F["probe"], gold_probe=F["gold_probe"], model_out=F["model_out"], W=F["W"],
             auroc_before=json.dumps(F["auroc_before"]), auroc_after_erasure=json.dumps(F["auroc_after_erasure"]),
             auroc_after_probe_dir=json.dumps(F["auroc_after_removing_probe_direction"]), auroc_after_control=F["auroc_after_control_erasure"],
             auroc_probe_dir=F["auroc_probe_direction_by_template"], auroc_dom_dir=F["auroc_dom_direction_by_template"],
             minutes=(time.time() - t0) / 60)
    del model
    analyse(a, np.load(raw_path), meta, items, t, layer, layer_info)
    ckpt.unlink(missing_ok=True)                     # only after the analysis has succeeded


if __name__ == "__main__":
    main()

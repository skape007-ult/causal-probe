"""Round 2, item 3: is a decodable structural property of the premise causally used for the answer?

Default property: adjacency of the queried pair in the skeleton ("the two variables named in the
hypothesis are correlated and no conditioning set separates them"), on the debiased Corr2Cause pool.
Background (probe_content_*.json, notes/log.md 13 Nov 16:45): the property is linearly decodable at
AUROC 0.89-0.98, and the models' answers depend on it with a sign that differs by hypothesis
template. This script intervenes on the direction(s) that carry the property and asks whether the
answer moves the way that observational pattern predicts, compared with matched control directions.

Needs the model (GPU) for the intervention pass; `--analyse-only` recomputes the JSON from the saved
raw logit differences (cache/<model>/<tag>/steering_feature_<site>.npz) on CPU.
Writes results/feature_causal_<model>_<tag>_<site>.json, site = <prop>_<position>_<layer spec>.

Site. One token position (`final` = last prompt token, `content_end` = last token of the hypothesis)
and one layer: `best` = the layer where the property probe has the best cross-validated AUROC on the
selection half, `first90` = the earliest layer that reaches 90% of that layer's AUROC excess over 0.5.
The answer is always read at the final token.

Directions (unit vectors; everything is fitted on the selection half of the pool's blocks):
  feat_probe   weight vector of the logistic probe for the property
  feat_dom     difference of class means for the property, after removing each template's mean
  gold_probe, gold_dom, model_out     the same for the gold label / the model's median-split answer
  feat_inlp    a rank-k subspace: k rounds of "fit the property probe, project its direction out"
Controls (same construction, no information about the property):
  shuf_probe / shuf_dom / shuf_inlp   N_SHUF repetitions with the property labels shuffled WITHIN
                                      template (so the template marginals are kept)
  iso          N_ISO isotropic random directions
Interventions on evaluation items (report half, up to --per-template items per template):
  add      +/- 2 natural SDs of the direction (its own spread over the selection half)
  ablate   set the item's component along the direction (or subspace) to the dataset mean
  shift    move the item along the direction towards the OTHER property class by the real
           direction's class gap, expressed in own SDs (for controls: the same number of own SDs)
Readouts, each for the real direction and for the distribution over its control family:
  slope per natural SD, by template; the correlation across templates between that slope and the
  observational dependence of the answer on the property; the change in how well the answer tracks
  the gold label and the property (AUROC) after ablation; the mean signed shift effect.
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import cache, data, models, probes, prompts, stats, steer
from causal_probe.config import ALL_MODELS, RESULTS, SEED

PROPS = {"pair_adjacent_in_skeleton": lambda F: F[:, 2], "pair_correlated": lambda F: F[:, 0],
         "pair_has_common_neighbour": lambda F: F[:, 8] > 0}
SHORT = {"pair_adjacent_in_skeleton": "adj", "pair_correlated": "corr", "pair_has_common_neighbour": "cn"}
ADD_SD = 2.0


def template_centre(X, tm, fit_mask):
    """X minus the mean of its template (means from the rows in fit_mask)."""
    out = X.copy()
    for g in np.unique(tm):
        out[tm == g] -= X[fit_mask & (tm == g)].mean(0)
    return out


def shuffle_within(t, tm, rng):
    out = t.copy()
    for g in np.unique(tm):
        m = np.where(tm == g)[0]
        out[m] = t[rng.permutation(m)]
    return out


def inlp(X, t, C, k):
    """k orthonormal directions: fit the probe, remove its direction from X, repeat."""
    Q, Xr = [], X.copy()
    for _ in range(k):
        w = steer.probe_direction(Xr, t, C)
        for q in Q:
            w = w - (w @ q) * q
        w = steer.unit(w)
        Q.append(w)
        Xr = Xr - np.outer((Xr - Xr.mean(0)) @ w, w)
    return np.stack(Q)


def fit_directions(X, sel, t, y, mcal, tm, C, n_shuf, n_iso, rank, log):
    """Returns names, family labels, unit directions [n, d], and the INLP bases [1 + n_shuf, rank, d]."""
    rng = np.random.default_rng(SEED + 211)
    Xs, ts, tms = X[sel], t[sel], tm[sel]
    Xc = template_centre(Xs, tms, np.ones(len(Xs), bool))
    names, fam, D = [], [], []

    def add(n, f, v):
        names.append(n); fam.append(f); D.append(steer.unit(np.asarray(v, np.float64)))

    add("feat_probe", "real", steer.probe_direction(Xs, ts, C))
    add("feat_dom", "real", steer.diff_of_means(Xc, ts))
    add("gold_probe", "real", steer.probe_direction(Xs, y[sel], C))
    add("gold_dom", "real", steer.diff_of_means(Xc, y[sel]))
    add("model_out", "real", steer.probe_direction(Xs, mcal[sel], C))
    bases = [inlp(Xs, ts, C, rank)]
    log(f"real directions fitted (rank-{rank} subspace included)")
    for s in range(n_shuf):
        tsh = shuffle_within(ts, tms, rng)
        add(f"shuf_probe_{s}", "shuf_probe", steer.probe_direction(Xs, tsh, C))
        add(f"shuf_dom_{s}", "shuf_dom", steer.diff_of_means(Xc, tsh))
        bases.append(inlp(Xs, tsh, C, rank))
    for s in range(n_iso):
        add(f"iso_{s}", "iso", rng.standard_normal(X.shape[1]))
    return names, np.array(fam), np.stack(D).astype(np.float32), np.stack(bases).astype(np.float32)


def choose_layer(a, resid, items, sel, t, pi, log):
    path = RESULTS / f"feature_select_{a.model}_{a.tag}_{SHORT[a.feature]}_{a.position}.json"
    if not path.exists():
        l, c, grid = probes.quick_select(resid, t, items, sel, a.select_stride, log, pos=pi, metric="auroc")
        path.write_text(json.dumps({"model": a.model, "tag": a.tag, "feature": a.feature, "position": a.position,
                                    "best_layer": l, "best_C": c, "stride": a.select_stride,
                                    "selection_half_cv_auroc": grid}, indent=1))
    s = json.loads(path.read_text())
    grid = {(int(k.split("|")[0]), float(k.split("|")[1])): v for k, v in s["selection_half_cv_auroc"].items()}
    layers = sorted({l for l, _ in grid})
    best_by_layer = {l: max((v, c) for (ll, c), v in grid.items() if ll == l) for l in layers}
    top = best_by_layer[s["best_layer"]][0]
    if a.layer_spec == "best":
        layer = s["best_layer"]
    elif a.layer_spec == "first90":
        layer = next(l for l in layers if best_by_layer[l][0] - 0.5 >= 0.9 * (top - 0.5))
    else:
        layer = min(layers, key=lambda l: abs(l - int(a.layer_spec)))
    return int(layer), float(best_by_layer[layer][1]), {"selection_auroc_at_layer": float(best_by_layer[layer][0]),
                                                        "best_layer": int(s["best_layer"]), "selection_auroc_at_best_layer": float(top)}


def build_vectors(x, mu, D, fam, bases, sd, t_i, gap_sd):
    """All intervention vectors for one item (see the module docstring). Returns [K, d] float32."""
    n = len(D)
    proj = (x - mu) @ D.T                                              # [n]
    add = np.concatenate([(ADD_SD * sd)[:, None] * D, -(ADD_SD * sd)[:, None] * D])     # +all, then -all
    abl = -proj[:, None] * D
    sgn = 1.0 if t_i == 0 else -1.0
    sh_idx = [j for j in range(n) if fam[j] in ("shuf_probe", "shuf_dom")]
    shift = np.stack([sgn * gap_sd["feat_probe"] * sd[0] * D[0], sgn * gap_sd["feat_dom"] * sd[1] * D[1]]
                     + [sgn * gap_sd["feat_probe" if fam[j] == "shuf_probe" else "feat_dom"] * sd[j] * D[j] for j in sh_idx])
    sub = np.stack([-(B.T @ (B @ (x - mu))) for B in bases])           # rank-k ablation, real then controls
    return np.concatenate([np.zeros((1, len(x)), np.float32), add, abl, shift, sub]).astype(np.float32)


def weighted_auroc(yy, ss, zz):
    num = den = 0.0
    for k in np.unique(zz):
        m = zz == k
        if len(np.unique(yy[m])) < 2:
            continue
        w = float((yy[m] == 1).sum() * (yy[m] == 0).sum())
        num += w * stats.auroc(yy[m], ss[m]); den += w
    return num / den if den else float("nan")


def place(real, ctrl, signed=False):
    """Where a real value falls among control values: fraction of controls below it (by absolute
    value unless signed), the control mean / SD / range, and the real value in control SDs."""
    ctrl = np.asarray(ctrl, float)
    a, c = (real, ctrl) if signed else (abs(real), np.abs(ctrl))
    return {"value": float(real), "n_controls": int(len(ctrl)), "percentile": float((c < a).mean()),
            "exceeds_all_controls": bool((c < a).all()), "control_mean": float(ctrl.mean()), "control_sd": float(ctrl.std()),
            "control_min": float(ctrl.min()), "control_max": float(ctrl.max()),
            "z_vs_controls": float((real - ctrl.mean()) / (ctrl.std() + 1e-12))}


def analyse(a, z, meta_all, items, t_all, layer_info):
    names, fam = [str(n) for n in z["names"]], np.array([str(f) for f in z["families"]])
    idx, out, sd = z["idx"], z["out"].astype(np.float64), z["own_sd"]
    n = len(names)
    y_all = np.array(meta_all["labels"]); ld_all = np.array(meta_all["logit_diff"])
    tm_all = np.array([i["template"] for i in items]); q_all = np.array([i["query"] for i in items])
    g_all = np.array([i["group"] for i in items])
    y, t, tm, q, g = y_all[idx], t_all[idx], tm_all[idx], q_all[idx], g_all[idx]
    base = out[:, 0]
    plus, minus = out[:, 1:1 + n], out[:, 1 + n:1 + 2 * n]
    abl = out[:, 1 + 2 * n:1 + 3 * n]
    sh_idx = [j for j in range(n) if fam[j] in ("shuf_probe", "shuf_dom")]
    shift = out[:, 1 + 3 * n:1 + 3 * n + 2 + len(sh_idx)]
    sub = out[:, 1 + 3 * n + 2 + len(sh_idx):]
    slope = (plus - minus) / (2 * ADD_SD)                              # [items, dirs], logits per natural SD
    templates = sorted(set(tm))
    fam_of = {"feat_probe": "shuf_probe", "feat_dom": "shuf_dom", "gold_probe": "shuf_probe", "gold_dom": "shuf_dom",
              "model_out": "shuf_probe"}
    J = {nm: names.index(nm) for nm in fam_of}

    def ci(fn, *arrs, m=None):
        m = np.ones(len(idx), bool) if m is None else m
        return stats.bootstrap_ci(fn, tuple(x[m] for x in arrs), g[m], n_boot=1000)

    res = {"model": a.model, "tag": a.tag, "feature": a.feature, "position": a.position, "layer_spec": a.layer_spec,
           "layer": int(z["layer"]), "n_layers": int(z["n_layers"]), "probe_C": float(z["C"]), **layer_info,
           "n_items": int(len(idx)), "n_items_by_template": {k: int((tm == k).sum()) for k in templates},
           "add_in_own_sd": ADD_SD, "rank": int(z["rank"]),
           "baseline_max_abs_diff_vs_cache": float(np.abs(base - ld_all[idx]).max()),
           "baseline_logit_diff_sd": float(base.std()),
           "class_gap_in_own_sd": {k: float(v) for k, v in zip(("feat_probe", "feat_dom"), z["gap_sd"])},
           "direction_geometry": {"own_sd_over_s": {nm: float(sd[J[nm]] / float(z["s_unit"])) for nm in J},
                                  "cos_feat_probe_vs_feat_dom": float(z["D"][0] @ z["D"][1]),
                                  "cos_feat_probe_vs_gold_probe": float(z["D"][0] @ z["D"][2]),
                                  "cos_feat_dom_vs_gold_dom": float(z["D"][1] @ z["D"][3]),
                                  "cos_feat_probe_vs_model_out": float(z["D"][0] @ z["D"][4]),
                                  "report_half_auroc_for_property": {nm: float(v) for nm, v in zip(("feat_probe", "feat_dom"), z["rep_auroc"])}}}

    # observational dependence of the answer on the property, per template (all 2400 items, cached logits)
    obs = {}
    for k in templates:
        m = tm_all == k
        gaps = [ld_all[m & (q_all == s) & (t_all == 1)].mean() - ld_all[m & (q_all == s) & (t_all == 0)].mean()
                for s in np.unique(q_all[m]) if 0 < t_all[m & (q_all == s)].mean() < 1]
        obs[k] = {"n_items": int(m.sum()), "within_string_auroc_answer_for_property": weighted_auroc(t_all[m], ld_all[m], q_all[m]),
                  "within_string_auroc_answer_for_gold": weighted_auroc(y_all[m], ld_all[m], q_all[m]),
                  "mean_logit_diff_gap_property_1_minus_0": float(np.mean(gaps)),
                  "p_valid_given_property_1": float(y_all[m & (t_all == 1)].mean()),
                  "p_valid_given_property_0": float(y_all[m & (t_all == 0)].mean())}
    res["observational_by_template"] = obs
    obs_gap = np.array([obs[k]["mean_logit_diff_gap_property_1_minus_0"] for k in templates])

    # A. slopes per natural SD, pooled and by template, each against its control family
    S = np.stack([slope[tm == k].mean(0) for k in templates])          # [templates, dirs]
    res["slope_per_own_sd"] = {}
    for nm, f in fam_of.items():
        j, cf = J[nm], fam == f
        r = {"pooled": {**ci(lambda v: float(v.mean()), slope[:, j]), "vs_matched_controls": place(slope[:, j].mean(), slope[:, cf].mean(0)),
                        "vs_isotropic": place(slope[:, j].mean(), slope[:, fam == "iso"].mean(0))},
             "matched_control_family": f, "by_template": {}}
        for ki, k in enumerate(templates):
            r["by_template"][k] = {**ci(lambda v: float(v.mean()), slope[:, j], m=tm == k),
                                   "vs_matched_controls": place(S[ki, j], S[ki, cf]),
                                   "signed_percentile_among_controls": float((S[ki, cf] < S[ki, j]).mean())}
        # does the per-template pattern of causal slopes follow the observational pattern?
        def pat(col):
            return float(np.corrcoef(col, obs_gap)[0, 1]) if np.std(col) > 0 and np.std(obs_gap) > 0 else float("nan")
        r["pattern_across_templates"] = {
            "templates": templates, "causal_slope": S[:, j].tolist(), "observational_gap": obs_gap.tolist(),
            "n_sign_agreements": int((np.sign(S[:, j]) == np.sign(obs_gap)).sum()),
            "pearson_r": place(pat(S[:, j]), [pat(S[:, c]) for c in np.where(cf)[0]], signed=True),
            "sign_agreements_controls": [int((np.sign(S[:, c]) == np.sign(obs_gap)).sum()) for c in np.where(cf)[0]],
            "range_across_templates": place(float(S[:, j].max() - S[:, j].min()), (S[:, cf].max(0) - S[:, cf].min(0)))}
        res["slope_per_own_sd"][nm] = r

    # B. ablation: how well do the answers track the gold label and the property, before and after?
    def tracking(ld, m):
        return {"auroc_gold": stats.auroc(y[m], ld[m]), "auroc_property": stats.auroc(t[m], ld[m])}

    res["ablation"] = {}
    groups_abl = {"feat_probe": (abl[:, J["feat_probe"]], abl[:, fam == "shuf_probe"]),
                  "feat_dom": (abl[:, J["feat_dom"]], abl[:, fam == "shuf_dom"]),
                  "gold_probe": (abl[:, J["gold_probe"]], abl[:, fam == "shuf_probe"]),
                  f"feat_rank{int(z['rank'])}_subspace": (sub[:, 0], sub[:, 1:])}
    for nm, (real, ctrl) in groups_abl.items():
        r = {"pooled_mean_abs_delta_logit_diff": place(float(np.abs(real - base).mean()), np.abs(ctrl - base[:, None]).mean(0)),
             "by_template": {}}
        for k in templates:
            m = tm == k
            b0 = tracking(base, m)
            r["by_template"][k] = {
                "baseline": b0,
                "auroc_gold_after": ci(lambda yy, s: stats.auroc(yy, s), y, real, m=m),
                "change_in_auroc_gold": place(stats.auroc(y[m], real[m]) - b0["auroc_gold"],
                                              [stats.auroc(y[m], ctrl[m, c]) - b0["auroc_gold"] for c in range(ctrl.shape[1])], signed=True),
                "change_in_auroc_property": place(stats.auroc(t[m], real[m]) - b0["auroc_property"],
                                                  [stats.auroc(t[m], ctrl[m, c]) - b0["auroc_property"] for c in range(ctrl.shape[1])], signed=True),
                "change_in_distance_of_auroc_property_from_half": place(
                    abs(stats.auroc(t[m], real[m]) - 0.5) - abs(b0["auroc_property"] - 0.5),
                    [abs(stats.auroc(t[m], ctrl[m, c]) - 0.5) - abs(b0["auroc_property"] - 0.5) for c in range(ctrl.shape[1])], signed=True),
                "mean_abs_delta_logit_diff": place(float(np.abs(real[m] - base[m]).mean()), np.abs(ctrl[m] - base[m, None]).mean(0))}
        res["ablation"][nm] = r

    # C. shift towards the other property class by the natural class gap
    sgn = np.where(t == 0, 1.0, -1.0)
    res["shift_to_other_class"] = {}
    for c0, nm, f in ((0, "feat_probe", "shuf_probe"), (1, "feat_dom", "shuf_dom")):
        real = shift[:, c0]
        ctrl = shift[:, 2:][:, [i for i, j in enumerate(sh_idx) if fam[j] == f]]
        eff = sgn * (real - base)                                      # > 0: answer moves towards Yes when pushed towards property = 1
        ceff = sgn[:, None] * (ctrl - base[:, None])
        r = {"pooled_mean_effect_towards_yes_per_push_towards_property": {**ci(lambda v: float(v.mean()), eff),
                                                                         "vs_matched_controls": place(eff.mean(), ceff.mean(0), signed=True)},
             "by_template": {}}
        for k in templates:
            m = tm == k
            r["by_template"][k] = {
                **ci(lambda v: float(v.mean()), eff, m=m), "vs_matched_controls": place(eff[m].mean(), ceff[m].mean(0), signed=True),
                "auroc_property_baseline": stats.auroc(t[m], base[m]), "auroc_property_after_shift": stats.auroc(t[m], real[m]),
                "auroc_property_after_shift_controls_mean": float(np.mean([stats.auroc(t[m], ctrl[m, c]) for c in range(ctrl.shape[1])])),
                "observational_gap": obs[k]["mean_logit_diff_gap_property_1_minus_0"]}
        res["shift_to_other_class"][nm] = r
    res["minutes_gpu"] = float(z["minutes"])
    (RESULTS / f"feature_causal_{a.model}_{a.tag}_{a.site}.json").write_text(json.dumps(res, indent=1))
    p = res["slope_per_own_sd"]["feat_probe"]; d = res["slope_per_own_sd"]["feat_dom"]
    print(f"{a.model}/{a.tag} {a.site}: layer {res['layer']}/{res['n_layers']}  property AUROC (report half) "
          f"{res['direction_geometry']['report_half_auroc_for_property']['feat_probe']:.3f}")
    for nm, r in (("feat_probe", p), ("feat_dom", d), ("gold_probe", res["slope_per_own_sd"]["gold_probe"]), ("model_out", res["slope_per_own_sd"]["model_out"])):
        print(f"  {nm:11s} pooled slope/SD {r['pooled']['est']:+.4f} (controls pct {r['pooled']['vs_matched_controls']['percentile']:.2f}) | by template "
              + " ".join(f"{k[:10]} {r['by_template'][k]['est']:+.3f}" for k in templates)
              + f" | pattern r {r['pattern_across_templates']['pearson_r']['value']:+.2f} (pct {r['pattern_across_templates']['pearson_r']['percentile']:.2f}), "
                f"signs {r['pattern_across_templates']['n_sign_agreements']}/{len(templates)}")
    print("  observational gap by template: " + " ".join(f"{k[:10]} {v:+.3f}" for k, v in zip(templates, obs_gap)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", default="c2c_debiased", choices=["c2c_debiased", "c2c_probe"])
    ap.add_argument("--feature", default="pair_adjacent_in_skeleton", choices=list(PROPS))
    ap.add_argument("--position", default="final", choices=["final", "content_end"])
    ap.add_argument("--layer-spec", default="best", help="best | first90 | <cache layer index>")
    ap.add_argument("--per-template", type=int, default=120)
    ap.add_argument("--n-shuf", type=int, default=20)
    ap.add_argument("--n-iso", type=int, default=20)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--sub-batch", type=int, default=16)
    ap.add_argument("--select-stride", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=85.0)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    a.site = f"{SHORT[a.feature]}_{'hyp' if a.position == 'content_end' else 'final'}_{a.layer_spec}"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    resid, meta, items = cache.load_cache(a.model, a.tag)
    pi = meta["positions"].index(a.position)
    F = probes.structural_features(items)
    t = np.asarray(PROPS[a.feature](F)).astype(int)
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"])
    tm = np.array([i["template"] for i in items])
    sel = data.selection_mask(items); rep = ~sel
    layer, C, layer_info = choose_layer(a, resid, items, sel, t, pi, log)
    raw_path = cache.cache_dir(a.model, a.tag) / f"steering_feature_{a.site}.npz"
    if a.analyse_only:
        analyse(a, np.load(raw_path, allow_pickle=True), meta, items, t, layer_info)
        return
    log(f"site {a.site}: layer {layer}, C {C:g}, selection-half AUROC {layer_info['selection_auroc_at_layer']:.3f}")
    X = probes.layer_features(resid, layer, pi)
    mu = X[sel].mean(0)
    s_unit = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
    mcal = (ld0 > np.median(ld0)).astype(int)
    names, fam, D, bases = fit_directions(X, sel, t, y, mcal, tm, C, a.n_shuf, a.n_iso, a.rank, log)
    P = (X[sel] - mu) @ D.T
    sd = P.std(0).astype(np.float32)
    Pc = template_centre(P, tm[sel], np.ones(int(sel.sum()), bool))
    gap_sd = {nm: float((Pc[t[sel] == 1, j].mean() - Pc[t[sel] == 0, j].mean()) / sd[j]) for j, nm in enumerate(names[:2])}
    rep_auroc = [stats.auroc(t[rep], (X[rep] - mu) @ D[j]) for j in (0, 1)]
    log(f"{len(D)} directions; class gap along feat_probe {gap_sd['feat_probe']:.2f} own SDs, feat_dom {gap_sd['feat_dom']:.2f}; "
        f"report-half AUROC for the property {rep_auroc[0]:.3f} / {rep_auroc[1]:.3f}")

    rng = np.random.default_rng(SEED + 17)
    # evaluation items in round-robin order over templates, so that any prefix is balanced over
    # templates (the run is shortened to a prefix if the timing check says it would not fit)
    per = [rng.permutation(np.where(rep & (tm == k))[0])[: a.per_template] for k in sorted(set(tm))]
    idx = np.array([p[r] for r in range(max(len(p) for p in per)) for p in per if r < len(p)])
    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    K = build_vectors(X[idx[0]], mu, D, fam, bases, sd, int(t[idx[0]]), gap_sd).shape[0]
    out = np.zeros((len(idx), K), np.float32)
    ckpt = cache.cache_dir(a.model, a.tag) / f"steering_feature_{a.site}_partial.npz"
    start = 0
    if ckpt.exists():
        zc = np.load(ckpt)
        m_ = len(zc["idx"])          # a checkpoint of a shortened run holds a prefix of the item list
        if zc["out"].shape[1] == K and np.array_equal(zc["idx"], idx[:m_]) and np.allclose(zc["D"], D, atol=1e-6):
            idx, out = idx[:m_], out[:m_]
            start = int(zc["done"]); out[:start] = zc["out"][:start]
            log(f"resuming at item {start}/{len(idx)}")
    t_loop = time.time()
    for n_, i in enumerate(idx):
        if n_ < start:
            continue
        if n_ >= len(out):          # the run was shortened by the timing check
            break
        enc = prompts.encode_with_positions(tok, items[i])
        vecs = torch.tensor(build_vectors(X[i], mu, D, fam, bases, sd, int(t[i]), gap_sd))
        out[n_] = steer.steered_logit_diffs(model, enc["input_ids"], layer, vecs, ans, sub_batch=a.sub_batch, pos=enc["pos"][a.position])
        if n_ - start == 4:
            per_item = (time.time() - t_loop) / 5
            proj = per_item * (len(idx) - n_ - 1) / 60
            log(f"[timing] {per_item:.2f} s/item ({K} vectors) -> projected {proj:.1f} min for the remaining {len(idx) - n_ - 1} items")
            if proj > a.max_minutes and start == 0:
                keep = int(a.max_minutes * 60 / per_item)
                if keep < 100:
                    raise SystemExit(f"projected {proj:.0f} min > --max-minutes {a.max_minutes:.0f} and fewer than 100 items would fit")
                log(f"[timing] shortening the run to the first {keep} items (balanced over templates) to fit {a.max_minutes:.0f} min")
                idx, out = idx[:keep], out[:keep]
        if (n_ + 1) % 50 == 0:
            np.savez(ckpt, idx=idx, out=out, done=n_ + 1, D=D)
            log(f"{n_ + 1}/{len(idx)}")
        if model.device.type == "mps" and n_ % 5 == 0:
            torch.mps.empty_cache()
    np.savez(raw_path, idx=idx, out=out, names=names, families=fam, D=D, own_sd=sd, layer=layer, n_layers=resid.shape[1] - 1,
             C=C, s_unit=s_unit, rank=a.rank, gap_sd=[gap_sd["feat_probe"], gap_sd["feat_dom"]], rep_auroc=rep_auroc,
             minutes=(time.time() - t0) / 60)
    ckpt.unlink(missing_ok=True)
    del model
    analyse(a, np.load(raw_path, allow_pickle=True), meta, items, t, layer_info)


if __name__ == "__main__":
    main()

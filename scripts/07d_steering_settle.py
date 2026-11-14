"""One steering arm that stood out among 20 controls: is it a property of the gold label, of adjacency, or of one draw?
Needs the model (GPU). Debiased pool, last token of the hypothesis. Writes results/steering_settle_<model>_c2c_debiased_hyp.json.

Background (eighth audit, notes/audit_round2_runs_raw.md, Q1). In the 20-control runs two gold-label directions at the
end of the hypothesis ranked first among 21 in absolute slope: Qwen 3B, difference of class means (+0.74 per s), and
Qwen 1.5B, probe weights (-0.25 per s). Each is one of 12 such comparisons on this pool (24 in all), z about 2 to 3
against its controls, not first in any neighbouring arm. The audit found the existing data cannot decide between a
chance top rank and a modest real effect, that 100 controls instead of 20 would only refine one percentile of the same
draw, and named what would decide it. This script runs those things, at the same site:

  A. independent draws of the direction. The 25 blocks of the pool are split in halves in several ways; the direction
     and its controls are fitted on one half and steered on items of the other. Split 0 is the original one (it
     reproduces the 20-control run and extends it to --n-shuf controls). Split 1 is its mirror image: fitted on the
     half that was only ever used for evaluation, steered on the half that was only ever used for fitting. Splits 2..
     are random halvings (they overlap with both, and with each other).
  B. controls that keep adjacency. Within a hypothesis string the gold label is associated with whether the queried
     pair is adjacent in the graph skeleton. `within_string` controls shuffle the gold label inside each hypothesis
     string (as before); `within_cell` controls shuffle it inside each (hypothesis string x adjacency) cell, so they
     keep exactly the association of the label with adjacency and destroy the rest. A gold direction that exceeds the
     first family but not the second acts through adjacency.
  C. the adjacency direction itself: the difference of class means of adjacency among "X directly causes Y" items, and
     pooled over all items, against the same for adjacency shuffled within hypothesis string.
  D. a profile over layers: the same comparison two layers below and above the site.
  E. size at natural scale: smaller steps (0.1 s and 0.3 s) for the gold directions and 20 controls, and the slope times
     the distance between the class means along the direction.

PRIMARY TEST, fixed before the run (notes/log.md, Sat 14 Nov): split 1, the site's layer, the direction named by --arm
(the one that stood out), against its --n-shuf within-string controls, one-sided in the sign it had in the 20-control
run, per unit of s. p = (1 + number of controls at least as extreme in that sign) / (1 + number of controls).
Everything else in the output is descriptive and is labelled so.

Slopes: (LD(+a) - LD(-a)) / (2a) per item, a in units of s (root mean squared distance of the fitting half's
activations from their mean), averaged over the steered items; LD = Yes-No answer logit difference at the final token.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from causal_probe import cache, data, models, probes, prompts, steer
from causal_probe.config import ALL_MODELS, RESULTS, SEED

TAG, POSITION = "c2c_debiased", "content_end"
SMALL_ALPHAS = (0.1, 0.3)
N_ISO, N_SMALL = 20, 20


def probe_dir(X, y, C, max_iter=2000):
    """Unit direction of a standardised logistic probe in raw activation space, and whether the fit converged.
    (steer.probe_direction stops at 300 iterations; at C = 1 that left 21 of 22 fits unconverged in the 20-control runs.)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X), y)
    return steer.unit(clf.coef_[0] / sc.scale_), bool(clf.n_iter_[0] < max_iter)


def dom(X, lab):
    return steer.unit(X[lab == 1].mean(0) - X[lab == 0].mean(0))


def shuffle_within(lab, cells, rng):
    """Permute `lab` inside every cell (cells: array of hashable cell ids, same length)."""
    out = lab.copy()
    for c in np.unique(cells):
        m = np.where(cells == c)[0]
        out[m] = lab[rng.permutation(m)]
    return out


def block_splits(items, n_splits):
    """Fitting-half masks. 0: the project's selection half. 1: its complement. 2..: seeded random halves of the blocks."""
    blk = np.array([i["block"] for i in items])
    sel = data.selection_mask(items)
    out = [sel, ~sel]
    ub = np.unique(blk)
    k = 0
    while len(out) < n_splits:
        fit_blocks = set(np.random.default_rng(SEED + 900 + k).permutation(ub)[: len(ub) // 2].tolist())
        m = np.array([b in fit_blocks for b in blk])
        k += 1
        if any((m == o).all() or (m == ~o).all() for o in out):
            continue
        out.append(m)
    return out[:n_splits]


def build(X, F, y, t, strs, tm, mcal, C, k, n_ws, n_wc, n_adj, full, with_adjacency, arm, n_other):
    """Unit directions for one (split, layer), fitted on the fitting half F. Returns families, D [n, d], alphas, info.
    n_ws / n_wc: number of within-string / within-cell labellings for the family that matches `arm` (the other kind of
    direction gets n_other of the same labellings, the first ones)."""
    Xf, yf, tf, sf, tmf = X[F], y[F], t[F], strs[F], tm[F]
    fam, D, conv = [], [], {}

    def add(f, v):
        fam.append(f); D.append(np.asarray(v, np.float64))

    # the same generator and order of draws as 07c_steering_controls.py for split 0, so that its first 20 labellings
    # are those of the 20-control run: 100 isotropic draws first (the first N_ISO are kept), then the shuffles
    rng = np.random.default_rng(SEED + 101 + 1000 * k)
    iso = [rng.standard_normal(X.shape[1]) for _ in range(100)]
    add("gold_diffmeans", dom(Xf, yf))
    g, c_ = probe_dir(Xf, yf, C); add("gold_probe", g); conv["gold_probe"] = c_
    m, c_ = probe_dir(Xf, mcal[F], C); add("model_out", m); conv["model_out"] = c_
    if full:
        for v in iso[:N_ISO]:
            add("isotropic", steer.unit(v))
    n_conv = n_fit = 0
    n_pr, n_dm = (n_ws, min(n_ws, n_other)) if arm == "gold_probe" else (min(n_ws, n_other), n_ws)
    for i in range(n_ws):
        ys = shuffle_within(yf, sf, rng)            # every labelling is drawn, so the sequence does not depend on the arm
        if i < n_pr:
            p, c_ = probe_dir(Xf, ys, C); n_conv += c_; n_fit += 1
            add("within_string_probe", p)
        if i < n_dm:
            add("within_string_diffmeans", dom(Xf, ys))
    rng_c = np.random.default_rng(SEED + 202 + 1000 * k)
    cell = np.array([f"{a}|{b}" for a, b in zip(sf, tf)])
    for i in range(n_wc):
        yc = shuffle_within(yf, cell, rng_c)
        if arm == "gold_diffmeans" or i < n_other:
            add("within_cell_diffmeans", dom(Xf, yc))
        if arm == "gold_probe":
            p, c_ = probe_dir(Xf, yc, C); n_conv += c_; n_fit += 1
            add("within_cell_probe", p)
    conv["control_probes_share_converged"] = n_conv / max(n_fit, 1)
    if with_adjacency:
        par = tmf == "parent"
        add("adjacency_parent_diffmeans", dom(Xf[par], tf[par]))
        add("adjacency_pooled_diffmeans", dom(Xf, tf))
        rng_a = np.random.default_rng(SEED + 303 + 1000 * k)
        for _ in range(n_adj):
            ts = shuffle_within(tf, sf, rng_a)
            add("adjacency_parent_shuffled", dom(Xf[par], ts[par]))
            add("adjacency_pooled_shuffled", dom(Xf, ts))
    fam = np.array(fam); D = np.stack(D)
    alphas = [(j, 1.0) for j in range(len(D))]
    if full:                                           # smaller steps for the two gold directions and the first controls
        small = [j for j, f in enumerate(fam) if f in ("gold_diffmeans", "gold_probe")] + \
                [j for j, f in enumerate(fam) if f == "within_string_diffmeans"][:N_SMALL] + \
                [j for j, f in enumerate(fam) if f == "within_string_probe"][:N_SMALL]
        alphas += [(j, a) for a in SMALL_ALPHAS for j in small]
    return fam, D.astype(np.float32), alphas, conv


def rank_stats(real, ctrl):
    """Where a real slope stands among its controls. signed_rank: 1 = most positive among 1 + n."""
    ctrl = np.asarray(ctrl, float)
    return {"slope": float(real), "n_controls": int(len(ctrl)), "control_mean": float(ctrl.mean()), "control_sd": float(ctrl.std(ddof=1)),
            "control_min": float(ctrl.min()), "control_max": float(ctrl.max()), "control_max_abs": float(np.abs(ctrl).max()),
            "z_vs_controls": float((real - ctrl.mean()) / ctrl.std(ddof=1)),
            "signed_rank_1_is_most_positive": int(1 + (ctrl > real).sum()),
            "abs_rank_1_is_largest": int(1 + (np.abs(ctrl) > abs(real)).sum()),
            "share_of_controls_smaller_in_absolute_value": float((np.abs(ctrl) < abs(real)).mean())}


def one_sided_p(real, ctrl, sign):
    ctrl = np.asarray(ctrl, float)
    return float((1 + (sign * ctrl >= sign * real).sum()) / (1 + len(ctrl)))


MATCH = {"gold_diffmeans": ("within_string_diffmeans", "within_cell_diffmeans"), "gold_probe": ("within_string_probe", "within_cell_probe"),
         "model_out": ("within_string_probe",), "adjacency_parent_diffmeans": ("adjacency_parent_shuffled",),
         "adjacency_pooled_diffmeans": ("adjacency_pooled_shuffled",)}


def summarise(fam, slopes, own_sd, gap_s, small, tm_items, slope_items):
    """Descriptive summary of one (split, layer)."""
    out = {"real": {}, "families": {}}
    for f in sorted(set(fam.tolist())):
        v = slopes[fam == f]
        if len(v) > 1:
            out["families"][f] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                                  "median_abs": float(np.median(np.abs(v))), "median_own_sd_over_s": float(np.median(own_sd[fam == f]))}
    for name, fams in MATCH.items():
        j = np.where(fam == name)[0]
        if not len(j):
            continue
        j = int(j[0])
        r = {"slope_per_s": float(slopes[j]), "own_sd_over_s": float(own_sd[j]), "class_gap_along_direction_over_s": float(gap_s[j]),
             "effect_of_moving_across_the_class_gap_logits": float(slopes[j] * gap_s[j]),
             "share_of_items_with_positive_slope": float((slope_items[:, j] > 0).mean()), "versus": {}}
        for f in fams:
            if (fam == f).sum() > 1:
                r["versus"][f] = rank_stats(slopes[j], slopes[fam == f])
        if tm_items is not None:
            r["by_template"] = {}
            for g in sorted(set(tm_items.tolist())):
                mk = tm_items == g
                e = {"n_items": int(mk.sum()), "slope_per_s": float(slope_items[mk, j].mean())}
                for f in fams:
                    if (fam == f).sum() > 1:
                        cg = slope_items[mk][:, fam == f].mean(0)
                        e[f"signed_rank_vs_{f}"] = int(1 + (cg > e["slope_per_s"]).sum())
                r["by_template"][g] = e
        if small and name in small:
            r["slope_per_s_at_smaller_steps"] = small[name]
        out["real"][name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--arm", default="gold_diffmeans", choices=["gold_diffmeans", "gold_probe"], help="the direction of the primary test")
    ap.add_argument("--n", type=int, default=150, help="steered items per split")
    ap.add_argument("--n-shuf", type=int, default=100, help="within-string controls in splits 0 and 1 (and within-cell controls in split 0)")
    ap.add_argument("--n-other", type=int, default=20, help="controls per family elsewhere")
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--layer-offsets", default="-2,2", help="extra layers for split 0, relative to the site's layer")
    ap.add_argument("--sub-batch", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=70.0)
    ap.add_argument("--allow-new-site", action="store_true", help="do not require the 20-control result (smoke tests)")
    a = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    resid, meta, items = cache.load_cache(a.model, TAG)
    pi = meta["positions"].index(POSITION)
    base_json = RESULTS / f"steering_controls_{a.model}_{TAG}_hyp.json"
    if base_json.exists():
        b = json.loads(base_json.read_text())
        layer, C, base = int(b["layer"]), float(b["probe_C"]), {k: v["slope_per_s"]["est"] for k, v in b["real"].items()}
        sign = float(np.sign(base[a.arm]))
    elif a.allow_new_site:
        layer, C, _ = steer.choose_site(a.model, TAG, resid, meta, items, POSITION, 6)
        base, sign = None, 1.0
    else:
        raise SystemExit(f"{base_json.name} is not in this checkout: this run must sit at the site of the 20-control run")
    L = resid.shape[1] - 1
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"])
    t = probes.structural_features(items)[:, 2].astype(int)
    strs = np.array([i["query"] for i in items]); tm = np.array([i["template"] for i in items])
    mcal = (ld0 > np.median(ld0)).astype(int)
    splits = block_splits(items, a.n_splits)
    offs = [int(o) for o in a.layer_offsets.split(",") if o.strip()]
    configs = [(k, layer) for k in range(len(splits))] + [(0, layer + o) for o in offs if 1 <= layer + o <= L]
    log(f"site: last token of the hypothesis, layer {layer}/{L}, C {C:g}; arm {a.arm}, expected sign {sign:+.0f}; "
        f"{len(splits)} splits, extra layers {[l for _, l in configs[len(splits):]]}")

    plan = []
    for k, l in configs:
        X = probes.layer_features(resid, l, pi)
        F = splits[k]; E = ~F
        full = (k == 0 and l == layer)
        n_ws = a.n_shuf if (k in (0, 1) and l == layer) else a.n_other
        n_wc = a.n_shuf if full else a.n_other
        fam, D, alphas, conv = build(X, F, y, t, strs, tm, mcal, C, k, n_ws, n_wc, a.n_other, full, with_adjacency=(l == layer and k in (0, 1)),
                                     arm=a.arm, n_other=a.n_other)
        mu = X[F].mean(0)
        s_unit = float(np.sqrt(((X[F] - mu) ** 2).sum(1).mean()))
        own_sd = ((X[F] - mu) @ D.T).std(0) / s_unit
        gap_s = ((X[E & (y == 1)].mean(0) - X[E & (y == 0)].mean(0)) @ D.T) / s_unit      # gold class gap on the steered half, along each direction
        # the same item sample as 07c for split 0 (same generator); other splits draw their own
        r3 = np.random.default_rng(SEED + 13 + 1000 * k)
        per = min(a.n // 2, int((E & (y == 1)).sum()), int((E & (y == 0)).sum()))
        idx = np.sort(np.concatenate([r3.choice(np.where(E & (y == c))[0], per, replace=False) for c in (1, 0)]))
        cols = np.concatenate([np.zeros((1, D.shape[1]), np.float32)] + [s * al * s_unit * D[j][None] for j, al in alphas for s in (1.0, -1.0)])
        plan.append(dict(split=k, layer=l, full=full, fam=fam, D=D, alphas=alphas, conv=conv, s_unit=s_unit, own_sd=own_sd, gap_s=gap_s, idx=idx, vecs=cols))
        log(f"split {k} layer {l}: {len(D)} directions, {len(cols)} vectors per item, {len(idx)} items; probes converged: "
            f"gold {conv['gold_probe']}, controls {100 * conv['control_probes_share_converged']:.0f}%")
    total_vi = sum(len(p["vecs"]) * len(p["idx"]) for p in plan)

    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    raw_path = cache.cache_dir(a.model, TAG) / "steering_settle_hyp.npz"
    part = cache.cache_dir(a.model, TAG) / "steering_settle_hyp_partial.npz"
    done = {}
    if part.exists():
        try:
            z = np.load(part)
            done = {k_: z[k_] for k_ in z.files}
        except Exception as e:
            log(f"partial file unreadable ({type(e).__name__}); starting again")
    t_loop, vi_done = time.time(), 0
    for ci, p in enumerate(plan):
        key = f"out_{p['split']}_{p['layer']}"
        if key in done and done[key].shape == (len(p["idx"]), len(p["vecs"])) and np.array_equal(done.get(f"idx_{p['split']}_{p['layer']}"), p["idx"]):
            p["out"] = done[key]
            log(f"split {p['split']} layer {p['layer']}: taken from the partial file")
            continue
        out = np.zeros((len(p["idx"]), len(p["vecs"])), np.float32)
        vecs = torch.tensor(p["vecs"])
        for n_, i in enumerate(p["idx"]):
            enc = prompts.encode_with_positions(tok, items[i])
            out[n_] = steer.steered_logit_diffs(model, enc["input_ids"], p["layer"], vecs, ans, sub_batch=a.sub_batch, pos=enc["pos"][POSITION])
            vi_done += len(p["vecs"])
            if ci == 0 and n_ == 5:
                per_vi = (time.time() - t_loop) / vi_done
                proj = per_vi * (total_vi - vi_done) / 60
                left = a.max_minutes - (time.time() - t0) / 60
                log(f"[timing] {1000 * per_vi:.2f} ms per vector and item -> {proj:.1f} min for the rest; {left:.1f} min left of {a.max_minutes:.0f}")
                if proj > 0.85 * left:
                    raise SystemExit("projected run time too long: lower --n-shuf or --n-splits")
        p["out"] = out
        done[key] = out; done[f"idx_{p['split']}_{p['layer']}"] = p["idx"]
        tmp = part.with_name(part.name + ".tmp.npz")
        np.savez(tmp, **done); os.replace(tmp, part)
        log(f"split {p['split']} layer {p['layer']} done; baseline max |difference| vs cache {np.abs(out[:, 0] - ld0[p['idx']]).max():.3f}")
    del model

    res = {"model": a.model, "tag": TAG, "position": POSITION, "layer": layer, "n_layers": int(L), "probe_C": C, "arm": a.arm,
           "expected_sign_from_the_20_control_run": sign, "slopes_in_the_20_control_run": base,
           "n_shuf": a.n_shuf, "n_other": a.n_other, "step_in_s": 1.0, "smaller_steps_in_s": list(SMALL_ALPHAS), "configs": []}
    raw = {}
    for p in plan:
        out, fam = p["out"].astype(np.float64), p["fam"]
        n1 = len(fam)
        plus, minus = out[:, 1:1 + 2 * n1:2], out[:, 2:2 + 2 * n1:2]
        slope_items = (plus - minus) / 2.0                                   # [items, directions], per s at a = 1
        slopes = slope_items.mean(0)
        small = {}
        extra = p["alphas"][n1:]
        for e_i, (j, al) in enumerate(extra):
            c0 = 1 + 2 * n1 + 2 * e_i
            small.setdefault(str(fam[j]) if fam[j] in ("gold_diffmeans", "gold_probe") else f"{fam[j]}#{j}", {})[f"{al:g}"] = \
                float(((out[:, c0] - out[:, c0 + 1]) / (2 * al)).mean())
        ctrl_small = {}
        for name_, d_ in small.items():
            if "#" in name_:
                for al_, v_ in d_.items():
                    ctrl_small.setdefault(name_.split("#")[0], {}).setdefault(al_, []).append(v_)
        small_real = {k_: v_ for k_, v_ in small.items() if "#" not in k_}
        for k_, fam_c in (("gold_diffmeans", "within_string_diffmeans"), ("gold_probe", "within_string_probe")):
            if k_ in small_real and fam_c in ctrl_small:
                small_real[k_] = {al_: {"slope_per_s": v_, "signed_rank_among_first_controls": int(1 + (np.array(ctrl_small[fam_c][al_]) > v_).sum()),
                                        "n_controls": len(ctrl_small[fam_c][al_])} for al_, v_ in small_real[k_].items()}
        tm_items = tm[p["idx"]] if p["full"] else None
        s = summarise(fam, slopes, p["own_sd"], p["gap_s"], small_real, tm_items, slope_items)
        s.update(split=p["split"], layer=p["layer"], n_items=int(len(p["idx"])), s_unit=p["s_unit"], probes_converged=p["conv"],
                 fitting_half={0: "selection half (the original split)", 1: "report half (mirror image of the original split)"}.get(p["split"], "random half of the blocks"),
                 baseline_max_abs_diff_vs_cache=float(np.abs(out[:, 0] - ld0[p["idx"]]).max()))
        res["configs"].append(s)
        raw.update({f"out_{p['split']}_{p['layer']}": p["out"], f"idx_{p['split']}_{p['layer']}": p["idx"], f"fam_{p['split']}_{p['layer']}": fam,
                    f"own_sd_{p['split']}_{p['layer']}": p["own_sd"], f"alphas_{p['split']}_{p['layer']}": np.array(p["alphas"], np.float64)})
        if p["split"] == 1 and p["layer"] == layer:
            cf = {"gold_diffmeans": "within_string_diffmeans", "gold_probe": "within_string_probe"}[a.arm]
            j = int(np.where(fam == a.arm)[0][0])
            res["primary_test"] = {
                "what": f"split 1 (fitted on the report half, steered on the selection half), layer {layer}, {a.arm} against {int((fam == cf).sum())} "
                        f"{cf} controls, one-sided in the sign of the 20-control run ({sign:+.0f}), per unit of s",
                "slope": float(slopes[j]), "p_one_sided": one_sided_p(slopes[j], slopes[fam == cf], sign), **{k_: v_ for k_, v_ in rank_stats(slopes[j], slopes[fam == cf]).items() if k_ != "slope"},
                "fixed_before_the_run": True}
    # descriptive: the arm's direction across splits and layers
    cf = {"gold_diffmeans": "within_string_diffmeans", "gold_probe": "within_string_probe"}[a.arm]
    res["arm_across_splits_and_layers_descriptive"] = [
        {"split": c["split"], "layer": c["layer"], "slope": c["real"][a.arm]["slope_per_s"],
         "p_one_sided_vs_within_string": one_sided_p(c["real"][a.arm]["slope_per_s"],
                                                     raw_slopes(raw, c["split"], c["layer"], cf), sign),
         **{k_: c["real"][a.arm]["versus"][cf][k_] for k_ in ("n_controls", "z_vs_controls", "signed_rank_1_is_most_positive", "abs_rank_1_is_largest")}}
        for c in res["configs"]]
    res["minutes"] = round((time.time() - t0) / 60, 1)
    np.savez(raw_path, **raw)
    (RESULTS / f"steering_settle_{a.model}_{TAG}_hyp.json").write_text(json.dumps(res, indent=1))
    part.unlink(missing_ok=True)
    if "primary_test" in res:
        pt = res["primary_test"]
        print(f"PRIMARY TEST: {pt['what']}\n  slope {pt['slope']:+.3f}; controls mean {pt['control_mean']:+.3f}, SD {pt['control_sd']:.3f}, "
              f"range {pt['control_min']:+.3f} to {pt['control_max']:+.3f}; one-sided p = {pt['p_one_sided']:.3f}")
    for c in res["arm_across_splits_and_layers_descriptive"]:
        print(f"  split {c['split']} layer {c['layer']}: {a.arm} slope {c['slope']:+.3f}, z {c['z_vs_controls']:+.2f}, signed rank {c['signed_rank_1_is_most_positive']}"
              f"/{c['n_controls'] + 1}, one-sided p {c['p_one_sided_vs_within_string']:.3f}")
    v = res["configs"][0]["real"][a.arm]["versus"]
    ws_, wc_ = MATCH[a.arm]
    if ws_ in v and wc_ in v:
        print(f"  split 0: {a.arm} vs within-string controls z {v[ws_]['z_vs_controls']:+.2f} (n = {v[ws_]['n_controls']}); "
              f"vs controls that keep adjacency z {v[wc_]['z_vs_controls']:+.2f} (n = {v[wc_]['n_controls']})")


def raw_slopes(raw, k, l, family):
    out, fam = raw[f"out_{k}_{l}"].astype(np.float64), raw[f"fam_{k}_{l}"]
    n1 = len(fam)
    return ((out[:, 1:1 + 2 * n1:2] - out[:, 2:2 + 2 * n1:2]) / 2.0).mean(0)[fam == family]


if __name__ == "__main__":
    main()

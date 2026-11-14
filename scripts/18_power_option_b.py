"""Power of the test proposed for the 7B model ("option B"), computed BEFORE anything is built or run. CPU only.

The test (the "second question" of scripts/07d_steering_settle.py, made the primary test): at the last token of the
hypothesis, steer along the gold label's direction beyond adjacency (the label minus its mean in its (hypothesis string,
adjacency) cell, projected on the centred activations of a fitting half, with that half's adjacency direction projected
out), and compare its slope, one-sided, with 100 control directions built in the same way from labels shuffled within those
cells. "Detects" = p = (1 + controls at least as large) / 101 <= 0.05.

What is assumed (each assumption is a choice, stated so that a reviewer can attack it):
  A1  Effect size "the size of the answers' dependence on validity" (share f = 1): the answer depends on the activation at the
      site through ONE linear direction u (the true label-beyond-adjacency direction), by exactly as much as the answers
      depend on the label within cells: moving an item across the within-cell class gap along u changes the answer logit by
      g, the within-cell regression coefficient of the 7B answer logit on the label. Half that: f = 0.5.
  A2  The response is linear in the step up to the test's step of one unit of s (many natural class gaps).
  A3  u is not known. A direction fitted on the fitting half F has cosine c_F with it; with u_E the same direction fitted
      on the other half E as a stand-in, slope(d) = f * g * (u_E . d) * s_unit / gap_F(u_E), where gap_F(u_E) is the
      within-cell class gap of the F items along u_E. The stand-in's own error cancels in that ratio for the real direction.
  A4  Everything else the model does when a direction is added is label-free noise in the slope, the same for the real
      direction and its controls: normal with SD sigma. sigma = MULT x the largest SD of class-mean-difference control slopes
      at the end of the hypothesis on the debiased pool in the three Qwen models (the Gemma site does not respond);
      MULT = 2 by the user's instruction, 1 and 4 for sensitivity.
  A5  The site responds at all (not true of Gemma-2-2B's layer; no way to know for the 7B model without running it).
Not modelled: A5 failing; non-linearity; a 4-bit model that cannot be steered by the code as it is.

Writes results/power_option_b_qwen7b.json.
"""
import json

import numpy as np

from causal_probe import cache, data, probes
from causal_probe.config import RESULTS, SEED

MODEL, TAG, POS = "qwen7b-4bit", "c2c_debiased", "content_end"
N_CTRL, N_NOISE, N_SPLITS = 100, 400, 8


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cell_mean(v, cells):
    out = np.zeros(len(v))
    for c in np.unique(cells):
        out[cells == c] = v[cells == c].mean()
    return out


def shuffle_within(lab, cells, rng):
    out = lab.copy()
    for c in np.unique(cells):
        m = np.where(cells == c)[0]
        out[m] = lab[rng.permutation(m)]
    return out


def splits(items, n):
    blk = np.array([i["block"] for i in items]); sel = data.selection_mask(items)
    out = [sel, ~sel]; ub = np.unique(blk); k = 0
    while len(out) < n:
        fb = set(np.random.default_rng(SEED + 900 + k).permutation(ub)[: len(ub) // 2].tolist()); k += 1
        m = np.array([b in fb for b in blk])
        if not any((m == o).all() or (m == ~o).all() for o in out):
            out.append(m)
    return out


def directions(X, y, t, strs, rng, n_ctrl):
    """The test's real direction and controls on one half, as in 07d_steering_settle.build()."""
    Xc = X - X.mean(0)
    cell = np.array([f"{a}|{b}" for a, b in zip(strs, t)])
    ybar = cell_mean(y.astype(float), cell); tbar = cell_mean(t.astype(float), strs)
    a_dir = unit((t - tbar) @ Xc)
    perp = lambda v: unit(v - (v @ a_dir) * a_dir)       # noqa: E731
    real = perp((y - ybar) @ Xc)
    ctrl = np.stack([perp((shuffle_within(y, cell, rng) - ybar) @ Xc) for _ in range(n_ctrl)]) if n_ctrl else None
    return real, ctrl, (Xc, ybar)


def main():
    d = cache.cache_dir(MODEL, TAG)
    meta = json.loads((d / "meta.json").read_text()); items = cache.items_for(TAG, meta)
    y = np.array(meta["labels"]); ld = np.array(meta["logit_diff"], float)
    strs = np.array([i["query"] for i in items]); t = probes.structural_features(items)[:, 2].astype(int)
    resid = np.load(d / "resid.npy", mmap_mode="r"); pi = meta["positions"].index(POS); L = resid.shape[1] - 1
    cell = np.array([f"{a}|{b}" for a, b in zip(strs, t)]); ybar = cell_mean(y.astype(float), cell)
    r = y - ybar
    g = float((r * ld).sum() / (r * r).sum())                  # logits: the answers' dependence on the label within (string, adjacency) cells
    g_string = float(((y - cell_mean(y.astype(float), strs)) * ld).sum() / ((y - cell_mean(y.astype(float), strs)) ** 2).sum())
    mu = json.loads((RESULTS / "steering_controls_multiplicity.json").read_text())
    sds = {a["model"]: a["control_sd"] for a in mu["arms"] if a["tag"] == TAG and a["token"] != "final" and a["direction"] == "gold_diffmeans"
           and a["site_responds_positive_control_beyond_all_controls"]}
    sigma1 = max(sds.values())
    out = {"model": MODEL, "position": POS, "test": "gold label beyond adjacency (adjacency direction projected out) against 100 within-cell controls, one-sided p <= 0.05",
           "answers_dependence_on_label_within_string_and_adjacency_cells_logits": g, "answers_dependence_on_label_within_string_logits": g_string,
           "control_slope_sd_per_s_in_the_smaller_models_end_of_hypothesis": sds, "sigma_base": sigma1, "layers": {}}
    # What share of a behavioural dependence did a natural-scale shift at such a site carry in the smaller models? The adjacency test
    # translated items by the adjacency class-mean difference at the best end-of-hypothesis layer; "contrast / benchmark" is that share
    # for a property their answers demonstrably depend on. (Gemma-2-2B's site does not respond.)
    out["share_carried_at_the_end_of_hypothesis_in_the_smaller_models_adjacency"] = {}
    out["positive_control_slope_per_s_end_of_hypothesis"] = {a["model"]: a["positive_control_slope_per_s"] for a in mu["arms"]
                                                             if a["tag"] == TAG and a["token"] != "final" and a["direction"] == "gold_diffmeans"}
    for m_ in ("qwen0.5b", "qwen1.5b", "qwen3b"):
        f_ = RESULTS / f"adjacency_{m_}_c2c_debiased_hyp_best.json"
        if f_.exists():
            a_ = json.loads(f_.read_text())
            out["share_carried_at_the_end_of_hypothesis_in_the_smaller_models_adjacency"][m_] = a_["translation"]["contrast"]["est"] / a_["full_mediation_benchmark"]
    sp = splits(items, N_SPLITS)
    rng = np.random.default_rng(SEED + 18)
    for l in range(8, L + 1, 2):
        X = np.asarray(resid[:, l, pi, :], dtype=np.float32).astype(np.float64)
        rec = {"per_split": []}
        for k, F in enumerate(sp):
            E = ~F
            real, ctrl, (Xc_f, ybar_f) = directions(X[F], y[F], t[F], strs[F], np.random.default_rng(SEED + 202 + 1000 * k), N_CTRL)
            u_e, _, _ = directions(X[E], y[E], t[E], strs[E], rng, 0)
            s_unit = float(np.sqrt((Xc_f ** 2).sum(1).mean()))
            rf = y[F] - ybar_f
            gap_f = float((rf * (Xc_f @ u_e)).sum() / (rf * rf).sum())            # within-cell class gap of the F items along the other half's direction (raw units)
            sd_f = float((Xc_f @ u_e).std())
            Xc_e = X[E] - X[E].mean(0); celle = np.array([f"{a}|{b}" for a, b in zip(strs[E], t[E])]); re_ = y[E] - cell_mean(y[E].astype(float), celle)
            gap_e_real = float((re_ * (Xc_e @ real)).sum() / (re_ * re_).sum())   # out-of-sample class gap along the tested direction
            rec["per_split"].append({"split": k, "cos_real_with_other_half": float(real @ u_e), "cos_controls_with_other_half_sd": float((ctrl @ u_e).std()),
                                     "gap_F_along_other_half_direction_over_s": gap_f / s_unit, "gap_in_natural_sds": gap_f / sd_f,
                                     "out_of_sample_gap_along_tested_direction_over_s": gap_e_real / s_unit, "s_unit": s_unit,
                                     "slope_per_s_at_full_share": g * float(real @ u_e) * s_unit / gap_f if gap_f > 0 else None,
                                     "_ctrl": (g * (ctrl @ u_e) * s_unit / gap_f).tolist() if gap_f > 0 else None})
        ok = [s_ for s_ in rec["per_split"] if s_["slope_per_s_at_full_share"] is not None]
        rec["splits_with_positive_out_of_sample_gap"] = len(ok)
        rec["power"] = {}
        for mult in (1, 2, 4):
            for f in (1.0, 0.5, 0.25, 0.1, 0.05, 0.0):
                hits = []
                for s_ in rec["per_split"]:
                    if s_["slope_per_s_at_full_share"] is None:      # the label beyond adjacency is not linearly there on this split: the effect cannot be carried
                        hits.append(0.05); continue
                    sig = mult * sigma1
                    real_s = f * s_["slope_per_s_at_full_share"] + sig * rng.standard_normal(N_NOISE)
                    ctrl_s = f * np.array(s_["_ctrl"])[None, :] + sig * rng.standard_normal((N_NOISE, N_CTRL))
                    p = (1 + (ctrl_s >= real_s[:, None]).sum(1)) / (N_CTRL + 1)
                    hits.append(float((p <= 0.05).mean()))
                rec["power"][f"sigma_x{mult}_share_{f:g}"] = float(np.mean(hits))
        for s_ in rec["per_split"]:
            s_.pop("_ctrl")
        fs = [s_["slope_per_s_at_full_share"] for s_ in ok]
        rec["slope_per_s_at_full_share_median"] = float(np.median(fs)) if fs else None
        rec["cos_real_with_other_half_median"] = float(np.median([s_["cos_real_with_other_half"] for s_ in rec["per_split"]]))
        rec["gap_in_natural_sds_median"] = float(np.median([s_["gap_in_natural_sds"] for s_ in rec["per_split"]]))
        out["layers"][str(l)] = rec
        pw = rec["power"]
        print(f"L{l:2d} cos(real, other half) {rec['cos_real_with_other_half_median']:+.2f}  gap {rec['gap_in_natural_sds_median']:+.2f} SD  "
              f"slope at full share {rec['slope_per_s_at_full_share_median'] if fs else float('nan'):6.2f}/s | sigma x2: f=1 {pw['sigma_x2_share_1']:.2f}  f=.5 {pw['sigma_x2_share_0.5']:.2f}  "
              f"f=.25 {pw['sigma_x2_share_0.25']:.2f}  f=.1 {pw['sigma_x2_share_0.1']:.2f}  f=.05 {pw['sigma_x2_share_0.05']:.2f}  f=0 {pw['sigma_x2_share_0']:.2f} | x4: f=1 {pw['sigma_x4_share_1']:.2f} f=.5 {pw['sigma_x4_share_0.5']:.2f}", flush=True)
    out["n_layers"] = L
    keys = list(next(iter(out["layers"].values()))["power"])
    out["power_range_over_layers"] = {k: [min(v["power"][k] for v in out["layers"].values()), max(v["power"][k] for v in out["layers"].values())] for k in keys}
    out["slope_per_s_at_full_share_range_over_layers"] = [min(v["slope_per_s_at_full_share_median"] for v in out["layers"].values()),
                                                          max(v["slope_per_s_at_full_share_median"] for v in out["layers"].values())]
    (RESULTS / "power_option_b_qwen7b.json").write_text(json.dumps(out, indent=1))
    print(f"g within cells {g:+.3f} logits; within string {g_string:+.3f}; sigma base {sigma1:.3f} per s ({sds})")


if __name__ == "__main__":
    main()

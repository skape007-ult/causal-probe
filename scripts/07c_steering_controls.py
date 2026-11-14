"""Matched controls for the steering result (added after the Tier 2 audit, finding 2 and 5).
Needs the model (GPU). Writes results/steering_controls_<model>_<tag>.json.

Question. Is the effect of a steered direction on the answer larger than that of directions of the
same KIND that carry no label information? Three random directions were too few to say, and
isotropic random directions are the wrong comparison for a difference of class means (which lies
in the high-variance part of activation space) and for probe weights (which point into the
low-variance part).

Round 2 additions. `--position content_end` steers at the last token of the hypothesis instead of
the final prompt token (directions are then fitted on activations at that token, at the layer where
the gold probe is best there on the selection half; the answer is still read at the final token).
If the main steering run's JSON is not present (a fresh GPU session), the layer is chosen by the
same rule the main run uses (steer.choose_site).

Directions, all unit vectors at the same layer and token as the main steering run, fitted on
the selection half:
  real       gold_probe, gold_diffmeans, model_out               (as in 07_steering.py)
  isotropic  N_ISO random Gaussian directions                    (control: arbitrary direction)
  shuffled_probe   N_SHUF probe weight vectors fitted to shuffled gold labels
                                                                 (matched control for gold_probe)
  shuffled_diffmeans  N_SHUF differences of class means for shuffled gold labels
                                                                 (matched control for gold_diffmeans)
Measure: slope = (LD(+a) - LD(-a)) / (2a) per item at a = ALPHA (in units of s), averaged over
report-half items; LD = Yes-No answer logit difference. For each real direction we report where
|slope| falls in the distribution of |slope| of each control family (percentile and ratio to the
family's median and maximum), and the same in units of the direction's own natural SD.
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import cache, data, models, probes, prompts, stats, steer
from causal_probe.config import ALL_MODELS, RESULTS, SEED

N_ISO, N_SHUF, ALPHA = 100, 20, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", required=True, choices=["c2c_probe", "cladder", "c2c_debiased"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--sub-batch", type=int, default=16)
    ap.add_argument("--suffix", default="", help="read layer/C from steering_<model>_<tag><suffix>.json")
    ap.add_argument("--position", default="final", choices=["final", "content_end"])
    ap.add_argument("--select-stride", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=85.0)
    ap.add_argument("--n-shuf", type=int, default=N_SHUF, help="shuffled-label directions of each kind (probe, difference of means)")
    ap.add_argument("--out-suffix", default="", help="appended to the output names, e.g. _c100 for a run with 100 controls. Such a run "
                                                     "must sit at the site of the run without the suffix, whose result file must be in the checkout")
    ap.add_argument("--allow-new-site", action="store_true", help="with --out-suffix: do not require that result file (smoke tests)")
    a = ap.parse_args()
    t0 = time.time()
    resid, meta, items = cache.load_cache(a.model, a.tag)
    pi = meta["positions"].index(a.position)
    sj_path = RESULTS / f"steering_{a.model}_{a.tag}{a.suffix}.json"
    out_suffix = a.suffix + ("" if a.position == "final" else "_hyp") + a.out_suffix
    if a.position == "final" and sj_path.exists():
        sj = json.loads(sj_path.read_text())
        layer, C, how = sj["layer"], sj["probe_C"], "main steering run"
    else:
        layer, C, how = steer.choose_site(a.model, a.tag, resid, meta, items, a.position, a.select_stride)
    print(f"site: {a.position}, layer {layer}, C {C:g} ({how})", flush=True)
    if a.out_suffix and not a.allow_new_site:
        # a run with more controls must sit at the site of the run it refines (seventh audit, F8)
        base_json = RESULTS / f"steering_controls_{a.model}_{a.tag}{a.suffix}{'' if a.position == 'final' else '_hyp'}.json"
        if not base_json.exists():
            raise SystemExit(f"{base_json.name} is not in this checkout: a '{a.out_suffix}' run must use the site of the run without the suffix. "
                             "Ingest that run and re-pin the notebook first (or pass --allow-new-site).")
        if int(json.loads(base_json.read_text())["layer"]) != int(layer):
            raise SystemExit(f"site mismatch: {base_json.name} is at layer {json.loads(base_json.read_text())['layer']}, the selection here gives layer {layer}")
        print(f"same site as {base_json.name}", flush=True)
    X = probes.layer_features(resid, layer, pi)
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"])
    groups = np.array([i["group"] for i in items])
    sel = data.selection_mask(items); rep = ~sel
    mcal = (ld0 > np.median(ld0)).astype(int)
    mu = X[sel].mean(0)
    s_unit = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
    rng = np.random.default_rng(SEED + 101)
    fam, D = [], []

    def add(name, v):
        fam.append(name); D.append(steer.unit(np.asarray(v, np.float64)))

    add("gold_probe", steer.probe_direction(X[sel], y[sel], C))
    add("gold_diffmeans", steer.diff_of_means(X[sel], y[sel]))
    add("model_out", steer.probe_direction(X[sel], mcal[sel], C))
    for _ in range(N_ISO):
        add("isotropic", rng.standard_normal(X.shape[1]))
    # On the debiased pool the gold label is balanced within every hypothesis string; a shuffle over the whole
    # pool is not, so its directions carry information about which hypothesis is asked (fifth audit, F8).
    # There the labels are shuffled within hypothesis string. On the other item sets the shuffle is global, as before.
    within = a.tag == "c2c_debiased"
    strs_sel = np.array([i["query"] for i in items])[sel]

    def shuffled(lab):
        if not within:
            return rng.permutation(lab)
        out_ = lab.copy()
        for s_ in np.unique(strs_sel):
            m_ = np.where(strs_sel == s_)[0]
            out_[m_] = lab[rng.permutation(m_)]
        return out_
    for _ in range(a.n_shuf):      # the first 20 shuffles of a longer run are those of the 20-control run (same generator, same order)
        ys = shuffled(y[sel])
        add("shuffled_probe", steer.probe_direction(X[sel], ys, C))
        add("shuffled_diffmeans", steer.diff_of_means(X[sel], ys))
    D = np.stack(D).astype(np.float32)
    fam = np.array(fam)
    own_sd = ((X[sel] - mu) @ D.T).std(0) / s_unit                       # natural SD along each direction, in s
    print(f"{len(D)} directions fitted [{time.time() - t0:.0f}s]", flush=True)

    r3 = np.random.default_rng(SEED + 13)
    per = min(a.n // 2, int((rep & (y == 1)).sum()), int((rep & (y == 0)).sum()))
    idx = np.sort(np.concatenate([r3.choice(np.where(rep & (y == c))[0], per, replace=False) for c in (1, 0)]))
    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    vecs = torch.tensor(np.concatenate([np.zeros((1, D.shape[1]), np.float32), ALPHA * s_unit * D, -ALPHA * s_unit * D]))
    out = np.zeros((len(idx), len(vecs)), np.float32)
    t_loop = time.time()
    for n, i in enumerate(idx):
        enc = prompts.encode_with_positions(tok, items[i])
        out[n] = steer.steered_logit_diffs(model, enc["input_ids"], layer, vecs, ans, sub_batch=a.sub_batch,
                                           pos=enc["pos"][a.position])
        if n == 4:
            proj = (time.time() - t_loop) / 5 * (len(idx) - 5) / 60
            print(f"[timing] projected {proj:.1f} min for the remaining {len(idx) - 5} items", flush=True)
            if proj > a.max_minutes:
                raise SystemExit("projected run time too long; lower --n")
        if model.device.type == "mps" and n % 5 == 0:
            torch.mps.empty_cache()
    k = len(D)
    slope_items = (out[:, 1:1 + k] - out[:, 1 + k:]) / (2 * ALPHA)        # [items, directions], logits per s
    slope = slope_items.mean(0)
    np.savez(cache.cache_dir(a.model, a.tag) / f"steering_controls{out_suffix}.npz", idx=idx, out=out, families=fam,
             layer=layer, s_unit=s_unit, alpha=ALPHA, own_sd_over_s=own_sd)

    def dist(v):
        v = np.abs(v)
        return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(v.max()),
                "rms": float(np.sqrt((v ** 2).mean()))}

    res = {"model": a.model, "tag": a.tag, "position": a.position, "layer": int(layer), "layer_selection": how,
           "n_layers": int(resid.shape[1] - 1), "probe_C": C, "alpha": ALPHA, "s_unit": s_unit,
           "control_labels_shuffled": "within hypothesis string" if within else "over the whole item set",
           "n_shuffled_label_directions_of_each_kind": int(a.n_shuf), "n_items": int(len(idx)), "baseline_max_abs_diff_vs_cache": float(np.abs(out[:, 0] - ld0[idx]).max()),
           "families": {}, "real": {}}
    for f in ("isotropic", "shuffled_probe", "shuffled_diffmeans"):
        m = fam == f
        res["families"][f] = {"abs_slope_per_s": dist(slope[m]), "abs_slope_per_own_sd": dist(slope[m] * own_sd[m]),
                              "own_sd_over_s_median": float(np.median(own_sd[m])),
                              "frac_sign_consistent_directions": float(np.mean(np.abs((slope_items[:, m] > 0).mean(0) - 0.5) > 0.45))}
    match = {"gold_probe": "shuffled_probe", "gold_diffmeans": "shuffled_diffmeans", "model_out": "shuffled_probe"}
    for j, name in enumerate(("gold_probe", "gold_diffmeans", "model_out")):
        r = {"slope_per_s": stats.bootstrap_ci(lambda t: float(t.mean()), (slope_items[:, j],), groups[idx], n_boot=1000),
             "own_sd_over_s": float(own_sd[j]), "slope_per_own_sd": float(slope[j] * own_sd[j]),
             "frac_items_positive_slope": float((slope_items[:, j] > 0).mean()), "versus": {}}
        for f in ("isotropic", match[name]):
            m = fam == f
            r["versus"][f] = {
                "percentile_of_abs_slope_per_s": float((np.abs(slope[m]) < abs(slope[j])).mean()),
                "ratio_to_median_abs_slope_per_s": float(abs(slope[j]) / np.median(np.abs(slope[m]))),
                "ratio_to_max_abs_slope_per_s": float(abs(slope[j]) / np.abs(slope[m]).max()),
                "percentile_of_abs_slope_per_own_sd": float((np.abs(slope[m] * own_sd[m]) < abs(slope[j] * own_sd[j])).mean()),
                "ratio_to_median_abs_slope_per_own_sd": float(abs(slope[j] * own_sd[j]) / np.median(np.abs(slope[m] * own_sd[m])))}
        r["matched_control_family"] = match[name]
        res["real"][name] = r
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (RESULTS / f"steering_controls_{a.model}_{a.tag}{out_suffix}.json").write_text(json.dumps(res, indent=1))
    for name, r in res["real"].items():
        mf = r["matched_control_family"]
        print(f"{name:15s} slope {r['slope_per_s']['est']:+.3f}/s | vs isotropic: percentile {r['versus']['isotropic']['percentile_of_abs_slope_per_s']:.2f}"
              f" | vs {mf}: percentile {r['versus'][mf]['percentile_of_abs_slope_per_s']:.2f}, x{r['versus'][mf]['ratio_to_median_abs_slope_per_s']:.1f} its median")


if __name__ == "__main__":
    main()

"""Tier 2a: what does adding a decodable direction do to the answer? Steering + directional ablation.

Scope (Tier 2 audit): one layer and the final prompt token only. A null effect here shows that later
layers do not read that direction AT THIS SITE; it cannot show that the decoded information is unused.
The summary written by this script is superseded by 07b_steering_analysis.py (signed slopes, both
signs, natural scale) and 07c_steering_controls.py (matched random controls).

Writes results/steering_<model>_<tag>.json (and raw logit diffs to cache/<model>/<tag>/steering.npz).

Directions (unit vectors at the layer selected in 05_confound.py; fitted on the *selection half*
of premises/stories only, evaluated on items from the *report half*):
  gold_probe      weight vector of the gold-label logistic probe
  gold_diffmeans  mean(gold-yes activations) - mean(gold-no activations)
  model_out       probe for the model's own (median-split) answer
  random_k        3 isotropic random directions (too few, and not matched to either kind of real
                  direction: see 07c_steering_controls.py for the controls the comparison needs)
Interventions at the final prompt position:
  add  +/- alpha * s * direction, where s = RMS distance of an activation from the mean
       activation at that layer. s is a norm over ALL dimensions, not the spread along one
       direction: along a probe weight vector alpha = 1 is tens to hundreds of natural SDs
  ablate  remove the item's own component along the direction (set it to the dataset mean)
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import cache, data, models, probes, prompts, stats, steer
from causal_probe.config import ALL_MODELS, MODELS, RESULTS, SEED

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_RANDOM = 3


def quick_select(resid, y, items, sel, stride: int = 1, log=print):
    """Choose (layer, C) for the gold probe when 05_confound.py has not been run for this
    model/tag (Kaggle runs; the debiased pool). Same rule as 05_confound - best selection-half
    accuracy over the (layer, C) grid, report half never touched - but cheaper: grouped 3-fold
    cross-validation with one seed inside the selection half, optionally every `stride`-th layer."""
    idx = np.where(sel)[0]
    folds = data.grouped_folds([items[j] for j in idx], 3, SEED)      # indices into idx
    layers = sorted(set(range(1, resid.shape[1], stride)) | {resid.shape[1] - 1})
    import os
    from joblib import Parallel, delayed

    def one(l):
        X = probes.layer_features(resid, l, 0)[idx]
        return {c: sum(int(((probes.fit_predict(X[tr], y[idx][tr], X[te], C=c) > 0) == y[idx][te]).sum())
                       for tr, te in folds) / len(idx) for c in probes.PROBE_CS}

    rows = Parallel(n_jobs=int(os.environ.get("CAUSAL_PROBE_JOBS", "2")))(delayed(one)(l) for l in layers)
    acc = {(l, c): v for l, r in zip(layers, rows) for c, v in r.items()}
    for l in layers:
        log(f"[select] layer {l}: " + " ".join(f"C={c:g} {acc[(l, c)]:.3f}" for c in probes.PROBE_CS))
    (l, c) = max(acc, key=acc.get)
    return int(l), float(c), {f"{k[0]}|{k[1]:g}": v for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", required=True, choices=["c2c_probe", "cladder", "c2c_debiased"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layer", type=int, default=None, help="cache layer index; default: the selected layer")
    ap.add_argument("--rel-depth", type=float, default=None, help="alternative to --layer: round(depth * n_layers)")
    ap.add_argument("--suffix", default="", help="appended to output names (layer-sensitivity runs)")
    ap.add_argument("--sub-batch", type=int, default=16,
                    help="steering vectors per forward pass. 79 vectors per item; 16 stalled Gemma on the 8 GB Mac")
    ap.add_argument("--select-stride", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=85.0,
                    help="abort after the 5-item timing check if the projected run time exceeds this")
    a = ap.parse_args()
    t0 = time.time()
    resid, meta, items = cache.load_cache(a.model, a.tag)
    conf_path = RESULTS / f"confound_{a.model}_{a.tag}.json"
    sel_path = RESULTS / f"steer_select_{a.model}_{a.tag}.json"
    if conf_path.exists():
        conf = json.loads(conf_path.read_text())
        sel_layer, probe_C, selection = conf["selected_layer_gold"], conf.get("selected_C_gold", probes.PROBE_C), "05_confound"
    else:
        if not sel_path.exists():
            y_, g_ = np.array(meta["labels"]), np.array([i["group"] for i in items])
            l_, c_, grid = quick_select(resid, y_, items, data.selection_mask(items), a.select_stride)
            sel_path.write_text(json.dumps({"model": a.model, "tag": a.tag, "selected_layer_gold": l_,
                                            "selected_C_gold": c_, "stride": a.select_stride,
                                            "selection_half_cv_accuracy": grid}, indent=1))
        conf = json.loads(sel_path.read_text())
        sel_layer, probe_C, selection = conf["selected_layer_gold"], conf["selected_C_gold"], "quick_select"
    layer = sel_layer
    if a.rel_depth is not None:
        layer = int(round(a.rel_depth * (resid.shape[1] - 1)))
    if a.layer is not None:
        layer = a.layer
    X = probes.layer_features(resid, layer, 0)
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"])
    groups = np.array([i["group"] for i in items])
    # identical split to 05_confound.py
    sel = data.selection_mask(items); rep = ~sel

    mcal = (ld0 > np.median(ld0)).astype(int)
    mu = X[sel].mean(0)
    s_unit = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
    dirs = {"gold_probe": steer.probe_direction(X[sel], y[sel], probe_C),
            "gold_diffmeans": steer.diff_of_means(X[sel], y[sel]),
            "model_out": steer.probe_direction(X[sel], mcal[sel], probe_C)}
    r2 = np.random.default_rng(SEED + 11)
    for k in range(N_RANDOM):
        dirs[f"random_{k}"] = steer.unit(r2.standard_normal(X.shape[1]))
    names = list(dirs)
    D = np.stack([dirs[n] for n in names]).astype(np.float32)          # [n_dirs, d]

    # evaluation items: report half, balanced by gold label
    r3 = np.random.default_rng(SEED + 13)
    per = min(a.n // 2, int((rep & (y == 1)).sum()), int((rep & (y == 0)).sum()))
    idx = np.concatenate([r3.choice(np.where(rep & (y == c))[0], per, replace=False) for c in (1, 0)])
    idx.sort()

    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    # vectors per item: [baseline] + [dir x alpha x sign] + [ablate per dir]
    add = np.stack([sg * al * s_unit * D[d] for d in range(len(names)) for al in ALPHAS for sg in (1, -1)])
    out = np.zeros((len(idx), 1 + len(add) + len(names)), np.float32)
    # checkpoint: raw logit differences every 25 items; a re-run resumes if the setup is identical
    ckpt = cache.cache_dir(a.model, a.tag) / f"steering{a.suffix}_partial.npz"
    start = 0
    if ckpt.exists():
        z = np.load(ckpt)
        same_dirs = "D" in z.files and z["D"].shape == D.shape and np.allclose(z["D"], D, atol=1e-6) and float(z["s_unit"]) == s_unit
        if int(z["layer"]) == layer and np.array_equal(z["idx"], idx) and z["out"].shape == out.shape and same_dirs:
            start = int(z["done"]); out[:start] = z["out"][:start]
            print(f"resuming at item {start}/{len(idx)}", flush=True)
    t_loop = time.time()
    for n, i in enumerate(idx):
        if n < start:
            continue
        x = X[i]
        abl = np.stack([-((x - mu) @ D[d]) * D[d] for d in range(len(names))])
        vecs = torch.tensor(np.concatenate([np.zeros((1, X.shape[1]), np.float32), add, abl]))
        ids = prompts.encode_with_positions(tok, items[i])["input_ids"]
        out[n] = steer.steered_logit_diffs(model, ids, layer, vecs, ans, sub_batch=a.sub_batch)
        if n - start == 4:      # timing check on the first 5 items before committing to the full run
            per_item = (time.time() - t_loop) / 5
            proj = per_item * (len(idx) - n - 1) / 60
            print(f"[timing] {per_item:.2f} s/item over 5 items -> projected {proj:.1f} min for the remaining "
                  f"{len(idx) - n - 1} items", flush=True)
            if proj > a.max_minutes:
                raise SystemExit(f"projected {proj:.0f} min > --max-minutes {a.max_minutes:.0f}; "
                                 f"lower --sub-batch or --n and re-run")
        if (n + 1) % 25 == 0:
            np.savez(ckpt, idx=idx, out=out, done=n + 1, layer=layer, D=D, s_unit=s_unit)
            print(f"{n + 1}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
        if model.device.type == "mps" and n % 10 == 0:
            torch.mps.empty_cache()
    np.savez(cache.cache_dir(a.model, a.tag) / f"steering{a.suffix}.npz", idx=idx, out=out, names=names,
             alphas=ALPHAS, layer=layer, s_unit=s_unit, D=D, probe_C=probe_C)
    ckpt.unlink(missing_ok=True)

    # ---------------- summarise
    yy, gg = y[idx], groups[idx]
    base = out[:, 0]
    bpred = (base > 0).astype(int)
    base_err = float(np.abs(base - ld0[idx]).max())
    if base_err >= 0.5:      # recorded, not fatal: a long GPU run should not be lost to this check
        print(f"WARNING: unsteered baseline differs from the cached logit difference by up to {base_err:.2f}", flush=True)
    A = out[:, 1:1 + len(add)].reshape(len(idx), len(names), len(ALPHAS), 2)   # [..., 0]=+ , 1 = -
    ABL = out[:, 1 + len(add):]

    def ci(fn, *arrs):
        return stats.bootstrap_ci(fn, arrs, gg, n_boot=1000)

    def summarise_dir(Ad, abl):
        res = {"add": [], "ablate": {
            "flip_rate": ci(lambda b, x: float(np.mean((x > 0) != (b > 0))), base, abl),
            "accuracy": ci(lambda t, x: stats.accuracy(t, (x > 0).astype(int)), yy, abl),
            "mean_abs_delta_logit_diff": float(np.abs(abl - base).mean()),
            "mean_delta_logit_diff": float((abl - base).mean())}}
        for ai, al in enumerate(ALPHAS):
            plus, minus = Ad[:, ai, 0], Ad[:, ai, 1]
            toward = np.where(yy == 1, plus, minus)       # push towards the gold answer
            away = np.where(yy == 1, minus, plus)
            res["add"].append({
                "alpha": al,
                "accuracy_toward_gold": ci(lambda t, x: stats.accuracy(t, (x > 0).astype(int)), yy, toward),
                "accuracy_away_from_gold": ci(lambda t, x: stats.accuracy(t, (x > 0).astype(int)), yy, away),
                "flip_rate_plus": ci(lambda b, x: float(np.mean((x > 0) != (b > 0))), base, plus),
                "flip_rate_minus": ci(lambda b, x: float(np.mean((x > 0) != (b > 0))), base, minus),
                "frac_yes_plus": float(np.mean(plus > 0)), "frac_yes_minus": float(np.mean(minus > 0)),
                "mean_delta_logit_diff_plus": float((plus - base).mean()),
                "mean_delta_logit_diff_minus": float((minus - base).mean())})
        return res

    res = {"model": a.model, "tag": a.tag, "layer": int(layer), "n_items": int(len(idx)),
           "n_layers": int(resid.shape[1] - 1), "selected_layer_gold": int(sel_layer), "layer_selection": selection,
           "device": str(model.device), "sub_batch": a.sub_batch,
           "alphas": ALPHAS, "s_unit_rms_deviation": s_unit, "probe_C": probe_C,
           "mean_resid_norm": float(np.linalg.norm(X[sel], axis=1).mean()),
           "baseline": {"accuracy": ci(lambda t, b: stats.accuracy(t, (b > 0).astype(int)), yy, base),
                        "frac_yes": float(bpred.mean()), "mean_logit_diff": float(base.mean()),
                        "max_abs_diff_vs_cache": base_err, "baseline_matches_cache": bool(base_err < 0.5)},
           "direction_geometry": {
               "cos_gold_probe_vs_diffmeans": float(D[0] @ D[1]),
               "cos_gold_probe_vs_model_out": float(D[0] @ D[2]),
               "cos_diffmeans_vs_model_out": float(D[1] @ D[2]),
               "class_mean_separation_over_s_unit": float(np.linalg.norm(X[sel][y[sel] == 1].mean(0) - X[sel][y[sel] == 0].mean(0)) / s_unit),
               "proj_std_over_s_unit": {n: float(((X[sel] - mu) @ D[k]).std() / s_unit) for k, n in enumerate(names)}},
           "directions": {}}
    for k, n in enumerate(names[:3]):
        res["directions"][n] = summarise_dir(A[:, k], ABL[:, k])
    # random: average the per-direction statistics over the random directions
    rs = [summarise_dir(A[:, k], ABL[:, k]) for k in range(3, len(names))]
    res["directions"]["random_each"] = rs
    res["directions"]["random_mean"] = {
        "add": [{"alpha": al,
                 **{key: float(np.mean([r["add"][ai][key]["est"] for r in rs]))
                    for key in ("accuracy_toward_gold", "accuracy_away_from_gold", "flip_rate_plus", "flip_rate_minus")},
                 "mean_abs_delta_logit_diff": float(np.mean([abs(r["add"][ai]["mean_delta_logit_diff_plus"]) for r in rs]))}
                for ai, al in enumerate(ALPHAS)],
        "ablate": {"flip_rate": float(np.mean([r["ablate"]["flip_rate"]["est"] for r in rs])),
                   "mean_abs_delta_logit_diff": float(np.mean([r["ablate"]["mean_abs_delta_logit_diff"] for r in rs]))}}
    (RESULTS / f"steering_{a.model}_{a.tag}{a.suffix}.json").write_text(json.dumps(res, indent=1))
    print(f"{a.model}/{a.tag} layer {layer} base acc {res['baseline']['accuracy']['est']:.3f}  [{time.time() - t0:.0f}s]")
    for n in names[:3] + ["random_mean"]:
        row = res["directions"][n]["add"]
        get = (lambda r, k: r[k]["est"]) if n != "random_mean" else (lambda r, k: r[k])
        print(f"  {n:15s} toward-gold acc by alpha: " + " ".join(f"{get(r, 'accuracy_toward_gold'):.2f}" for r in row)
              + " | flip+ " + " ".join(f"{get(r, 'flip_rate_plus'):.2f}" for r in row))


if __name__ == "__main__":
    main()

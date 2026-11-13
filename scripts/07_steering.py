"""Tier 2a: is the probed direction causally used? Steering + directional ablation.

Writes results/steering_<model>_<tag>.json (and raw logit diffs to cache/<model>/<tag>/steering.npz).

Directions (unit vectors at the layer selected in 05_confound.py; fitted on the *selection half*
of premises/stories only, evaluated on items from the *report half*):
  gold_probe      weight vector of the gold-label logistic probe
  gold_diffmeans  mean(gold-yes activations) - mean(gold-no activations)
  model_out       probe for the model's own (median-split) answer
  random_k        3 random directions (control: same norm, no meaning)
Interventions at the final prompt position:
  add  +/- alpha * s * direction, where s = RMS distance of an activation from the mean
       activation at that layer (so alpha = 1 is "one typical item-to-item deviation")
  ablate  remove the item's own component along the direction (set it to the dataset mean)
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import cache, models, probes, prompts, stats, steer
from causal_probe.config import ALL_MODELS, MODELS, RESULTS, SEED

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_RANDOM = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--tag", required=True, choices=["c2c_probe", "cladder"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    conf = json.loads((RESULTS / f"confound_{a.model}_{a.tag}.json").read_text())
    layer = a.layer if a.layer is not None else conf["selected_layer_gold"]
    probe_C = conf.get("selected_C_gold", probes.PROBE_C)
    resid, meta, items = cache.load_cache(a.model, a.tag)
    X = probes.layer_features(resid, layer, 0)
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"])
    groups = np.array([i["group"] for i in items])
    # identical split to 05_confound.py
    ug = np.unique(groups)
    rng = np.random.default_rng(SEED + 7)
    sel_groups = set(rng.permutation(ug)[: len(ug) // 2])
    sel = np.array([g in sel_groups for g in groups]); rep = ~sel

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
    idx = np.concatenate([r3.choice(np.where(rep & (y == c))[0], a.n // 2, replace=False) for c in (1, 0)])
    idx.sort()

    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    # vectors per item: [baseline] + [dir x alpha x sign] + [ablate per dir]
    add = np.stack([sg * al * s_unit * D[d] for d in range(len(names)) for al in ALPHAS for sg in (1, -1)])
    out = np.zeros((len(idx), 1 + len(add) + len(names)), np.float32)
    for n, i in enumerate(idx):
        x = X[i]
        abl = np.stack([-((x - mu) @ D[d]) * D[d] for d in range(len(names))])
        vecs = torch.tensor(np.concatenate([np.zeros((1, X.shape[1]), np.float32), add, abl]))
        ids = prompts.encode_with_positions(tok, items[i])["input_ids"]
        out[n] = steer.steered_logit_diffs(model, ids, layer, vecs, ans)
        if n % 50 == 0:
            print(f"{n}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
        if model.device.type == "mps" and n % 10 == 0:
            torch.mps.empty_cache()
    np.savez(cache.cache_dir(a.model, a.tag) / "steering.npz", idx=idx, out=out, names=names,
             alphas=ALPHAS, layer=layer, s_unit=s_unit)

    # ---------------- summarise
    yy, gg = y[idx], groups[idx]
    base = out[:, 0]
    bpred = (base > 0).astype(int)
    assert np.abs(base - ld0[idx]).max() < 0.5, "baseline differs from cache"
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
           "alphas": ALPHAS, "s_unit_rms_deviation": s_unit, "probe_C": probe_C,
           "mean_resid_norm": float(np.linalg.norm(X[sel], axis=1).mean()),
           "baseline": {"accuracy": ci(lambda t, b: stats.accuracy(t, (b > 0).astype(int)), yy, base),
                        "frac_yes": float(bpred.mean()), "mean_logit_diff": float(base.mean()),
                        "max_abs_diff_vs_cache": float(np.abs(base - ld0[idx]).max())},
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
    (RESULTS / f"steering_{a.model}_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"{a.model}/{a.tag} layer {layer} base acc {res['baseline']['accuracy']['est']:.3f}  [{time.time() - t0:.0f}s]")
    for n in names[:3] + ["random_mean"]:
        row = res["directions"][n]["add"]
        get = (lambda r, k: r[k]["est"]) if n != "random_mean" else (lambda r, k: r[k])
        print(f"  {n:15s} toward-gold acc by alpha: " + " ".join(f"{get(r, 'accuracy_toward_gold'):.2f}" for r in row)
              + " | flip+ " + " ".join(f"{get(r, 'flip_rate_plus'):.2f}" for r in row))


if __name__ == "__main__":
    main()

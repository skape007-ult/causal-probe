"""Tier 2b: where does a change of variable in the prompt reach the answer position?
Activation patching on Corr2Cause minimal pairs.

Scope (Tier 2 audit, finding 14). The models do not separate these pairs by gold label (mean D is
about zero), and the effect size below is normalised by each pair's own D. So the result localises
where ANY difference between the two prompts reaches the final token, not where a causal judgement
is made. The random-donor control is uninformative as run (few pairs have a donor; finding 15).

Writes results/patching_<model>.json and cache/<model>/patching.npz (raw per-pair numbers).

Part A  residual-stream patching, clean -> corrupt, for every (layer, token region), plus a
        random-donor control (same layer/region, donor = clean run of an unrelated premise with
        the same token count).
Part B  components: attribution patching (gradient x activation difference) for every attention
        head and MLP, then real activation patching of the top-K and K random components to
        validate the approximation.

Effect size. For a pair, D = LD(clean) - LD(corrupt), where LD is the Yes-No answer logit
difference. Pooled recovery of a patch = sum_pairs sign(D) * (LD(patched) - LD(corrupt)) / sum_pairs |D|.
1 means patching makes the corrupt run behave like the clean run; 0 means no effect. Pooling
(instead of averaging per-pair ratios) avoids dividing by near-zero D for pairs the model
barely distinguishes, and does not require the model to be *right* about the pair.
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from causal_probe import cache, data, models, patch
from causal_probe.config import ALL_MODELS, MODELS, RESULTS, SEED

REGIONS = ["diff_tokens", "premise_rest", "hypothesis", "template_tail", "final"]


def pooled(delta, eff):
    """delta [n], eff [n, ...] = LD(patched) - LD(corrupt) -> pooled recovery [...]"""
    s = np.sign(delta).reshape(-1, *([1] * (eff.ndim - 1)))
    return np.nansum(s * eff, 0) / np.abs(delta).sum()


def boot_pooled(delta, eff, groups, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    mem = [np.where(inv == k)[0] for k in range(len(ug))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))])
        vals.append(pooled(delta[idx], eff[idx]))
    lo, hi = np.nanquantile(np.array(vals), [0.025, 0.975], axis=0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-comp-pairs", type=int, default=40)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-components", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model)
    ans = models.answer_token_ids(tok)
    L = len(models.get_layers(model))
    layers = sorted(set(range(0, L + 1, a.layer_stride)) | {L})
    pool = [i for i in data.load_c2c("train") if i["num_variables"] <= a.max_vars]
    res = {"model": a.model, "layers": layers, "regions": REGIONS, "max_vars": a.max_vars, "kinds": {}}
    raw = {}

    for kind in ("hyp", "prem"):
        pairs = patch.build_pairs(pool, kind, a.n_pairs * 2, seed=SEED)
        encs = [(p, patch.encode_pair(tok, p)) for p in pairs]
        encs = [(p, e) for p, e in encs if e is not None][: a.n_pairs]
        n = len(encs)
        ld_clean, ld_cor = np.zeros(n), np.zeros(n)

        def clean_run(k):
            """Clean forward pass of pair k -> (logit diff, residual stream [L+1, T, d]).
            Recomputed on demand instead of stored: 60 pairs x 16-45 MB each would not fit
            next to the model in 8 GB."""
            rec = patch.ResidRecorder(model)
            try:
                ld = patch.run_logit_diff(model, encs[k][1]["clean_ids"], ans)[0]
                return ld, rec.pop()
            finally:
                rec.remove()

        with torch.no_grad():
            for k, (p, e) in enumerate(encs):
                ld_cor[k] = patch.run_logit_diff(model, e["corrupt_ids"], ans)[0]
        groups = np.array([p["group"] for p, _ in encs])
        grid = np.full((n, len(layers), len(REGIONS)), np.nan, np.float32)
        ctrl = np.full_like(grid, np.nan)
        rng = np.random.default_rng(SEED + 3)
        for k, (p, e) in enumerate(encs):
            ld_clean[k], cr = clean_run(k)
            grid[k] = patch.resid_patch_grid(model, e, ans, layers, cr, a.batch)
            del cr
            # random-donor control: clean run of a different premise with the same token count
            cands = [j for j in range(n) if groups[j] != groups[k] and encs[j][1]["n_tokens"] == e["n_tokens"]]
            if cands:
                _, dr = clean_run(int(rng.choice(cands)))
                ctrl[k] = patch.resid_patch_grid(model, e, ans, layers, dr, a.batch)
                del dr
            if k % 10 == 0:
                print(f"[{kind}] pair {k}/{n} {time.time() - t0:.0f}s", flush=True)
            if model.device.type == "mps":
                torch.mps.empty_cache()
        delta = ld_clean - ld_cor
        eff, ceff = grid - ld_cor[:, None, None], ctrl - ld_cor[:, None, None]
        has_ctrl = ~np.isnan(ctrl[:, 0, -1])
        rec_pool = pooled(delta, eff)
        lo, hi = boot_pooled(delta, eff, groups)
        ctrl_pool = pooled(delta[has_ctrl], ceff[has_ctrl]) if has_ctrl.any() else np.full_like(rec_pool, np.nan)
        from causal_probe import stats
        kres = {
            "n_pairs": n, "n_premises": int(len(set(groups))), "mean_tokens": float(np.mean([e["n_tokens"] for _, e in encs])),
            "mean_n_diff_tokens": float(np.mean([e["n_diff"] for _, e in encs])),
            "behaviour_on_pairs": {
                "mean_LD_clean": float(ld_clean.mean()), "mean_LD_corrupt": float(ld_cor.mean()),
                "mean_delta": stats.bootstrap_ci(lambda d: float(d.mean()), (delta,), groups),
                "frac_delta_positive": stats.bootstrap_ci(lambda d: float((d > 0).mean()), (delta,), groups),
                "mean_abs_delta": float(np.abs(delta).mean()),
                "frac_clean_answered_yes": float((ld_clean > 0).mean()),
                "frac_corrupt_answered_no": float((ld_cor <= 0).mean()),
                "frac_both_correct": float(((ld_clean > 0) & (ld_cor <= 0)).mean())},
            "pooled_recovery": rec_pool.tolist(), "pooled_recovery_lo": lo.tolist(), "pooled_recovery_hi": hi.tolist(),
            "control_pooled_recovery": ctrl_pool.tolist(), "n_pairs_with_control": int(has_ctrl.sum()),
            "note_regions": "premise_rest is empty for kind=hyp (premises are identical); positions before the "
                            "first differing token are identical in both runs and are not patched.",
        }
        res["kinds"][kind] = kres
        raw.update({f"{kind}_grid": grid, f"{kind}_ctrl": ctrl, f"{kind}_ld_clean": ld_clean, f"{kind}_ld_cor": ld_cor,
                    f"{kind}_groups": groups})
        print(f"[{kind}] n={n} mean delta {delta.mean():.3f} frac>0 {(delta > 0).mean():.2f}; "
              f"final-position recovery by layer: " + " ".join(f"{v:.2f}" for v in rec_pool[:, -1]), flush=True)

        # ------------------------------------------------ Part B: components
        if a.skip_components:
            continue
        m = min(a.n_comp_pairs, n)
        H = patch.n_heads(model)
        io = patch.ComponentIO(model)
        try:
            attr_z = np.zeros((m, L, H)); attr_mlp = np.zeros((m, L))
            for k in range(m):
                e = encs[k][1]
                _, ca, _ = patch.record_components(model, io, e["clean_ids"], ans, grad=False)
                _, ka, kg = patch.record_components(model, io, e["corrupt_ids"], ans, grad=True)
                for l in range(L):
                    d = (ca[("z", l)] - ka[("z", l)]) * kg[("z", l)]
                    attr_z[k, l] = d.reshape(d.shape[0], H, -1).sum((0, 2)).numpy()
                    attr_mlp[k, l] = float(((ca[("mlp", l)] - ka[("mlp", l)]) * kg[("mlp", l)]).sum())
                del ca, ka, kg
                if model.device.type == "mps":
                    torch.mps.empty_cache()
            assert np.isfinite(attr_z).all() and np.isfinite(attr_mlp).all(), "non-finite attribution"
            dm = delta[:m]
            comps = [("z", l, h) for l in range(L) for h in range(H)] + [("mlp", l, -1) for l in range(L)]
            attr = np.concatenate([attr_z.reshape(m, -1), attr_mlp], 1)             # [m, n_comp]
            attr_pool = pooled(dm, attr)
            order = np.argsort(-np.abs(attr_pool))
            top = order[: a.top_k].tolist()
            rnd = np.random.default_rng(SEED + 5).choice(order[a.top_k:], a.top_k, replace=False).tolist()
            real = np.zeros((m, len(top) + len(rnd)))
            for k in range(m):
                e = encs[k][1]
                _, ca, _ = patch.record_components(model, io, e["clean_ids"], ans, grad=False)
                for c, ci in enumerate(top + rnd):
                    typ, l, h = comps[ci]
                    spec = {("z", l): ([h], ca[("z", l)])} if typ == "z" else {("mlp", l): ca[("mlp", l)]}
                    real[k, c] = patch.patch_components(model, io, e["corrupt_ids"], ans, spec) - ld_cor[k]
            real_pool = pooled(dm, real)
            sel = top + rnd
            kres["components"] = {
                "n_pairs": m, "n_heads": H, "n_layers": L,
                "attribution_pooled_heads": attr_pool[: L * H].reshape(L, H).tolist(),
                "attribution_pooled_mlps": attr_pool[L * H:].tolist(),
                "sum_attribution_all_components": float(attr_pool.sum()),
                "validated": [{"component": f"L{comps[ci][1]}H{comps[ci][2]}" if comps[ci][0] == "z" else f"MLP{comps[ci][1]}",
                               "set": "top" if c < len(top) else "random",
                               "attribution": float(attr_pool[ci]), "real_patching": float(real_pool[c])}
                              for c, ci in enumerate(sel)],
                "pearson_attr_vs_real": float(pearsonr(attr_pool[sel], real_pool)[0]),
                "spearman_attr_vs_real": float(spearmanr(attr_pool[sel], real_pool)[0]),
                "mean_abs_real_top": float(np.abs(real_pool[: len(top)]).mean()),
                "mean_abs_real_random": float(np.abs(real_pool[len(top):]).mean()),
            }
            raw.update({f"{kind}_attr": attr, f"{kind}_real": real, f"{kind}_sel": np.array(sel)})
            print(f"[{kind}] components: top {[v['component'] for v in kres['components']['validated'][:a.top_k]]} "
                  f"pearson {kres['components']['pearson_attr_vs_real']:.2f}", flush=True)
        except Exception as ex:  # recorded, not hidden (e.g. half-precision gradients on MPS)
            kres["components"] = {"error": f"{type(ex).__name__}: {ex}"[:400]}
            print("[components] FAILED", kres["components"]["error"], flush=True)
        finally:
            io.remove()

    res["minutes"] = (time.time() - t0) / 60
    (cache.CACHE / a.model).mkdir(parents=True, exist_ok=True)
    np.savez(cache.CACHE / a.model / "patching.npz", **raw)
    (RESULTS / f"patching_{a.model}.json").write_text(json.dumps(res, indent=1))
    print(f"done {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()

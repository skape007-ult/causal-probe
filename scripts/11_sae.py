"""Tier 3a: sparse-autoencoder features for causal structure (Gemma-2-2B, Gemma Scope SAEs).

  --stage features   CPU, from the cache. Find SAE features whose activation differs between
                     CLadder causal graph types; test them on held-out stories; dump top examples.
  --stage ablate     GPU. Remove the top features' contributions from the residual stream and
                     measure the effect on the Yes-No answer logit difference, against
                     activation-matched random features.

Why only the "nonsense" stories: in CLadder every real-world story (smoking, alarm, ...) uses a
single graph type, so a feature that fires on "smoking" would look like a "confounding" feature.
The ten nonsense stories (made-up words) each occur with all graph types. Features are selected
on nonsense stories 0-4 and evaluated on nonsense stories 5-9.

SAE caveat: Gemma Scope SAEs were trained on the base model gemma-2-2b; we apply them to the
instruction-tuned gemma-2-2b-it (reconstruction quality on our activations is reported).

Writes results/sae_features.json, results/sae_ablation.json, results/sae_top_examples.md.
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import cache, models, prompts, stats
from causal_probe.config import RESULTS, SEED

MODEL = "gemma2b"
RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_LAYERS = [6, 12, 18]            # block indices; cache index = block + 1
STRUCTS = ["confounding", "collision", "chain", "fork", "mediation"]
TOP_K = 10
POS = {"premise_end": 2, "final": 0}


def load_sae(block: int, device="cpu"):
    from sae_lens import SAE
    out = SAE.from_pretrained(release=RELEASE, sae_id=f"layer_{block}/width_16k/canonical", device=device)
    sae = out[0] if isinstance(out, tuple) else out
    return sae.eval()


@torch.no_grad()
def encode(sae, X: np.ndarray, bs: int = 512) -> np.ndarray:
    out = [sae.encode(torch.tensor(X[i:i + bs], dtype=torch.float32)).numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


@torch.no_grad()
def recon_stats(sae, X: np.ndarray) -> dict:
    x = torch.tensor(X[:512], dtype=torch.float32)
    f = sae.encode(x)
    xh = sae.decode(f)
    fvu = float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    return {"fraction_variance_unexplained": fvu, "mean_l0": float((f > 0).float().sum(-1).mean()),
            "cosine_recon": float(torch.nn.functional.cosine_similarity(x, xh, dim=-1).mean())}


def auroc_cols(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every column of F for binary y (rank-based, ties averaged)."""
    from scipy.stats import rankdata
    r = rankdata(F, axis=0)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def stage_features():
    resid, meta, items = cache.load_cache(MODEL, "cladder")
    ns = np.array([i["group"].startswith("nonsense") for i in items])
    story_no = np.array([int(i["group"][-1]) if i["group"].startswith("nonsense") else -1 for i in items])
    disc, evl = ns & (story_no <= 4), ns & (story_no >= 5)
    graph = np.array([i["graph_id"] for i in items])
    groups = np.array([i["group"] for i in items])
    res = {"model": MODEL, "sae_release": RELEASE, "width": "16k", "n_nonsense_items": int(ns.sum()),
           "n_discovery": int(disc.sum()), "n_eval": int(evl.sum()), "structures": STRUCTS, "layers": {}}
    md = ["# SAE features: top-activating examples\n",
          "Features were selected on nonsense stories 0-4 (AUROC for one graph type vs the rest) and are shown with "
          "their held-out AUROC on nonsense stories 5-9. 'My reading' is a human-style label written after looking at "
          "the examples; 'unclear' is used freely.\n"]
    rng = np.random.default_rng(SEED)
    for block in SAE_LAYERS:
        sae = load_sae(block)
        lres = {"positions": {}}
        for pname, pi in POS.items():
            X = np.asarray(resid[:, block + 1, pi, :], dtype=np.float32)
            F = encode(sae, X)
            pres = {"reconstruction": recon_stats(sae, X), "n_features_ever_active": int((F[ns] > 0).any(0).sum()),
                    "structures": {}}
            alive = (F[disc] > 0).mean(0) >= 0.02            # fires on at least 2% of discovery items
            for st in STRUCTS:
                yd, ye = (graph[disc] == st).astype(int), (graph[evl] == st).astype(int)
                a_d = auroc_cols(F[disc], yd)
                a_d[~alive] = 0.5
                top = np.argsort(-a_d)[:TOP_K]
                a_e = auroc_cols(F[evl][:, top], ye)
                # null: how good is the best of 16k features on shuffled structure labels?
                null = [float(np.sort(auroc_cols(F[disc][:, alive], rng.permutation(yd)))[-TOP_K:].mean()) for _ in range(5)]
                feats = []
                for f, ad, ae in zip(top, a_d[top], a_e):
                    act = F[:, f]
                    order = np.argsort(-np.where(ns, act, -1))[:8]
                    feats.append({
                        "feature": int(f), "neuronpedia": f"gemma-2-2b/{block}-gemmascope-res-16k/{int(f)}",
                        "auroc_discovery": float(ad), "auroc_heldout_stories": float(ae),
                        "mean_act_target": float(act[ns & (graph == st)].mean()),
                        "mean_act_other": float(act[ns & (graph != st)].mean()),
                        "firing_rate_target": float((act[ns & (graph == st)] > 0).mean()),
                        "firing_rate_other": float((act[ns & (graph != st)] > 0).mean()),
                        "top_examples": [{"uid": items[j]["uid"], "graph": items[j]["graph_id"], "story": items[j]["group"],
                                          "query_type": items[j]["query_type"], "act": float(act[j]),
                                          "text_tail": (items[j]["context"][-160:] + " | " + items[j]["query"])[-260:]}
                                         for j in order],
                        "top8_graph_purity": float(np.mean([items[j]["graph_id"] == st for j in order])),
                    })
                pres["structures"][st] = {
                    "n_target_discovery": int(yd.sum()), "n_target_eval": int(ye.sum()),
                    "top_features": feats,
                    "mean_heldout_auroc_top10": float(np.mean(a_e)),
                    "n_top10_with_heldout_auroc_above_0.7": int((a_e > 0.7).sum()),
                    "shuffled_label_null_mean_top10_discovery_auroc": float(np.mean(null)),
                    "mean_discovery_auroc_top10": float(a_d[top].mean()),
                }
                if pname == "premise_end":
                    md.append(f"\n## Layer {block} · {pname} · structure = **{st}**\n")
                    for ft in feats[:5]:
                        md.append(f"\n**Feature {ft['feature']}** — AUROC discovery {ft['auroc_discovery']:.2f}, held-out stories "
                                  f"{ft['auroc_heldout_stories']:.2f}; fires on {ft['firing_rate_target']:.0%} of {st} vs "
                                  f"{ft['firing_rate_other']:.0%} of others; top-8 purity {ft['top8_graph_purity']:.0%}\n")
                        for ex in ft["top_examples"][:5]:
                            md.append(f"- ({ex['graph']}, {ex['story']}, {ex['query_type']}, act {ex['act']:.1f}) …{ex['text_tail']}")
            lres["positions"][pname] = pres
            print(f"layer {block} {pname}: FVU {pres['reconstruction']['fraction_variance_unexplained']:.2f} L0 {pres['reconstruction']['mean_l0']:.0f} | "
                  + " ".join(f"{st}:{pres['structures'][st]['mean_heldout_auroc_top10']:.2f}" for st in STRUCTS), flush=True)
        res["layers"][str(block)] = lres
        del sae
    (RESULTS / "sae_features.json").write_text(json.dumps(res, indent=1))
    (RESULTS / "sae_top_examples.md").write_text("\n".join(md))


class FeatureAblator:
    """Subtract selected SAE features' decoder contributions from the residual stream at one
    block's output, at every token position. The SAE error term is left untouched."""

    def __init__(self, model, sae, block: int):
        self.sae, self.feats = sae, None
        self.h = models.get_layers(model)[block].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.feats is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        f = self.sae.encode(h.float())[..., self.feats]                       # [B, T, k]
        h = h - (f @ self.sae.W_dec[self.feats]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        self.h.remove()


def stage_ablate(n_per_struct: int, block: int):
    feats_json = json.loads((RESULTS / "sae_features.json").read_text())
    resid, meta, items = cache.load_cache(MODEL, "cladder")
    ld0 = np.array(meta["logit_diff"]); y = np.array(meta["labels"])
    graph = np.array([i["graph_id"] for i in items])
    ns = np.array([i["group"].startswith("nonsense") for i in items])
    story_no = np.array([int(i["group"][-1]) if ns[k] else -1 for k, i in enumerate(items)])
    evl = ns & (story_no >= 5)
    tok = models.load_tokenizer(MODEL)
    model = models.load_model(MODEL)
    ans = models.answer_token_ids(tok)
    dev = next(model.parameters()).device
    sae = load_sae(block, device="cpu").to(dev)
    # activation statistics for matching random features (premise_end, eval items)
    F = encode(load_sae(block), np.asarray(resid[:, block + 1, POS["premise_end"], :], dtype=np.float32))
    mean_act = F[ns].mean(0)
    rng = np.random.default_rng(SEED + 21)
    abl = FeatureAblator(model, sae, block)
    out = {"model": MODEL, "block": block, "n_per_structure": n_per_struct, "top_k": TOP_K, "structures": {}}
    t0 = time.time()
    try:
        for st in STRUCTS:
            top = [f["feature"] for f in feats_json["layers"][str(block)]["positions"]["premise_end"]["structures"][st]["top_features"]]
            # matched random sets: for each top feature, a random other feature with the closest mean activation
            pool = np.array([f for f in np.argsort(mean_act) if f not in set(top) and mean_act[f] > 0])
            rsets = []
            for _ in range(3):
                rs = []
                for f in top:
                    near = pool[np.argsort(np.abs(mean_act[pool] - mean_act[f]))[:20]]
                    rs.append(int(rng.choice([c for c in near if c not in rs])))
                rsets.append(rs)
            tgt = rng.choice(np.where(evl & (graph == st))[0], min(n_per_struct, int((evl & (graph == st)).sum())), replace=False)
            oth = rng.choice(np.where(evl & (graph != st))[0], n_per_struct, replace=False)
            rows = {}
            for name, idx in (("target_structure", tgt), ("other_structures", oth)):
                d = {"baseline": [], "top": [], "random": [[] for _ in rsets]}
                for i in idx:
                    ids = torch.tensor([prompts.encode_with_positions(tok, items[i])["input_ids"]])
                    def run(feats):
                        abl.feats = None if feats is None else torch.tensor(feats, device=dev)
                        with torch.no_grad():
                            h = models.body_forward(model, ids, torch.ones_like(ids)).last_hidden_state[:, -1]
                        al = models.answer_logits_from_hidden(model, h, ans)
                        ny = len(ans["yes"])
                        return float(torch.logsumexp(al[0, :ny], -1) - torch.logsumexp(al[0, ny:], -1))
                    d["baseline"].append(run(None)); d["top"].append(run(top))
                    for k, rs in enumerate(rsets):
                        d["random"][k].append(run(rs))
                    torch.mps.empty_cache() if dev.type == "mps" else None
                b, t = np.array(d["baseline"]), np.array(d["top"])
                r = np.array(d["random"])
                g = np.array([items[i]["group"] for i in idx])
                yy = y[idx]
                rows[name] = {
                    "n": int(len(idx)),
                    "baseline_matches_cache_max_abs": float(np.abs(b - ld0[idx]).max()),
                    "mean_abs_delta_top": stats.bootstrap_ci(lambda a_, b_: float(np.abs(a_ - b_).mean()), (t, b), g, n_boot=1000),
                    "mean_abs_delta_random": float(np.abs(r - b).mean()),
                    "mean_abs_delta_random_each": np.abs(r - b).mean(1).tolist(),
                    "top_minus_random_mean_abs_delta": stats.bootstrap_ci(
                        lambda a_, b_, c_: float(np.abs(a_ - b_).mean() - np.abs(c_ - b_).mean()), (t, b, r.mean(0)), g, n_boot=1000),
                    "mean_signed_delta_top": float((t - b).mean()),
                    "flip_rate_top": float(((t > 0) != (b > 0)).mean()),
                    "flip_rate_random": float(((r > 0) != (b > 0)).mean()),
                    "accuracy_baseline": stats.accuracy(yy, (b > 0).astype(int)),
                    "accuracy_top_ablated": stats.accuracy(yy, (t > 0).astype(int)),
                }
            out["structures"][st] = {"top_features": top, "random_sets": rsets, **rows}
            print(f"{st}: target |dLD| top {rows['target_structure']['mean_abs_delta_top']['est']:.3f} vs random "
                  f"{rows['target_structure']['mean_abs_delta_random']:.3f}; other-structure items: top "
                  f"{rows['other_structures']['mean_abs_delta_top']['est']:.3f} random {rows['other_structures']['mean_abs_delta_random']:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    finally:
        abl.remove()
    (RESULTS / "sae_ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["features", "ablate"])
    ap.add_argument("--n-per-struct", type=int, default=40)
    ap.add_argument("--block", type=int, default=12)
    a = ap.parse_args()
    stage_features() if a.stage == "features" else stage_ablate(a.n_per_struct, a.block)

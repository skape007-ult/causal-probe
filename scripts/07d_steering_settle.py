"""One steering arm that stood out among 20 controls: a property of the gold label, of adjacency, or of one draw?
Needs the model (GPU). Debiased pool, last token of the hypothesis. Writes results/steering_settle_<model>_c2c_debiased_hyp.json.

Background (eighth audit, notes/audit_round2_runs_raw.md, Q1). In the 20-control runs two gold-label directions at the
end of the hypothesis ranked first among 21 in absolute slope: Qwen 3B, difference of class means (+0.74 per s), and
Qwen 1.5B, probe weights (-0.25 per s). Each is one of 24 such comparisons (12 on this pool), z about 2 to 3 against its
controls, not first in any neighbouring arm.

SECOND VERSION, after an independent review of the first before it ran (ninth audit, notes/audit_settle_code_raw.md):
  - Fitting the direction on the other half of the blocks is NOT an independent draw of it: the gold direction of the
    two halves is nearly one vector (cosine 0.84 for Qwen 3B). A direction whose slope is large for any reproducible
    reason stays large. So "p <= 0.05 in the mirror split" cannot be read as "replicates"; the arm was chosen as the
    most extreme of 24, and the test has to be strict enough for that: p <= 0.05 / 24, which needs 500 controls.
  - The gold label has one valid and one invalid item per premise; a shuffle within hypothesis string does not keep
    that, so those controls carry premise-level variance the gold direction cannot. The primary controls are now
    labellings with BOTH of the gold label's balances (within hypothesis string, within premise): `premise_balanced_*`.
  - "Controls that keep adjacency" built as class-mean differences share a common component (the difference under the
    cell rates) that the real direction has less of; with no effect at all the real direction fell outside them in a
    third to a half of runs. They are replaced by the part the shuffle actually randomises: the label minus its cell
    mean, projected on the centred activations (`gold_beyond_adjacency` against `within_cell_residual`). The same for
    adjacency (`adjacency_within_string`). For probe directions the within-cell family was calibrated and is kept.

Splits of the 25 blocks into a fitting half and a steered half: 0 = the project's split (reproduces the 20-control run:
same 150 items, same first 20 within-string labellings); 1 = its mirror image; 2.. = seeded random halvings.

PRIMARY TEST (notes/log.md, Sat 14 Nov, amended before any run): split 1, the site's layer, the direction named by
--arm, against --n-primary premise-balanced controls of the same kind, one-sided in the sign that direction had in the
20-control run, per unit of s. p = (1 + controls at least as extreme) / (1 + n). Everything else is descriptive.

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
N_COMPARISONS = 24            # gold-label directions compared with matched controls so far (results/steering_controls_multiplicity.json)
KIND = {"gold_diffmeans": "diffmeans", "gold_probe": "probe"}


def probe_dir(X, y, C, max_iter=2000):
    """Unit direction of a standardised logistic probe in raw activation space, and whether the fit converged."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X), y)
    return steer.unit(clf.coef_[0] / sc.scale_), bool(clf.n_iter_[0] < max_iter)


def probe_dirs(X, labellings, C):
    """Probe directions for many labellings, in parallel on the CPU cores (CAUSAL_PROBE_JOBS)."""
    if not len(labellings):
        return [], 1.0
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=int(os.environ.get("CAUSAL_PROBE_JOBS", "2")))(delayed(probe_dir)(X, lab, C) for lab in labellings)
    return [d for d, _ in out], float(np.mean([c for _, c in out]))


def dom(X, lab):
    return steer.unit(X[lab == 1].mean(0) - X[lab == 0].mean(0))


def cell_mean(v, cells):
    out = np.zeros(len(v))
    for c in np.unique(cells):
        out[cells == c] = v[cells == c].mean()
    return out


def shuffle_within(lab, cells, rng):
    """Permute `lab` inside every cell (cells: array of hashable cell ids, same length)."""
    out = lab.copy()
    for c in np.unique(cells):
        m = np.where(cells == c)[0]
        out[m] = lab[rng.permutation(m)]
    return out


class PremiseBalanced:
    """Random labellings with both balances of the gold label: every premise has one valid and one invalid item, and
    every hypothesis string is half valid. A Markov chain started at the gold labelling: each premise is an edge from
    the string of its valid item to the string of its invalid item; reversing a directed cycle of that graph swaps the
    labels of the premises on it and keeps both balances. `thin` cycle reversals separate two labellings."""

    def __init__(self, y, strs, groups, rng, burn=8000, thin=2500):
        prem = {}
        for i, g in enumerate(groups):
            prem.setdefault(g, []).append(i)
        assert all(len(v) == 2 for v in prem.values()), "every premise must have exactly two items on the fitting half"
        self.pairs = np.array(list(prem.values()))
        assert (y[self.pairs].sum(1) == 1).all(), "every premise must have one valid and one invalid item"
        ids = {s: k for k, s in enumerate(np.unique(strs))}
        self.node = np.array([ids[s] for s in strs])
        self.y, self.rng, self.thin = y.copy(), rng, thin
        self.first = (self.y[self.pairs[:, 0]] == 1)                       # is the pair's first item the valid one?
        self.out = {}
        for p in range(len(self.pairs)):
            self.out.setdefault(self._src(p), []).append(p)
        for _ in range(burn):
            self._reverse_cycle()

    def _src(self, p):
        return int(self.node[self.pairs[p, 0 if self.first[p] else 1]])

    def _dst(self, p):
        return int(self.node[self.pairs[p, 1 if self.first[p] else 0]])

    def _reverse_cycle(self):
        cur = self._src(int(self.rng.integers(len(self.pairs))))
        seen, edges = {cur: 0}, []
        while True:
            lst = self.out[cur]
            p = lst[int(self.rng.integers(len(lst)))]
            edges.append(p); cur = self._dst(p)
            if cur in seen:
                cyc = edges[seen[cur]:]
                break
            seen[cur] = len(edges)
        for p in cyc:
            self.out[self._src(p)].remove(p)
            self.first[p] = not self.first[p]
            self.out.setdefault(self._src(p), []).append(p)
            a, b = self.pairs[p]
            self.y[a], self.y[b] = self.y[b], self.y[a]

    def sample(self):
        for _ in range(self.thin):
            self._reverse_cycle()
        return self.y.copy()


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


def build(X, F, y, t, strs, grp, tm, mcal, C, k, arm, role, n_primary, n_shuf, n_other):
    """Unit directions for one (split, layer), fitted on the fitting half F.
    role: "primary" (split 1 at the site), "original" (split 0 at the site), "other" (random splits, other layers)."""
    Xf, yf, tf, sf, gf, tmf = X[F], y[F], t[F], strs[F], grp[F], tm[F]
    kind = KIND[arm]
    fam, D, conv = [], [], {}

    def add(f, v):
        fam.append(f); D.append(np.asarray(v, np.float64))

    def add_family(labellings, n_dm, n_pr, name):
        if n_pr:
            dirs, share = probe_dirs(Xf, labellings[:n_pr], C)
            conv[f"{name}_probe_share_converged"] = share
            for d in dirs:
                add(f"{name}_probe", d)
        for lab in labellings[:n_dm]:
            add(f"{name}_diffmeans", dom(Xf, lab))

    def counts(n_arm, n_rest):          # (difference-of-means, probe) counts: n_arm for the arm's kind, n_rest for the other kind
        return (n_arm, n_rest) if kind == "diffmeans" else (n_rest, n_arm)

    # same generator and order of draws as 07c_steering_controls.py for split 0, so that its first 20 within-string
    # labellings are those of the 20-control run: 100 isotropic draws first, then the shuffles
    rng = np.random.default_rng(SEED + 101 + 1000 * k)
    iso = [rng.standard_normal(X.shape[1]) for _ in range(100)]
    add("gold_diffmeans", dom(Xf, yf))
    g, c_ = probe_dir(Xf, yf, C); add("gold_probe", g); conv["gold_probe"] = c_
    m, c_ = probe_dir(Xf, mcal[F], C); add("model_out", m); conv["model_out"] = c_
    if role == "original":
        for v in iso[:N_ISO]:
            add("isotropic", steer.unit(v))
    # (1) labels shuffled within hypothesis string (the 20-control run's family)
    n_ws = n_shuf if role in ("primary", "original") else n_other
    ws = [shuffle_within(yf, sf, rng) for _ in range(n_ws)]
    add_family(ws, *counts(n_ws, n_other if role != "other" else 0), "within_string")
    # (2) labels with both of the gold label's balances (within string and within premise): the primary controls
    n_pb = {"primary": n_primary, "original": n_shuf, "other": n_other}[role]
    chain = PremiseBalanced(yf, sf, gf, np.random.default_rng(SEED + 404 + 1000 * k))
    pb = [chain.sample() for _ in range(n_pb)]
    for lab in pb[:3]:
        assert all(lab[sf == s_].mean() == 0.5 for s_ in np.unique(sf)[:8]) and (lab[chain.pairs].sum(1) == 1).all()
    conv["premise_balanced_mean_agreement_with_gold"] = float(np.mean([(lab == yf).mean() for lab in pb]))
    conv["premise_balanced_mean_agreement_between_successive"] = float(np.mean([(a_ == b_).mean() for a_, b_ in zip(pb[:-1], pb[1:])])) if len(pb) > 1 else None
    add_family(pb, *counts(n_pb, n_other if role != "other" else 0), "premise_balanced")
    if role != "other":
        # (3) beyond adjacency. Class-mean differences of labels shuffled within (string x adjacency) cells share the
        # difference under the cell rates and are not a calibrated family. What the shuffle randomises is the label
        # minus its cell mean; real and control are built from that part alone.
        Xc = Xf - Xf.mean(0)
        cell = np.array([f"{a}|{b}" for a, b in zip(sf, tf)])
        rng_c = np.random.default_rng(SEED + 202 + 1000 * k)
        ybar = cell_mean(yf.astype(float), cell)
        add("gold_beyond_adjacency", steer.unit((yf - ybar) @ Xc))
        wc = [shuffle_within(yf, cell, rng_c) for _ in range(n_shuf)]
        for lab in (wc if kind == "diffmeans" else wc[:n_other]):
            add("within_cell_residual", steer.unit((lab - ybar) @ Xc))
        if kind == "probe":             # for probe directions the within-cell family is calibrated as it is
            dirs, share = probe_dirs(Xf, wc, C)
            conv["within_cell_probe_share_converged"] = share
            for d in dirs:
                add("within_cell_probe", d)
        # (4) the adjacency direction itself
        par = tmf == "parent"
        tbar = cell_mean(tf.astype(float), sf)
        add("adjacency_parent_diffmeans", dom(Xf[par], tf[par]))
        add("adjacency_within_string", steer.unit((tf - tbar) @ Xc))
        rng_a = np.random.default_rng(SEED + 303 + 1000 * k)
        for _ in range(n_other):
            ts = shuffle_within(tf, sf, rng_a)
            add("adjacency_parent_shuffled", dom(Xf[par], ts[par]))
            add("adjacency_within_string_shuffled", steer.unit((ts - tbar) @ Xc))
    fam = np.array(fam); D = np.stack(D)
    assert np.isfinite(D).all(), "a direction is not finite (an empty class?)"
    alphas = [(j, 1.0) for j in range(len(D))]
    if role == "original":                 # smaller steps for the two gold directions and the first within-string controls
        small = [j for j, f in enumerate(fam) if f in ("gold_diffmeans", "gold_probe")] + \
                [j for j, f in enumerate(fam) if f == "within_string_diffmeans"][:N_SMALL] + \
                [j for j, f in enumerate(fam) if f == "within_string_probe"][:N_SMALL]
        alphas += [(j, a) for a in SMALL_ALPHAS for j in small]
    return fam, D.astype(np.float32), alphas, conv


def rank_stats(real, ctrl, real_sd=None, ctrl_sd=None):
    """Where a real slope stands among its controls. signed_rank: 1 = most positive among 1 + n."""
    ctrl = np.asarray(ctrl, float)
    r = {"slope": float(real), "n_controls": int(len(ctrl)), "control_mean": float(ctrl.mean()), "control_sd": float(ctrl.std(ddof=1)),
         "control_min": float(ctrl.min()), "control_max": float(ctrl.max()), "control_max_abs": float(np.abs(ctrl).max()),
         "z_vs_controls": float((real - ctrl.mean()) / ctrl.std(ddof=1)),
         "signed_rank_1_is_most_positive": int(1 + (ctrl > real).sum()),
         "abs_rank_1_is_largest": int(1 + (np.abs(ctrl) > abs(real)).sum()),
         "share_of_controls_smaller_in_absolute_value": float((np.abs(ctrl) < abs(real)).mean())}
    if real_sd is not None:
        r["abs_rank_per_natural_sd"] = int(1 + (np.abs(ctrl * ctrl_sd) > abs(real * real_sd)).sum())
        r["natural_sd_over_s"] = float(real_sd); r["natural_sd_controls_median"] = float(np.median(ctrl_sd))
    return r


def one_sided_p(real, ctrl, sign):
    ctrl = np.asarray(ctrl, float)
    return float((1 + (sign * ctrl >= sign * real).sum()) / (1 + len(ctrl)))


MATCH = {"gold_diffmeans": ("premise_balanced_diffmeans", "within_string_diffmeans"),
         "gold_probe": ("premise_balanced_probe", "within_string_probe", "within_cell_probe"),
         "gold_beyond_adjacency": ("within_cell_residual",),
         "model_out": ("within_string_probe", "premise_balanced_probe"),
         "adjacency_parent_diffmeans": ("adjacency_parent_shuffled",),
         "adjacency_within_string": ("adjacency_within_string_shuffled",)}


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
        r = {"slope_per_s": float(slopes[j]), "own_sd_over_s": float(own_sd[j]), "gold_class_gap_along_direction_over_s": float(gap_s[j]),
             "effect_of_moving_across_the_gold_class_gap_logits": float(slopes[j] * gap_s[j]),
             "share_of_items_with_positive_slope": float((slope_items[:, j] > 0).mean()), "versus": {}}
        for f in fams:
            if (fam == f).sum() > 1:
                r["versus"][f] = rank_stats(slopes[j], slopes[fam == f], own_sd[j], own_sd[fam == f])
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


def config_slopes(p):
    out, n1 = p["out"].astype(np.float64), len(p["fam"])
    return (out[:, 1:1 + 2 * n1:2] - out[:, 2:2 + 2 * n1:2]) / 2.0          # [items, directions], per s at a = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--arm", default="gold_diffmeans", choices=list(KIND), help="the direction of the primary test")
    ap.add_argument("--n", type=int, default=150, help="steered items per split")
    ap.add_argument("--n-primary", type=int, default=500, help="premise-balanced controls of the arm's kind in the primary configuration (split 1)")
    ap.add_argument("--n-shuf", type=int, default=100, help="controls per family of the arm's kind in splits 0 and 1 otherwise")
    ap.add_argument("--n-other", type=int, default=20, help="controls per family elsewhere")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--layer-offsets", default="-2,2", help="extra layers for split 0, relative to the site's layer (on a command line write --layer-offsets=-2,2)")
    ap.add_argument("--sub-batch", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=100.0)
    ap.add_argument("--allow-new-site", action="store_true", help="do not require the 20-control result (smoke tests)")
    a = ap.parse_args()
    assert a.n_other >= 2 and a.n_shuf >= a.n_other and a.n_primary >= a.n_other, "families need at least two members"
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
        assert sign != 0
    elif a.allow_new_site:
        layer, C, _ = steer.choose_site(a.model, TAG, resid, meta, items, POSITION, 6)
        base, sign = None, 1.0
    else:
        raise SystemExit(f"{base_json.name} is not in this checkout: this run must sit at the site of the 20-control run")
    L = resid.shape[1] - 1
    y = np.array(meta["labels"]); ld0 = np.array(meta["logit_diff"]); ntok = np.array(meta["n_tokens"], float)
    t = probes.structural_features(items)[:, 2].astype(int)
    strs = np.array([i["query"] for i in items]); tm = np.array([i["template"] for i in items]); grp = np.array([i["group"] for i in items])
    mcal = (ld0 > np.median(ld0)).astype(int)
    splits = block_splits(items, max(a.n_splits, 2))
    offs = [int(o) for o in a.layer_offsets.split(",") if o.strip() and int(o) != 0]
    # the primary configuration first: if the session ends early, it is the one that must exist
    configs = [(1, layer, "primary"), (0, layer, "original")] + [(0, layer + o, "other") for o in offs if 1 <= layer + o <= L] + \
              [(k, layer, "other") for k in range(2, len(splits))]
    log(f"site: last token of the hypothesis, layer {layer}/{L}, C {C:g}; arm {a.arm}, expected sign {sign:+.0f}; configurations {[(k, l) for k, l, _ in configs]}")

    plan, saved = [], {}
    for k, l, role in configs:
        X = probes.layer_features(resid, l, pi)
        F = splits[k]; E = ~F
        fam, D, alphas, conv = build(X, F, y, t, strs, grp, tm, mcal, C, k, a.arm, role, a.n_primary, a.n_shuf, a.n_other)
        mu = X[F].mean(0)
        s_unit = float(np.sqrt(((X[F] - mu) ** 2).sum(1).mean()))
        own_sd = ((X[F] - mu) @ D.T).std(0) / s_unit
        gap_s = ((X[E & (y == 1)].mean(0) - X[E & (y == 0)].mean(0)) @ D.T) / s_unit      # gold class gap on the steered half, along each direction
        r3 = np.random.default_rng(SEED + 13 + 1000 * k)                                   # the 20-control run's item sample for split 0
        per = min(a.n // 2, int((E & (y == 1)).sum()), int((E & (y == 0)).sum()))
        idx = np.sort(np.concatenate([r3.choice(np.where(E & (y == c))[0], per, replace=False) for c in (1, 0)]))
        cols = np.concatenate([np.zeros((1, D.shape[1]), np.float32)] + [s * al * s_unit * D[j][None] for j, al in alphas for s in (1.0, -1.0)])
        key = f"{k}_{l}"
        plan.append(dict(split=k, layer=l, role=role, key=key, fam=fam, D=D, alphas=alphas, conv=conv, s_unit=s_unit, own_sd=own_sd, gap_s=gap_s,
                         idx=idx, vecs=cols, check=float(np.abs(cols).sum())))
        saved.update({f"D_{key}": D.astype(np.float16), f"fam_{key}": fam, f"alphas_{key}": np.array(alphas, np.float64), f"own_sd_{key}": own_sd,
                      f"gap_s_{key}": gap_s, f"s_unit_{key}": s_unit, f"idx_{key}": idx})
        log(f"split {k} layer {l} ({role}): {len(D)} directions, {len(cols)} vectors per item, {len(idx)} items; converged: "
            + ", ".join(f"{n_} {v_ if isinstance(v_, bool) else round(v_, 2)}" for n_, v_ in conv.items() if "converged" in n_ or n_ == "gold_probe")
            + f"; premise-balanced labellings agree with gold on {100 * conv['premise_balanced_mean_agreement_with_gold']:.0f}% of items")
    # the directions and everything needed to read the outputs, before any GPU time: they cannot be rebuilt on the laptop
    # (the GPU run's activations are not exported), and a run that is cut short would otherwise leave outputs without labels
    np.savez(cache.cache_dir(a.model, TAG) / "steering_settle_hyp_plan.npz", **saved)

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
    work = lambda p_, items_=None: float(ntok[p_["idx"] if items_ is None else items_].sum() * len(p_["vecs"]))    # cost grows with prompt length x vectors
    t_loop, w_done, dropped = time.time(), 0.0, []
    ci = 0
    while ci < len(plan):
        p = plan[ci]; key = p["key"]
        if f"out_{key}" in done and done[f"out_{key}"].shape == (len(p["idx"]), len(p["vecs"])) and np.array_equal(done.get(f"idx_{key}"), p["idx"]) \
                and abs(float(done.get(f"check_{key}", -1)) - p["check"]) < 1e-4 * p["check"]:
            p["out"] = done[f"out_{key}"]
            log(f"split {p['split']} layer {p['layer']}: taken from the partial file")
            ci += 1
            continue
        out = np.zeros((len(p["idx"]), len(p["vecs"])), np.float32)
        vecs = torch.tensor(p["vecs"])
        for n_, i in enumerate(p["idx"]):
            enc = prompts.encode_with_positions(tok, items[i])
            out[n_] = steer.steered_logit_diffs(model, enc["input_ids"], p["layer"], vecs, ans, sub_batch=a.sub_batch, pos=enc["pos"][POSITION])
            w_done += ntok[i] * len(p["vecs"])
            if (n_ == 5 or n_ == len(p["idx"]) // 2) and w_done > 0:
                per_w = (time.time() - t_loop) / w_done
                left = a.max_minutes - (time.time() - t0) / 60
                rest = work(p, p["idx"][n_ + 1:]) + sum(work(q) for q in plan[ci + 1:])
                log(f"[timing] {per_w * 1e6:.2f} µs per token and vector -> {per_w * rest / 60:.1f} min for the rest; {left:.1f} min left of {a.max_minutes:.0f}")
                while len(plan) > ci + 1 and per_w * rest / 60 > 0.9 * left:      # drop configurations from the end, never the one that is running
                    q = plan.pop()
                    dropped.append([q["split"], q["layer"], q["role"]])
                    rest -= work(q)
                    log(f"[timing] dropping split {q['split']} layer {q['layer']} ({q['role']}) to fit the time limit")
                if ci == 0 and n_ == 5 and per_w * work(p, p["idx"][n_ + 1:]) / 60 > 0.95 * left:
                    raise SystemExit("the primary configuration alone does not fit the time limit: lower --n-primary")
        p["out"] = out
        done[f"out_{key}"] = out; done[f"idx_{key}"] = p["idx"]; done[f"check_{key}"] = p["check"]
        tmp = part.with_name(part.name + ".tmp.npz")
        np.savez(tmp, **done); os.replace(tmp, part)
        log(f"split {p['split']} layer {p['layer']} done; baseline max |difference| vs cache {np.abs(out[:, 0] - ld0[p['idx']]).max():.3f}")
        ci += 1
    del model

    kind = KIND[a.arm]
    res = {"model": a.model, "tag": TAG, "position": POSITION, "layer": layer, "n_layers": int(L), "probe_C": C, "arm": a.arm,
           "expected_sign_from_the_20_control_run": sign, "slopes_in_the_20_control_run": base,
           "n_primary": a.n_primary, "n_shuf": a.n_shuf, "n_other": a.n_other, "step_in_s": 1.0, "smaller_steps_in_s": list(SMALL_ALPHAS),
           "configurations_dropped_for_time": dropped, "configs": []}
    raw = {}
    for p in plan:
        out, fam = p["out"].astype(np.float64), p["fam"]
        n1 = len(fam)
        slope_items = config_slopes(p)
        slopes = slope_items.mean(0)
        small, ctrl_small = {}, {}
        for e_i, (j, al) in enumerate(p["alphas"][n1:]):
            c0 = 1 + 2 * n1 + 2 * e_i
            v_ = float(((out[:, c0] - out[:, c0 + 1]) / (2 * al)).mean())
            if fam[j] in ("gold_diffmeans", "gold_probe"):
                small.setdefault(str(fam[j]), {})[f"{al:g}"] = v_
            else:
                ctrl_small.setdefault(str(fam[j]), {}).setdefault(f"{al:g}", []).append(v_)
        for k_, fam_c in (("gold_diffmeans", "within_string_diffmeans"), ("gold_probe", "within_string_probe")):
            if k_ in small and fam_c in ctrl_small:
                small[k_] = {al_: {"slope_per_s": v_, "signed_rank_among_first_controls": int(1 + (np.array(ctrl_small[fam_c][al_]) > v_).sum()),
                                   "n_controls": len(ctrl_small[fam_c][al_])} for al_, v_ in small[k_].items()}
        tm_items = tm[p["idx"]] if p["role"] != "other" else None
        s = summarise(fam, slopes, p["own_sd"], p["gap_s"], small, tm_items, slope_items)
        s.update(split=p["split"], layer=p["layer"], role=p["role"], n_items=int(len(p["idx"])), s_unit=p["s_unit"], fitting_notes=p["conv"],
                 fitting_half={0: "selection half (the original split)", 1: "report half (mirror image of the original split)"}.get(p["split"], "random half of the blocks"),
                 baseline_max_abs_diff_vs_cache=float(np.abs(out[:, 0] - ld0[p["idx"]]).max()))
        res["configs"].append(s)
        raw.update({f"out_{p['key']}": p["out"], f"idx_{p['key']}": p["idx"], f"fam_{p['key']}": fam,
                    f"own_sd_{p['key']}": p["own_sd"], f"alphas_{p['key']}": np.array(p["alphas"], np.float64)})
        if p["role"] == "primary":
            cf = f"premise_balanced_{kind}"
            j = int(np.where(fam == a.arm)[0][0])
            c = slopes[fam == cf]
            pv = one_sided_p(slopes[j], c, sign)
            res["primary_test"] = {
                "what": f"split 1 (fitted on the report half, steered on the selection half), layer {layer}, {a.arm} against {len(c)} {cf} controls "
                        f"(labellings with both balances of the gold label), one-sided in the sign of the 20-control run ({sign:+.0f}), per unit of s",
                "slope": float(slopes[j]), "p_one_sided": pv, "threshold_for_the_most_extreme_of_24_comparisons": 0.05 / N_COMPARISONS,
                "survives_the_correction_for_24_comparisons": bool(pv <= 0.05 / N_COMPARISONS),
                "smallest_p_this_many_controls_can_give": 1 / (1 + len(c)),
                **{k_: v_ for k_, v_ in rank_stats(slopes[j], c, p["own_sd"][j], p["own_sd"][fam == cf]).items() if k_ != "slope"},
                "p_one_sided_vs_within_string_controls": one_sided_p(slopes[j], slopes[fam == f"within_string_{kind}"], sign),
                "note": "The direction of this split is not an independent draw: the gold direction fitted on the two halves is nearly one vector. "
                        "p <= 0.05 here means only that this direction's slope is reproducibly in the tail of its controls, which choosing the most "
                        "extreme of 24 comparisons would produce by itself; the threshold for that choice is 0.05/24.",
                "fixed_before_the_run": True}
    cf = f"premise_balanced_{kind}"
    res["arm_across_splits_and_layers_descriptive"] = []
    for p, c in zip(plan, res["configs"]):
        fam, slopes = p["fam"], config_slopes(p).mean(0)
        e = {"split": c["split"], "layer": c["layer"], "role": c["role"], "slope": c["real"][a.arm]["slope_per_s"]}
        for f in (cf, f"within_string_{kind}"):
            if (fam == f).sum() > 1:
                v = c["real"][a.arm]["versus"][f]
                e[f] = {"p_one_sided": one_sided_p(e["slope"], slopes[fam == f], sign),
                        **{k_: v[k_] for k_ in ("n_controls", "z_vs_controls", "signed_rank_1_is_most_positive", "abs_rank_1_is_largest", "abs_rank_per_natural_sd")}}
        res["arm_across_splits_and_layers_descriptive"].append(e)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    np.savez(raw_path, **raw)
    (RESULTS / f"steering_settle_{a.model}_{TAG}_hyp.json").write_text(json.dumps(res, indent=1))
    part.unlink(missing_ok=True)
    if "primary_test" in res:
        pt = res["primary_test"]
        print(f"PRIMARY TEST: {pt['what']}\n  slope {pt['slope']:+.3f}; controls mean {pt['control_mean']:+.3f}, SD {pt['control_sd']:.3f}, "
              f"range {pt['control_min']:+.3f} to {pt['control_max']:+.3f}; one-sided p = {pt['p_one_sided']:.4f} "
              f"(threshold {pt['threshold_for_the_most_extreme_of_24_comparisons']:.4f}: {'SURVIVES' if pt['survives_the_correction_for_24_comparisons'] else 'does not survive'}); "
              f"against within-string controls p = {pt['p_one_sided_vs_within_string_controls']:.3f}")
    for e in res["arm_across_splits_and_layers_descriptive"]:
        v = e.get(cf)
        if v:
            print(f"  split {e['split']} layer {e['layer']} ({e['role']}): {a.arm} slope {e['slope']:+.3f}, vs {v['n_controls']} premise-balanced controls z {v['z_vs_controls']:+.2f}, "
                  f"signed rank {v['signed_rank_1_is_most_positive']}/{v['n_controls'] + 1}, one-sided p {v['p_one_sided']:.3f}")
    for c in res["configs"][:2]:
        b_ = c["real"].get("gold_beyond_adjacency", {}).get("versus", {}).get("within_cell_residual")
        w_ = c["real"]["gold_probe"]["versus"].get("within_cell_probe")
        if b_:
            print(f"  split {c['split']}: the label beyond adjacency (label minus its string-and-adjacency cell mean): slope {b_['slope']:+.3f}, z {b_['z_vs_controls']:+.2f} "
                  f"against {b_['n_controls']} controls shuffled within cells" + (f"; gold probe against {w_['n_controls']} within-cell probes z {w_['z_vs_controls']:+.2f}" if w_ else ""))


if __name__ == "__main__":
    main()

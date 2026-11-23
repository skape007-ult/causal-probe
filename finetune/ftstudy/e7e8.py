"""E7/E8 (PREREG Amendment 3): the pieces that need no model. Site rule, directions and controls, item and donor
rules, statistics, and the power rule. GPU code is in finetune/scripts/e7e8_gpu.py. Written before the pilot.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from . import seedtest

ALPHAS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
N_CONTROLS = 20
CHI2_LOW20_DF2 = 0.446                  # lower 20% point of chi-square with 2 df (Amendment 3, Section 4 [C])
SEED_SD_FLOOR = 0.03
EFFECT = 0.10
ALPHA_FAMILY = 0.05 / 4
N_GRID = (300, 600, 1000)


# ------------------------------------------------------------------ site
def site_layer(S_sel_by_seed: np.ndarray, layers: list[int]) -> int | None:
    """Amendment 3, Section 1 [c2, F]. S_sel_by_seed: seeds x layers (selection-half out-of-fold S at content_end).
    Seed mean m(l), NaN layers ignored; excess e = m - 0.5; first layer with e >= 0.9 max e. None if max e <= 0.02."""
    m = np.nanmean(np.asarray(S_sel_by_seed, float), 0)
    ok = ~np.isnan(m)
    if not ok.any():
        return None
    e = m - 0.5
    mx = np.max(e[ok])
    if mx <= 0.02:
        return None
    return int(layers[int(np.where(ok & (e >= 0.9 * mx))[0][0])])


# ------------------------------------------------------------------ directions
def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def diff_of_means(X, y):
    y = np.asarray(y).astype(bool)
    return unit(X[y].mean(0) - X[~y].mean(0))


def projector(d_ans, d_text):
    """Gram-Schmidt in the order d_ans, d_text; returns a function d -> d minus its part in span{d_ans, d_text}."""
    u1 = unit(d_ans)
    u2 = unit(d_text - (d_text @ u1) * u1)

    def P(d):
        d = np.asarray(d, float)
        return d - (d @ u1) * u1 - (d @ u2) * u2
    return P


def shuffle_within_cells(y, cells, seed):
    """Labels permuted inside each cell; single-class cells are unchanged by construction."""
    rng = np.random.default_rng(seed)
    out = np.asarray(y).copy()
    for c in np.unique(cells):
        i = np.where(cells == c)[0]
        out[i] = out[rng.permutation(i)]
    return out


def directions(X, y, own, text_sign, strings, n_controls=N_CONTROLS):
    """X: selection-half activations at the site (n x d); y gold; own = model's answer sign; text_sign = sign of the
    out-of-fold tuned stack. Returns dict of unit directions and diagnostics (Amendment 3, Section 2)."""
    y, own, text_sign = (np.asarray(v).astype(int) for v in (y, own, text_sign))
    d_gold, d_ans, d_text = diff_of_means(X, y), diff_of_means(X, own), diff_of_means(X, text_sign)
    P = projector(d_ans, d_text)
    Pg = P(d_gold)
    g = unit(Pg)
    cells = np.array([f"{s}|{a}|{t}" for s, a, t in zip(strings, own, text_sign)])
    ctrl = []
    for k in range(1, n_controls + 1):
        ys = shuffle_within_cells(y, cells, k)
        ctrl.append(unit(P(diff_of_means(X, ys))))
    ctrl = np.array(ctrl)
    mixed = np.zeros(len(y), bool)
    within = 0.0
    for c in np.unique(cells):
        i = cells == c
        p = y[i].mean()
        mixed[i] = 0 < p < 1
        within += i.sum() * p * (1 - p)
    p_all = y.mean()
    V = {"g": g, "d_gold": d_gold, "d_ans": d_ans, "d_text": d_text, "controls": ctrl}
    sig = {k: float(np.std(X @ v)) for k, v in V.items() if k != "controls"}
    sig["controls"] = np.array([float(np.std(X @ v)) for v in ctrl])
    diag = {"r_projected_norm": float(np.linalg.norm(Pg) / np.linalg.norm(d_gold)),
            "cos_gold_ans": float(d_gold @ d_ans), "cos_gold_text": float(d_gold @ d_text),
            "share_items_in_mixed_cells": float(mixed.mean()),
            "share_gold_variance_within_cells": float(within / (len(y) * p_all * (1 - p_all))),
            "median_abs_cos_g_control": float(np.median(np.abs(ctrl @ g))),
            "n_cells": int(len(np.unique(cells)))}
    return V, sig, diag


def separation(h, y, v, sigma):
    """Report-half difference of class means of h.v, in units of sigma."""
    s = h @ v
    y = np.asarray(y).astype(bool)
    return float((s[y].mean() - s[~y].mean()) / sigma)


# ------------------------------------------------------------------ items and donors
def premise_order(groups, seed=0):
    ug = np.unique(groups)
    return list(np.random.default_rng(seed).permutation(ug))


def take_premises(groups, idx_pool, n_max, seed=0, exclude=()):
    """Whole premises from idx_pool in a seeded order, truncated at the premise boundary at or below n_max [F]."""
    g = np.asarray(groups)
    pool = np.asarray(idx_pool)
    out, ex = [], set(exclude)
    for p in premise_order(g[pool], seed):
        if p in ex:
            continue
        members = pool[g[pool] == p]
        if len(out) + len(members) > n_max:
            break
        out.extend(members.tolist())
    return np.array(sorted(out), int)


def donors(i_items, pool, y, strings, adj, groups, n_tokens, seed=0):
    """For each item: (opposite-label donor, same-label donor) indices or -1. First match in a seeded order of the
    pool: another premise, same string, same adjacency, token count within +-10%, opposite / same gold label."""
    order = np.asarray(pool)[np.random.default_rng(seed).permutation(len(pool))]
    out = []
    for i in i_items:
        ok = ((strings[order] == strings[i]) & (adj[order] == adj[i]) & (groups[order] != groups[i])
              & (np.abs(n_tokens[order] - n_tokens[i]) <= 0.10 * n_tokens[i]))
        opp = order[ok & (y[order] != y[i])]
        same = order[ok & (y[order] == y[i]) & (order != i)]
        out.append((int(opp[0]) if len(opp) else -1, int(same[0]) if len(same) else -1))
    return np.array(out, int)


# ------------------------------------------------------------------ statistics
def slopes(LD_steps):
    """LD_steps: items x 5 (alphas -2..2) -> least-squares slope per item."""
    a = ALPHAS - ALPHAS.mean()
    return (np.asarray(LD_steps, float) - np.asarray(LD_steps, float).mean(1, keepdims=True)) @ a / (a @ a)


def e7_items(b_g, b_ctrl, s_ld):
    """Per-item contribution to X7: b_i(g)/s_LD - mean_k b_i(control_k)/s_LD (b_ctrl: items x K)."""
    return (np.asarray(b_g) - np.asarray(b_ctrl).mean(1)) / s_ld


def e8_items(ld0, ld_opp, ld_same, y_opp, s_ld):
    """e_i = {[LD(patch j) - LD] - [LD(patch j') - LD]} x (2 y_j - 1) / s_LD."""
    return ((ld_opp - ld0) - (ld_same - ld0)) * (2 * np.asarray(y_opp) - 1) / s_ld


def premise_se(x, groups, n_boot=2000, seed=0):
    """SE of the item mean of x over premise-cluster resamples."""
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x); cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(ug), (n_boot, len(ug)))
    means = sums[k].sum(1) / cnt[k].sum(1)
    return float(np.std(means, ddof=1))


def seed_sd_bound(per_cond_values: dict) -> dict:
    """Pooled seed SD over the conditions (two seeds each -> 1 df each), its upper 80% bound with the floor [C]."""
    num, df = 0.0, 0
    for v in per_cond_values.values():
        v = np.asarray(v, float)
        num += ((v - v.mean()) ** 2).sum(); df += len(v) - 1
    s = float(np.sqrt(num / df)) if df else float("nan")
    b = s * np.sqrt(df / CHI2_LOW20_DF2) if df == 2 else s * np.sqrt(df / sps.chi2.ppf(0.20, df))
    return {"pooled_sd": s, "df": df, "upper80": float(b), "used": float(max(b, SEED_SD_FLOOR))}


def power(effect, seed_sd, prem_se, n_seeds=5, alpha=ALPHA_FAMILY, reps=20000, seed=0):
    """Registered test simulated: seed values = effect + common premise error (fully correlated across seeds) + seed
    noise; v_premise = prem_se^2 (known)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, prem_se, reps)[:, None]
    x = effect + common + rng.normal(0, seed_sd, (reps, n_seeds))
    D = x.mean(1)
    t, df = seedtest.t_and_df(D, x.var(1, ddof=1), n_seeds, None, 0, np.full(reps, prem_se ** 2))
    return float(np.mean(seedtest.p_two_sided(t, df) < alpha))


def choose_n_items(tests: dict, n_all: int, items_per_premise: float, pilot_premises: int, seed_sd: float):
    """tests: name -> premise SE of the seed mean measured on the pilot items (mean over the pilot seeds).
    Per test, the smallest N in the grid (plus all remaining) with power >= 0.6 at EFFECT; None = dropped.
    N_items = the largest N needed by a test not dropped [C]."""
    grid = [n for n in N_GRID if n < n_all] + [n_all]
    need, table = {}, {}
    for name, se_pilot in tests.items():
        table[name] = {}
        need[name] = None
        for N in grid:
            se = se_pilot * np.sqrt(pilot_premises / max(1.0, N / items_per_premise))
            pw = power(EFFECT, seed_sd, se)
            table[name][N] = pw
            if pw >= 0.6 and need[name] is None:
                need[name] = N
    kept = [v for v in need.values() if v is not None]
    return {"needed": need, "power_table": table, "N_items": max(kept) if kept else None,
            "dropped": [k for k, v in need.items() if v is None]}

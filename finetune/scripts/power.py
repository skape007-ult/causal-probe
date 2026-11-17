"""Size and power of the ONE test PREREG.md (version 2) registers, simulated for that test. CPU, a few minutes.

  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/power.py        -> finetune/power/power.json

The registered test (ftstudy/seedtest.py, the same code the analysis will call):
    D   = mean over seeds of the statistic in condition X  -  mean over seeds in condition Y   (or one condition
          minus a constant; or the mean over seeds of a within-seed difference)
    SE² = s_X²/n_X + s_Y²/n_Y  +  v_premise
          s² = variance of the per-seed statistics (seed-to-seed variation of trained models);
          v_premise = variance of D over 2,000 premise resamples with the trained models held fixed.
    t   = D / SE, referred to Student's t with the Welch-Satterthwaite degrees of freedom of the seed terms.
          Two-sided, alpha = 0.05.

Part 1 takes the premise standard error from the BUILT evaluation sets (real strings, real cell sizes, items
clustered in premises), by simulating scores on them. Part 2 simulates the test itself 200,000 times per setting:
false-positive rate under no difference, and power. Seed SD is not known before models exist; it is a grid.
The first version of this file simulated a different test (review 1, findings 2.1 to 2.7); nothing of it is kept.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]

import numpy as np                                    # noqa: E402
from scipy.stats import norm, rankdata                # noqa: E402

from ftstudy import seedtest                           # noqa: E402

REPS = 200_000


def d_of(auc):                                        # binormal: AUROC = Phi(d / sqrt 2)
    return np.sqrt(2) * norm.ppf(auc)


def cells_stat(y, s, cells, weighted=False):
    """Mean over cells (index arrays) of the AUROC of s for y; weighted by valid x invalid pairs if asked."""
    au, w = [], []
    for c in cells:
        yy = y[c]; n1 = int(yy.sum()); n0 = len(c) - n1
        r = rankdata(s[c])
        au.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)); w.append(n1 * n0)
    return float(np.average(au, weights=w if weighted else None))


def premise_se(y, cell_of, group, auc_x, auc_y=None, weighted=False, min_class=3, n_boot=300, n_rep=6, rng=None):
    """Bootstrap-over-premises SE of the statistic for one simulated model (auc_y None) or of the difference of two
    models scored on the same items. Scores: binormal signal + a premise effect (SD 0.5, the same for the valid and
    the invalid items of a premise, so it is noise for the within-cell ranking) + item noise; the two models share
    half of their item-noise variance. Returned: the mean over n_rep simulated data sets."""
    rng = rng or np.random.default_rng(0)
    ug, inv = np.unique(group, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]

    def stat(idx, sx, sy):
        cid = cell_of[idx]; yy = y[idx]
        cells = []
        for u in np.unique(cid):
            c = np.where(cid == u)[0]
            n1 = int(yy[c].sum())
            if min(n1, len(c) - n1) >= min_class:
                cells.append(c)
        if not cells:
            return np.nan
        a = cells_stat(yy, sx[idx], cells, weighted)
        return a if sy is None else a - cells_stat(yy, sy[idx], cells, weighted)
    out = []
    for _ in range(n_rep):
        shared = rng.standard_normal(len(y)); pe = 0.5 * rng.standard_normal(len(ug))[inv]
        mk = lambda auc: d_of(auc) * y + pe + np.sqrt(0.5) * shared + np.sqrt(0.5) * rng.standard_normal(len(y))
        sx = mk(auc_x); sy = mk(auc_y) if auc_y is not None else None
        full = np.arange(len(y))
        bs = [stat(np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))]), sx, sy) for _ in range(n_boot)]
        out.append(np.nanstd(bs, ddof=1))
        _ = stat(full, sx, sy)
    return float(np.mean(out))


def simulate(delta, n_x, n_y, sd_x, sd_y, tau, paired=False, rng=None, reps=REPS, tau_noise=0.1, alpha=0.05):
    """Rejection rate of the registered test when the true difference is delta, seed SDs are sd_x and sd_y, and the
    premise SE of D is tau (estimated by the bootstrap with a relative error of tau_noise).
    n_y = 0: one condition against a constant.  paired: n_x within-seed differences with SD sd_x."""
    rng = rng or np.random.default_rng(1)
    shift = tau * rng.standard_normal(reps)                         # the evaluation premises are one draw, shared by all models
    x = delta + sd_x * rng.standard_normal((reps, n_x))
    v_prem = (tau * (1 + tau_noise * rng.standard_normal(reps))) ** 2
    if n_y and not paired:
        yv = sd_y * rng.standard_normal((reps, n_y))
        t, df = seedtest.t_and_df(x.mean(1) - yv.mean(1) + shift, x.var(1, ddof=1), n_x, yv.var(1, ddof=1), n_y, v_prem)
    else:
        t, df = seedtest.t_and_df(x.mean(1) + shift, x.var(1, ddof=1), n_x, None, 0, v_prem)
    return float(np.mean(seedtest.p_two_sided(t, df) < alpha))


def main():
    from ftstudy import datasets as S
    from causal_probe import probes
    out = {"reps": REPS, "test": "D / sqrt(s_X^2/n_X + s_Y^2/n_Y + v_premise), Student t, Welch-Satterthwaite df of the seed terms, two-sided 0.05"}
    rng = np.random.default_rng(0)
    # ------------------------------------------------------------------ part 1: premise SE on the built sets
    LB, ID, IDN = S.materialise("eval_pool_lb"), S.materialise("eval_id48"), S.materialise("eval_id")
    spec = json.loads((ROOT / "finetune" / "itemsets" / "build_report.json").read_text())
    tau = {}
    y = np.array([i["label"] for i in LB]); g = np.array([i["group"] for i in LB]); q = np.array([i["query"] for i in LB])
    adj = probes.structural_features(LB)[:, 2].astype(int)
    qa = np.array([f"{a}|{b}" for a, b in zip(q, adj)])
    tau["POOL_LB S, one model at 0.60"] = premise_se(y, q, g, 0.60, rng=rng)
    tau["POOL_LB S, difference 0.60 vs 0.55"] = premise_se(y, q, g, 0.60, 0.55, rng=rng)
    tau["POOL_LB S, difference 0.80 vs 0.70"] = premise_se(y, q, g, 0.80, 0.70, rng=rng)
    tau["POOL_LB S inside string x adjacency cells (pair-weighted), one model at 0.55"] = premise_se(y, qa, g, 0.55, weighted=True, rng=rng)
    tau["POOL_LB S inside cells, difference to a fixed reference score"] = premise_se(y, qa, g, 0.60, 0.55, weighted=True, rng=rng)
    yi = np.array([i["label"] for i in ID]); gi = np.array([i["group"] for i in ID]); qi = np.array([i["query"] for i in ID])
    tau["ID48 S (strings with at least 3 per class), one model at 0.80"] = premise_se(yi, qi, gi, 0.80, rng=rng)
    tau["ID48 S, difference 0.85 vs 0.80"] = premise_se(yi, qi, gi, 0.85, 0.80, rng=rng)
    tau["ID S, same model on two versions of the items (0.80 vs 0.75, item noise 0.9 shared)"] = premise_se(yi, qi, gi, 0.80, 0.75, rng=rng)
    yn = np.array([i["label"] for i in IDN]); gn = np.array([i["group"] for i in IDN])
    tau["ID (natural) pooled AUROC, one model at 0.92 against a fixed reference"] = premise_se(yn, np.zeros(len(yn), int), gn, 0.92, 0.885, rng=rng)
    n_ok = sum(min((yi[qi == s] == 1).sum(), (yi[qi == s] == 0).sum()) >= 3 for s in np.unique(qi))
    out["premise_se"] = {k: round(v, 4) for k, v in tau.items()}
    out["ID48_strings_with_at_least_3_per_class"] = int(n_ok)
    out["sets"] = {"POOL_LB": spec["POOL_LB"], "ID48_items": len(ID), "ID_premises": int(len(np.unique(gi)))}
    t_lb = max(tau["POOL_LB S, difference 0.60 vs 0.55"], tau["POOL_LB S, difference 0.80 vs 0.70"])
    t_cell = tau["POOL_LB S inside string x adjacency cells (pair-weighted), one model at 0.55"]
    t_id = tau["ID48 S, difference 0.85 vs 0.80"]; t_id1 = tau["ID48 S (strings with at least 3 per class), one model at 0.80"]

    # ------------------------------------------------------------------ part 2: the test
    SDS = (0.005, 0.01, 0.02, 0.03, 0.04)
    size = {}
    for nx, ny in ((5, 5), (5, 3), (3, 3), (5, 0), (3, 0)):
        for sd in SDS:
            for tv in (0.0, 0.005, 0.015, 0.03):
                size[f"n={nx},{ny} seed_sd={sd} premise_se={tv}"] = simulate(0.0, nx, ny, sd, sd, tv, rng=rng)
        size[f"n={nx},{ny} unequal seed_sd=0.01/0.04 premise_se=0.01"] = simulate(0.0, nx, ny, 0.01, 0.04, 0.01, rng=rng)
        size[f"n={nx},{ny} unequal seed_sd=0.04/0.01 premise_se=0.01"] = simulate(0.0, nx, ny, 0.04, 0.01, 0.01, rng=rng)
    out["false_positive_rate"] = size
    out["false_positive_rate_max"] = max(size.values())
    out["false_positive_rate_max_at"] = max(size, key=size.get)

    def grid(nx, ny, tv, deltas, paired=False, alpha=0.05):
        return {f"delta={d} seed_sd={sd}": simulate(d, nx, ny, sd, sd, tv, paired=paired, rng=rng, alpha=alpha) for d in deltas for sd in SDS}
    out["power"] = {
        "E2a  S(B) - S(A) on POOL-LB, 5 vs 5": {"premise_se": round(t_lb, 4), **grid(5, 5, t_lb, (0.03, 0.05, 0.07, 0.10))},
        "E2b  S(B) - S(A48) on POOL-LB, 5 vs 3": {"premise_se": round(t_lb, 4), **grid(5, 3, t_lb, (0.05, 0.07, 0.10, 0.15))},
        "E2c  S(Bswap) - 0.5 on POOL-LB, 3 seeds (a control: 'power' is the chance of catching a leak of delta)":
            {"premise_se": round(tau["POOL_LB S, one model at 0.60"], 4), **grid(3, 0, tau["POOL_LB S, one model at 0.60"], (0.03, 0.05, 0.10))},
        "E3   S(A) - S(C) on ID48, 5 vs 3": {"premise_se": round(t_id, 4), **grid(5, 3, t_id, (0.05, 0.10, 0.15, 0.20))},
        "E4   within-seed drop of A on ID48 (5 paired differences; seed_sd is the SD of the differences)":
            {"premise_se": round(t_id, 4), **grid(5, 0, t_id, (0.03, 0.05, 0.10), paired=True)},
        "E5a  S inside cells - 0.5 on POOL-LB, 5 seeds": {"premise_se": round(t_cell, 4), **grid(5, 0, t_cell, (0.03, 0.05, 0.10))},
        "E5b  S(model) - S(nearest-premise copy) on POOL-LB, 5 seeds (the reference is a fixed score)":
            {"premise_se": round(t_lb, 4), **grid(5, 0, t_lb, (0.03, 0.05, 0.10))},
        "E1   pooled AUROC of A on ID minus the hypothesis-string prior (a fixed reference), 5 seeds":
            {"premise_se": round(tau["ID (natural) pooled AUROC, one model at 0.92 against a fixed reference"], 4),
             **grid(5, 0, tau["ID (natural) pooled AUROC, one model at 0.92 against a fixed reference"], (0.02, 0.03, 0.05))},
    }
    # the family of eleven tests is claimed on Holm-adjusted p: power at the strictest level a test can meet, 0.05 / 11
    FAM, DL = 11, (0.05, 0.07, 0.10, 0.15, 0.20)
    al = 0.05 / FAM
    out["family_size"] = FAM
    out["power_at_alpha_over_family"] = {
        "E2a  S(B) - S(A) on POOL-LB, 5 vs 5": grid(5, 5, t_lb, DL, alpha=al),
        "E2b  S(B) - S(A48) on POOL-LB, 5 vs 3": grid(5, 3, t_lb, DL, alpha=al),
        "E3   S(A) - S(C) on ID48, 5 vs 3": grid(5, 3, t_id, DL, alpha=al),
        "E4   within-seed difference of A on ID48, 5 seeds": grid(5, 0, t_id, DL, paired=True, alpha=al),
        "E5a  S inside cells - 0.5 on POOL-LB, 5 seeds": grid(5, 0, t_cell, DL, alpha=al),
        "E5bc S(model) - S(a fixed reference) on POOL-LB, 5 seeds": grid(5, 0, t_lb, DL, alpha=al),
    }
    out["registered_effect"] = {}
    for k, v in out["power_at_alpha_over_family"].items():
        ok = [d for d in DL if v[f"delta={d} seed_sd=0.02"] >= 0.6]
        out["registered_effect"][k] = {"smallest_delta_with_power_0.6_at_seed_sd_0.02": ok[0] if ok else None,
                                       **{f"power_there_seed_sd={sd}": v[f"delta={ok[0]} seed_sd={sd}"] for sd in (0.01, 0.02, 0.03) if ok}}
    out["unequal_seed_sd_no_premise_term_5v3"] = {"sd 0.01 in the 5, 0.04 in the 3": simulate(0.0, 5, 3, 0.01, 0.04, 0.0, rng=rng),
                                                  "sd 0.04 in the 5, 0.01 in the 3": simulate(0.0, 5, 3, 0.04, 0.01, 0.0, rng=rng)}
    p = ROOT / "finetune" / "power" / "power.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("premise_se", "false_positive_rate_max", "false_positive_rate_max_at")}, indent=1))
    for k, v in out["registered_effect"].items():
        print(k, {a: round(b, 2) if b is not None else None for a, b in v.items()})
    print(out["unequal_seed_sd_no_premise_term_5v3"])


if __name__ == "__main__":
    main()

"""Power calculations for finetune/PREREG.md. CPU, a few minutes. Writes finetune/power/power.json.

Every number here is computed BEFORE any fine-tuned model exists, from design sizes and from v1.0 result
files only. Assumptions are arguments, printed with the result; PREREG.md quotes them.

  A. within-hypothesis-string AUROC, difference between two conditions on the same items (E2, E5)
  B. a comparison that rests on few premises (the official test split and the authors' perturbed sets: 21) (E4)
  C. steering: gold slope against 500 matched controls, using v1.0's control spread (E7)
  D. probe minus text reference, from v1.0's interval widths (E6)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REPS = 400


def d_of(auc):                     # binormal: AUROC = Phi(d / sqrt 2)
    return np.sqrt(2) * norm.ppf(auc)


def mean_within_string_auc(y, s, strings):
    """Mean over strings of the AUROC; every string has the same number of items, first half valid (rank formula;
    scores are continuous, so there are no ties)."""
    k = len(np.unique(strings)); S = s.reshape(k, -1); n = S.shape[1] // 2
    ranks = S.argsort(1).argsort(1) + 1
    return float(((ranks[:, :n].sum(1) - n * (n + 1) / 2) / (n * n)).mean())


def power_within_string(auc_a, auc_b, n_strings, per_cell, rho=0.5, seed_sd=0.0, n_seeds=3, alpha=0.05, rng=None):
    """Two conditions scored on the same items (n_strings strings, per_cell valid + per_cell invalid each).
    Item noise is correlated rho between conditions. Each condition's AUROC is the mean of n_seeds models
    whose true AUROC varies with SD seed_sd. Test: two-sided z on the difference with the Monte-Carlo SE under
    this design (what a paired premise bootstrap estimates)."""
    rng = rng or np.random.default_rng(0)
    y = np.tile(np.r_[np.ones(per_cell), np.zeros(per_cell)], n_strings).astype(int)
    strings = np.repeat(np.arange(n_strings), 2 * per_cell)
    diffs = []
    for _ in range(REPS):
        e = rng.standard_normal(len(y))
        est = []
        for auc in (auc_a, auc_b):
            vals = []
            for _s in range(n_seeds):
                a = float(np.clip(auc + seed_sd * rng.standard_normal(), 0.5, 0.995))
                noise = rho * e + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(y))
                vals.append(mean_within_string_auc(y, d_of(a) * y + noise, strings))
            est.append(np.mean(vals))
        diffs.append(est[1] - est[0])
    diffs = np.array(diffs)
    se = diffs.std(ddof=1)
    return {"power": float(np.mean(np.abs(diffs) / se > norm.ppf(1 - alpha / 2))), "se_of_difference": float(se)}


def power_few_premises(auc_1, auc_2, n_prem, per_prem, pos_rate, prem_sd, alpha=0.05, rng=None, n_boot=200):
    """Pooled AUROC on n_prem premises, two score sets on the same items (e.g. original and perturbed, or two
    conditions); each premise shifts the signal strength of its items by N(0, prem_sd) in d units, shared between
    the two score sets. Test: premise bootstrap interval of the difference excludes 0."""
    rng = rng or np.random.default_rng(1)
    hits = 0
    for _ in range(REPS // 2):
        ys, s1, s2, g = [], [], [], []
        for p in range(n_prem):
            y = (rng.random(per_prem) < pos_rate).astype(int)
            u = prem_sd * rng.standard_normal()
            e = rng.standard_normal(per_prem)
            s1.append((d_of(auc_1) + u) * y + 0.7 * e + 0.714 * rng.standard_normal(per_prem))
            s2.append((d_of(auc_2) + u) * y + 0.7 * e + 0.714 * rng.standard_normal(per_prem))
            ys.append(y); g.append(np.full(per_prem, p))
        ys, s1, s2 = map(np.concatenate, (ys, s1, s2))
        idx = [np.arange(p * per_prem, (p + 1) * per_prem) for p in range(n_prem)]
        bs = []
        for _b in range(n_boot):
            ii = np.concatenate([idx[k] for k in rng.integers(0, n_prem, n_prem)])
            if ys[ii].min() == ys[ii].max():
                continue
            bs.append(roc_auc_score(ys[ii], s1[ii]) - roc_auc_score(ys[ii], s2[ii]))
        lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
        hits += (lo > 0) or (hi < 0)
    return {"power": hits / (REPS // 2)}


def power_steering(r, gold_noise=0.3, n_controls=500, alpha=0.05, rng=None):
    """Gold slope = r control SDs (true), measured with noise of gold_noise control SDs (v1.0: the 3B gold slope was
    0.738 and 0.569 on two fits, control SD 0.397 -> about 0.3). Controls ~ N(0, 1) (v1.0: the largest of 500 was
    3.2 SD, as a normal gives). One-sided p = (1 + #controls >= gold) / (n_controls + 1)."""
    rng = rng or np.random.default_rng(2)
    g = r + gold_noise * rng.standard_normal(4000)
    c = rng.standard_normal((4000, n_controls))
    p = (1 + (c >= g[:, None]).sum(1)) / (n_controls + 1)
    return float(np.mean(p <= alpha))


def main():
    out = {"reps": REPS}
    # A. Evaluation pool sizes: the v1.0 pool has 48 strings x 25 pairs; a length-balanced pool is assumed to keep
    #    between 12 and 25 pairs per string (PREREG fixes the real number after the data build, before any training).
    A = {}
    for per_cell in (12, 18, 25):
        for auc_a, auc_b in ((0.55, 0.60), (0.55, 0.65), (0.55, 0.70), (0.50, 0.55)):
            for seed_sd in (0.0, 0.03):
                A[f"pairs_per_string={per_cell} A={auc_a} B={auc_b} seed_sd={seed_sd}"] = \
                    power_within_string(auc_a, auc_b, 48, per_cell, seed_sd=seed_sd)
    out["A_within_string_difference"] = A
    # one model against chance (E5, inside cells of string x adjacency: roughly half the items are in two-class cells)
    out["A_one_model_vs_half"] = {f"pairs_per_string={pc} auc={a}": power_within_string(0.5, a, 48, pc, rho=0.0, n_seeds=1)
                                  for pc in (6, 12) for a in (0.55, 0.60)}
    # B. 21 premises, about 55 items each (official test: 1,162 items) and 107 each (perturbed sets: 2,246)
    B = {}
    for per_prem, pos in ((55, 0.155), (107, 0.152)):
        for a1, a2 in ((0.95, 0.85), (0.95, 0.90), (0.90, 0.87)):
            B[f"items_per_premise={per_prem} {a1} vs {a2}"] = power_few_premises(a1, a2, 21, per_prem, pos, prem_sd=0.5)
    out["B_21_premises"] = B
    # C. steering
    out["C_steering"] = {f"alpha={al} r={r}": power_steering(r, alpha=al) for al in (0.05, 0.025) for r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)}
    v1 = {}
    for m in ("qwen3b", "qwen1.5b"):
        pt = json.loads((ROOT / "results" / f"steering_settle_{m}_c2c_debiased_hyp.json").read_text())["primary_test"]
        v1[m] = {"gold_slope": pt["slope"], "control_sd": pt["control_sd"], "control_max": pt["control_max"],
                 "r_observed": abs(pt["z_vs_controls"]), "p_one_sided": pt["p_one_sided"]}
    out["C_v1_reference"] = v1
    # D. probe minus tuned text stack: v1.0's Gemma-2-9B interval on the report half was +0.80 [-0.08, +1.68] points
    se = (1.68 + 0.08) / 2 / 1.96
    out["D_probe_minus_text"] = {"se_points_report_half_v1": round(se, 3),
                                 **{f"gain={g} points, alpha=0.05/3": float(norm.cdf(g / se - norm.ppf(1 - 0.05 / 6))) for g in (1, 2, 3, 5)}}
    p = ROOT / "finetune" / "power" / "power.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        if isinstance(v, dict):
            print(k)
            for kk, vv in v.items():
                print("   ", kk, vv if not isinstance(vv, dict) else {a: round(b, 3) for a, b in vv.items()})


if __name__ == "__main__":
    main()

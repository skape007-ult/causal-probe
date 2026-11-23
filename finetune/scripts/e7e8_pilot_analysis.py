"""E7/E8 pilot analysis and the power rule (PREREG Amendment 3, Section 4). Laptop, no model.

  ../causal-probe/.venv/bin/python finetune/scripts/e7e8_pilot_analysis.py --ld <dir with <run>/ld.npy> [--synthetic]
    -> finetune/results/e7e8_pilot.json (pilot numbers, seed-SD bound, power table, N_items, dropped tests)

The pilot's X values enter only through the seed-SD estimate (Section 4 [C]); nothing else is chosen from them.
Column layout of ld.npy: 0 zero vector; 1..92 steering, 4 steps (-2, -1, +1, +2) for each of g, d_gold, d_ans and the
20 controls in that order; 93..96 patches (projected opposite, projected same, full opposite, full same).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]
import numpy as np                       # noqa: E402

from ftstudy import datasets as DS, e7e8 as E   # noqa: E402

PILOT = ["A_s0", "A_s1", "B_s0", "B_s1"]


def steps(LD, j):
    """items x 5 (alphas -2, -1, 0, +1, +2) for direction number j."""
    c = 1 + 4 * j
    return np.column_stack([LD[:, c], LD[:, c + 1], LD[:, 0], LD[:, c + 2], LD[:, c + 3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ld", required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    tag = "_synthetic" if a.synthetic else ""
    Z = np.load(ROOT / "finetune" / "e7e8" / f"pilot_inputs{tag}.npz")
    prep = json.loads((ROOT / "finetune" / "e7e8" / f"pilot_prepare{tag}.json").read_text())
    items = DS.materialise("eval_pool_lb")
    y = np.array([i["label"] for i in items]); groups = np.array([i["group"] for i in items])
    from causal_probe import data
    rep_idx = np.where(~data.selection_mask(items))[0]
    idx, D = Z["pilot_idx"], Z["donors"]
    gi = groups[idx]
    res = {"models": {}, "note": "pilot numbers; not a test"}
    X7, X8, se7, se8 = {"A": [], "B": []}, {"A": [], "B": []}, {"A": [], "B": []}, {"A": [], "B": []}
    for run in PILOT:
        f = Path(a.ld) / run / "ld.npy"
        ck = json.loads((Path(a.ld) / run / "checks.json").read_text())
        if ck["status"] != "OK":
            res["models"][run] = {"failed_checks": ck}
            continue
        LD = np.load(f).astype(float)
        if a.synthetic and LD.shape[0] < len(idx):
            idx_r, D_r, g_r = idx[: LD.shape[0]], D[: LD.shape[0]], gi[: LD.shape[0]]
        else:
            idx_r, D_r, g_r = idx, D, gi
        s_ld = float(np.std(LD[:, 0], ddof=1))
        b = np.array([E.slopes(steps(LD, j)) for j in range(23)])          # 23 x items
        x7 = E.e7_items(b[0], b[3:].T, s_ld)
        beta = {"g": float(b[0].mean() / s_ld), "d_gold": float(b[1].mean() / s_ld), "d_ans": float(b[2].mean() / s_ld),
                "controls_mean": float(b[3:].mean() / s_ld)}
        ok8 = (D_r[:, 0] >= 0) & (D_r[:, 1] >= 0)
        s_ld8 = float(np.std(LD[ok8, 0], ddof=1)) if ok8.sum() > 2 else float("nan")
        x8 = E.e8_items(LD[ok8, 0], LD[ok8, 93], LD[ok8, 94], y[D_r[ok8, 0]], s_ld8)
        x8_full = E.e8_items(LD[ok8, 0], LD[ok8, 95], LD[ok8, 96], y[D_r[ok8, 0]], s_ld8)
        c = run[0]
        X7[c].append(float(x7.mean())); X8[c].append(float(x8.mean()))
        se7[c].append(E.premise_se(x7, g_r)); se8[c].append(E.premise_se(x8, g_r[ok8]))
        rank = int((b[3:].mean(1) >= b[0].mean()).sum())
        res["models"][run] = {"s_LD": s_ld, "beta": beta, "X7": float(x7.mean()),
                              "X7_over_beta_d_ans": float(x7.mean() / beta["d_ans"]) if beta["d_ans"] else None,
                              "rank_of_g_among_controls_from_top": rank + 1, "item_sd_x7": float(x7.std(ddof=1)),
                              "premise_se_x7": se7[c][-1], "X8_projected": float(x8.mean()), "X8_full_descriptive": float(x8_full.mean()),
                              "item_sd_x8": float(x8.std(ddof=1)), "premise_se_x8": se8[c][-1],
                              "share_items_skipped_e8": float(1 - ok8.mean()), "checks": ck}
    sd7, sd8 = E.seed_sd_bound(X7), E.seed_sd_bound(X8)
    n_prem_pilot = prep["n_pilot_premises"]
    n_rep_left = int(len(rep_idx) - len(idx))
    ipp = len(rep_idx) / len(np.unique(groups[rep_idx]))
    tests = {}
    for c in ("A", "B"):
        if se7[c]:
            tests[f"E7({c})"] = float(np.mean(se7[c]))
        if se8[c]:
            tests[f"E8({c})"] = float(np.mean(se8[c]))
    skip8 = {c: max([res["models"][r]["share_items_skipped_e8"] for r in PILOT if r[0] == c and "s_LD" in res["models"][r]] or [1.0]) for c in "AB"}
    pw = {}
    for name, se in tests.items():
        sd = sd7 if name.startswith("E7") else sd8
        pw[name] = E.choose_n_items({name: se}, n_rep_left, ipp, n_prem_pilot, sd["used"])
    kept = [v["N_items"] for v in pw.values() if v["N_items"] is not None]
    res.update({"seed_sd_E7": sd7, "seed_sd_E8": sd8, "premise_se_pilot": tests, "items_per_premise": ipp,
                "pilot_premises": n_prem_pilot, "report_items_left": n_rep_left, "power": pw,
                "N_items": max(kept) if kept else None,
                "dropped": [k for k, v in pw.items() if v["N_items"] is None],
                "e8_descriptive_only_skip_over_30pct": {c: bool(v > 0.30) for c, v in skip8.items()}})
    p = ROOT / "finetune" / "results" / f"e7e8_pilot{tag}.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("N_items", "dropped", "seed_sd_E7", "seed_sd_E8", "premise_se_pilot", "e8_descriptive_only_skip_over_30pct")}, indent=1))


if __name__ == "__main__":
    main()

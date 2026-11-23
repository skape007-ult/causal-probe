"""E7/E8 pilot, laptop side (PREREG Amendment 3): site, directions, controls, pilot items, donors, and every vector
the GPU run adds. Needs E6's result file and activations; no model.

  OMP_NUM_THREADS=2 ../causal-probe/.venv/bin/python finetune/scripts/e7e8_prepare.py [--synthetic]
    -> finetune/e7e8/pilot_inputs.npz (committed: the notebook reads it), finetune/e7e8/pilot_prepare.json

Per pilot model (A_s0, A_s1, B_s0, B_s1), at (content_end, L_c):
  directions g, d_gold, d_ans, d_text, 20 controls (selection half), sigma of each; diagnostics (Section 2 [B, c3])
  pilot items: 200 report-half items as whole premises (seed 0); donors from the other report-half items
  vectors per item: alpha x sigma_v x v for alpha in (-2, -1, 1, 2) and v in (g, d_gold, d_ans, controls); the two
  projected patches ((h_j - h_i).g) g and the two full patches (h_j - h_i), opposite- and same-label donor
  check data: stored POOL-LB score of each pilot item; stored h at the site for the first 20 pilot items
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune"), str(ROOT / "finetune" / "scripts")]
import numpy as np                                    # noqa: E402

from causal_probe import data, probes                 # noqa: E402
from ftstudy import behaviour as BH, datasets as DS, e7e8 as E   # noqa: E402

PILOT = ["A_s0", "A_s1", "B_s0", "B_s1"]
N_PILOT = 200
POS = 1                                    # content_end in E6's resid (positions final, content_end)
OUT = ROOT / "finetune" / "e7e8"


def own_answer_S(Xs_sel, a_sel, Xs_rep, a_rep, q_rep, n_rep, g_sel):
    s = probes.fit_predict(Xs_sel, a_sel, Xs_rep, C=None, groups_tr=g_sel)
    return float(BH.Plan(a_rep, q_rep, n_rep).S(s))


def oof_text_sign(sel_items):
    import text_references as TR
    folds = data.grouped_folds(sel_items, 5, seed=0)
    out = np.zeros(len(sel_items))
    for tr, te in folds:
        a, b = TR.Side([sel_items[i] for i in tr]), TR.Side([sel_items[i] for i in te])
        comp = TR.components(a, b)
        out[np.asarray(te)], _, _ = TR.tuned_stack(a, b, comp)
    return (out > 0).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="planted activations and a planted E6 file (test only)")
    a = ap.parse_args()
    items = DS.materialise("eval_pool_lb")
    base = np.load(ROOT / "finetune" / "runs" / "base" / "scores.npz")
    y = np.array([i["label"] for i in items]); q = np.array([i["query"] for i in items])
    n = base["eval_pool_lb__n_tokens"].astype(float); groups = np.array([i["group"] for i in items])
    adj = probes.structural_features(items)[:, 2]
    sel = data.selection_mask(items); rep = ~sel
    si, ri = np.where(sel)[0], np.where(rep)[0]
    rng = np.random.default_rng(0)

    if a.synthetic:
        L = 29
        e6 = {"runs": {r: {"layers": list(range(L)), "positions": {"content_end": {"S_selection_oof":
              [0.5] * 9 + [0.8] * 20}}} for r in PILOT + ["A_s2", "A_s3", "A_s4", "B_s2", "B_s3", "B_s4"]}}
    else:
        e6 = json.loads((ROOT / "finetune" / "results" / "e6_descriptive.json").read_text())
    site = {}
    for c in ("A", "B"):
        runs = [f"{c}_s{s}" for s in range(5)]
        S = np.array([e6["runs"][r]["positions"]["content_end"]["S_selection_oof"] for r in runs])
        site[c] = E.site_layer(S, e6["runs"][runs[0]]["layers"])
    print("site layers (content_end):", site, flush=True)

    text_sign = oof_text_sign([items[i] for i in si])
    pilot_idx = E.take_premises(groups, ri, N_PILOT, seed=0)
    pilot_prem = set(groups[pilot_idx])
    pool = np.array([i for i in ri if groups[i] not in pilot_prem])
    D = E.donors(pilot_idx, pool, y, q, adj, groups, n, seed=0)
    out, meta = {"pilot_idx": pilot_idx, "donors": D}, {"site_layer": site, "n_pilot_items": int(len(pilot_idx)),
                                                        "n_pilot_premises": len(pilot_prem), "models": {}}
    for run in PILOT:
        c = run[0]
        L_c = site[c]
        if L_c is None:
            meta["models"][run] = {"not_run": "no gold signal at the site on the selection half"}
            continue
        if a.synthetic:
            R = rng.normal(size=(len(items), 29, 2, 32)).astype(np.float32)
            R[:, :, POS, 0] += (2 * y - 1)[:, None]
            ld = (2 * y - 1) * 0.5 + rng.normal(size=len(y))
        else:
            R = np.load(ROOT / "finetune" / "e6" / "data" / run / "resid.npy", mmap_mode="r")
            ld = np.array(json.loads((ROOT / "finetune" / "e6" / "data" / run / "meta.json").read_text())["answer_logit_diff"])
            st = np.load(ROOT / "finetune" / "runs" / run / "scores.npz")["eval_pool_lb__score"]
            assert np.abs(st - ld).max() < 0.05
        H = np.asarray(R[:, L_c, POS, :], np.float32).astype(float)
        own = (ld > 0).astype(int)
        V, sig, diag = E.directions(H[si], y[si], own[si], text_sign, q[si])
        diag["separation_g_report"] = E.separation(H[ri], y[ri], V["g"], sig["g"])
        diag["separation_controls_report_median"] = float(np.median([E.separation(H[ri], y[ri], v, s) for v, s in zip(V["controls"], sig["controls"])]))
        Hf = np.asarray(R[:, -1, 0, :], np.float32).astype(float)
        diag["own_answer_S_site"] = own_answer_S(H[si], own[si], H[ri], own[ri], q[ri], n[ri], groups[si])
        diag["own_answer_S_final_last_layer"] = own_answer_S(Hf[si], own[si], Hf[ri], own[ri], q[ri], n[ri], groups[si])
        hg = H[ri] @ V["g"]
        from e6_analysis import within_string_spearman
        diag["loading_g"] = {"answer": within_string_spearman(hg, ld[ri], q[ri]), "log_tokens": within_string_spearman(hg, np.log(n[ri]), q[ri]),
                             "adjacency": within_string_spearman(hg, adj[ri].astype(float), q[ri])}
        dirs = [V["g"], V["d_gold"], V["d_ans"]] + list(V["controls"])
        sigs = [sig["g"], sig["d_gold"], sig["d_ans"]] + list(sig["controls"])
        steer = np.array([al * s * v for v, s in zip(dirs, sigs) for al in (-2.0, -1.0, 1.0, 2.0)])        # (23*4) x d
        hi = H[pilot_idx]
        patch = np.zeros((len(pilot_idx), 4, H.shape[1]))
        for k, (j, jj) in enumerate(D):
            for col, dn in ((0, j), (1, jj)):
                if dn >= 0:
                    dv = H[dn] - hi[k]
                    patch[k, col] = (dv @ V["g"]) * V["g"]
                    patch[k, 2 + col] = dv
        out[f"{run}|steer"] = steer.astype(np.float16)
        out[f"{run}|patch"] = patch.astype(np.float16)
        out[f"{run}|stored_ld"] = ld[pilot_idx].astype(np.float32)
        out[f"{run}|h_check"] = hi[:20].astype(np.float32)
        out[f"{run}|check_dirs"] = np.array([V["g"]] + list(V["controls"]), np.float32)
        out[f"{run}|check_sigma"] = np.array([sig["g"]] + list(sig["controls"]), np.float32)
        meta["models"][run] = {"layer": L_c, "diagnostics": diag, "sigma": {k: v for k, v in sig.items() if k != "controls"}}
        print(run, "layer", L_c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in diag.items() if not isinstance(v, dict)}, flush=True)
    meta["n_donor_missing"] = {"opposite": int((D[:, 0] < 0).sum()), "same": int((D[:, 1] < 0).sum())}
    OUT.mkdir(exist_ok=True)
    tag = "_synthetic" if a.synthetic else ""
    np.savez_compressed(OUT / f"pilot_inputs{tag}.npz", **out)
    (OUT / f"pilot_prepare{tag}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", OUT / f"pilot_inputs{tag}.npz", "| donors missing:", meta["n_donor_missing"])


if __name__ == "__main__":
    main()

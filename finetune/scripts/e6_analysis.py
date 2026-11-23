"""E6 analysis (PREREG Amendment 2): the probe ladder on POOL-LB for base, A_s0-4, B_s0-4. Descriptive: no test, no p.

  OMP_NUM_THREADS=2 caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/e6_analysis.py
  ../causal-probe/.venv/bin/python finetune/scripts/e6_analysis.py --synthetic       # planted activations, no model

Inputs: finetune/e6/data/<run>/{resid.npy, meta.json} (from the E6 notebook; git-ignored), the stored scores in
finetune/runs/, the registered references (finetune/itemsets/, finetune/results/.refs_cache.pkl via behaviour.py).
Outputs: finetune/results/e6_<mode>.json and e6_<mode>_tables.md (regenerated from the json).

What is computed, as Amendment 2 fixes it:
  halves     v1.0 data.selection_mask on the POOL-LB items (a seeded half of the premises)
  probe      v1.0 probes.fit_predict, C from (0.01, 0.1, 1) by premise-grouped inner CV on the selection half, gold
             label, trained on the selection half, scored on the report half; per run, layer and position
  S          stat.S's statistic (via the vectorised Plan of ftstudy.behaviour, checked against stat.S) on the
             report half; v1.0 accuracy beside it
  layer      chosen per run on the selection half: largest S of premise-grouped 3-fold out-of-fold scores inside the
             selection half, at the C the inner CV chose (implementation detail of "largest selection-half S")
  references own answer; E5c text references as registered (A, B); the same families refitted on the selection
             half; memorisation reference of the run's training set (E5b member). All on the report half.
  intervals  Section 3's form, seeds pooled: D +- t x sqrt(seed variance / n + premise variance); premise variance from
             premise resamples of the report half (N_BOOT_CURVE for the curves, stat.N_BOOT for the summaries), S
             recomputed on every resample (eligibility and length fit included), seed mean inside each resample
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune"), str(ROOT / "finetune" / "scripts")]

import numpy as np                                             # noqa: E402
from scipy.stats import spearmanr                              # noqa: E402

from causal_probe import data, probes, stats                   # noqa: E402
from scipy import stats as sps                                 # noqa: E402
from ftstudy import behaviour as BH, datasets as DS, seedtest, stat   # noqa: E402

DATA = ROOT / "finetune" / "e6" / "data"
OUT = ROOT / "finetune" / "results"
RUNS = ["base"] + [f"{c}_s{s}" for c in ("A", "B") for s in range(5)]
COND = {"base": ["base"], "A": [f"A_s{s}" for s in range(5)], "B": [f"B_s{s}" for s in range(5)]}
POSITIONS = ("final", "content_end")
MAX_DIFF = 0.05                    # Amendment 2, item 2
N_BOOT_CURVE = 500
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)


def S_vec(e: BH.Plan, M):
    return e.S(np.asarray(M, float))


def boot_sets(y, q, n, groups, adj, n_boot, seed=0):
    """Premise resamples of the report half: each yields (index array, Plan built on the resampled items)."""
    out = []
    for idx in stat.boot_indices(groups, n_boot, seed):
        out.append((idx, BH.Plan(y[idx], q[idx], n[idx], adj[idx])))
    return out


def within_string_spearman(a, b, q):
    r, w = [], []
    for u in np.unique(q):
        c = q == u
        if c.sum() >= 5 and np.ptp(a[c]) > 0 and np.ptp(b[c]) > 0:
            r.append(spearmanr(a[c], b[c])[0]); w.append(c.sum())
    return float(np.average(r, weights=w)) if r else float("nan")


def load_run(run, items, synthetic, rng):
    if synthetic:
        y = np.array([i["label"] for i in items])
        L, d = 29, 24
        R = rng.normal(size=(len(items), L, 2, d)).astype(np.float16)
        if run != "base":                                    # planted: gold readable from layer 12 on; B more than A
            amp = (0.6 if run.startswith("A") else 1.2)
            for l in range(12, L):
                R[:, l, :, 0] += (amp * (2 * y - 1))[:, None].astype(np.float16)
        diff = (2 * y - 1) * (0.3 if run != "base" else 0.0) + rng.normal(size=len(y))
        return R, {"answer_logit_diff": diff.tolist(), "uids": [i["uid"] for i in items], "max_abs_diff_vs_stored_scores": 0.0}
    meta = json.loads((DATA / run / "meta.json").read_text())
    if meta["uids"] != [i["uid"] for i in items]:
        raise RuntimeError(f"{run}: activations are not in POOL-LB order")
    st = np.load(ROOT / "finetune" / "runs" / run / "scores.npz")
    if [str(u) for u in st["eval_pool_lb__uid"]] != meta["uids"]:
        raise RuntimeError(f"{run}: stored scores are not in POOL-LB order")
    meta["max_abs_diff_vs_stored_scores"] = float(np.abs(st["eval_pool_lb__score"] - np.array(meta["answer_logit_diff"])).max())
    if meta["max_abs_diff_vs_stored_scores"] > MAX_DIFF:
        raise RuntimeError(f"{run}: recomputed answers differ from the stored scores by {meta['max_abs_diff_vs_stored_scores']:.3f} "
                           f"(> {MAX_DIFF}); Amendment 2 item 2: its activations are not used")
    R = np.load(DATA / run / "resid.npy")
    return R, meta


def probe_cell(X, y, sel, rep, groups, sel_folds, e_sel_parts):
    """-> report-half scores, selection-half out-of-fold scores, chosen C."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    Xs, Xr = X[sel], X[rep]
    sc = StandardScaler().fit(Xs)
    clf = probes._fit(sc.transform(Xs), y[sel], None, groups[sel], probes.PROBE_CS, 300)
    C = float(clf.C_[0]) if isinstance(clf, LogisticRegressionCV) else float(clf.C)
    s_rep = clf.decision_function(sc.transform(Xr))
    s_oof = probes.oof_scores(Xs, y[sel], sel_folds, C=C)
    return s_rep, s_oof, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layer-stride", type=int, default=1, help="1 as registered; >1 only for a quick look")
    a = ap.parse_args()
    mode = "synthetic" if a.synthetic else "descriptive"
    rng = np.random.default_rng(0)
    items = DS.materialise("eval_pool_lb")
    base_sc = np.load(ROOT / "finetune" / "runs" / "base" / "scores.npz")
    assert [str(u) for u in base_sc["eval_pool_lb__uid"]] == [i["uid"] for i in items]
    y = np.array([i["label"] for i in items]); q = np.array([i["query"] for i in items])
    n = base_sc["eval_pool_lb__n_tokens"].astype(float); groups = np.array([i["group"] for i in items])
    adj = probes.structural_features(items)[:, 2]
    sel = data.selection_mask(items); rep = ~sel
    yr, qr, nr, gr, ar = y[rep], q[rep], n[rep], groups[rep], adj[rep]
    e_rep = BH.Plan(yr, qr, nr, ar)
    e_sel = BH.Plan(y[sel], q[sel], n[sel])
    sel_items = [items[i] for i in np.where(sel)[0]]
    sel_folds = [(np.asarray(tr), np.asarray(te)) for tr, te in data.grouped_folds(sel_items, 3, seed=0)]
    log(f"POOL-LB {len(items)} items: selection {sel.sum()}, report {rep.sum()} ({len(np.unique(gr))} premises)")

    # ---------------------------------------------------------------- references on the report half
    import behaviour as BS                                           # finetune/scripts/behaviour.py
    import text_references as TR
    sets, its = BS.build_sets(ROOT / "finetune" / "runs" / "base" / "scores.npz")
    refs = BS.cached_refs(sets, its)
    tz = np.load(ROOT / "finetune" / "itemsets" / "text_references.npz")
    ref_scores = {}
    for c in ("A", "B"):
        for nm in ("lookup", "hand_parse", "tuned_stack", "premise_counts_logistic", "premise_counts_boosting"):
            ref_scores[f"E5c_registered|{c}|{nm}"] = tz[f"{c}|pool_lb|{nm}"][rep]
        ref_scores[f"memorisation_E5b|{c}"] = np.asarray(refs["mem"][c]["pool_lb"])[rep]
    tr_side, te_side = TR.Side(sel_items), TR.Side([items[i] for i in np.where(rep)[0]])
    comp = TR.components(tr_side, te_side)
    ref_scores["selection_half_fit|lookup"] = comp["lookup"]
    ref_scores["selection_half_fit|hand_parse"] = comp["hand_parse"]
    ref_scores["selection_half_fit|tuned_stack"], C_stack, _ = TR.tuned_stack(tr_side, te_side, comp)
    ref_scores["selection_half_fit|premise_counts_logistic"] = DS.premise_counts_within_string(tr_side.items, te_side.items)
    ref_scores["selection_half_fit|premise_counts_boosting"] = DS.premise_counts_within_string(tr_side.items, te_side.items, boosting=True)
    ref_S = {k: float(S_vec(e_rep, v)) for k, v in ref_scores.items()}
    chk = {k: abs(ref_S[k] - stat.S(yr, v, qr, nr)) for k, v in ref_scores.items()}
    assert max(chk.values()) < 1e-4, chk
    log("references:", {k: round(v, 3) for k, v in ref_S.items()})

    # ---------------------------------------------------------------- probes
    per_run, scores_rep = {}, {}
    for run in RUNS:
        R, meta = load_run(run, items, a.synthetic, rng)
        L = R.shape[1]
        layers = list(range(0, L, a.layer_stride))
        ans = np.array(meta["answer_logit_diff"], float)
        r = {"max_abs_diff_vs_stored_scores": meta.get("max_abs_diff_vs_stored_scores"), "layers": layers,
             "answer_S_report": float(S_vec(e_rep, ans[rep])), "answer_accuracy_report": stats.accuracy(yr, (ans[rep] > 0).astype(int)),
             "positions": {}}
        for p, pname in enumerate(POSITIONS):
            Srep, Ssel, acc, Cs, M = [], [], [], [], []
            for l in layers:
                X = np.asarray(R[:, l, p, :], np.float32)
                s_rep, s_oof, C = probe_cell(X, y, sel, rep, groups, sel_folds, None)
                M.append(s_rep); Cs.append(C)
                Ssel.append(float(e_sel.S(s_oof)))
                acc.append(stats.accuracy(yr, (s_rep > 0).astype(int)))
            M = np.column_stack(M)
            Srep = S_vec(e_rep, M).tolist()
            ch = int(np.nanargmax(Ssel))
            s_ch = M[:, ch]
            r["positions"][pname] = {
                "S_report": Srep, "S_selection_oof": Ssel, "accuracy_report": acc, "C": Cs,
                "chosen_layer": layers[ch], "S_report_at_chosen": Srep[ch],
                "S_minus_answer_at_chosen": Srep[ch] - r["answer_S_report"],
                "S_cells_at_chosen": float(e_rep.S_cells(s_ch)),
                "spearman_with_answer_within_string_at_chosen": within_string_spearman(s_ch, ans[rep], qr)}
            scores_rep[(run, pname)] = M
            log(f"{run} {pname}: chosen layer {layers[ch]}, S report {Srep[ch]:.3f}, answer {r['answer_S_report']:.3f}, "
                f"max S {max(Srep):.3f} at layer {layers[int(np.argmax(Srep))]}")
        scores_rep[(run, "answer")] = ans[rep]
        per_run[run] = r
        del R

    # ---------------------------------------------------------------- per condition, with premise-bootstrap intervals
    # each resample: S of every column (all runs x layers x positions, answers, references) on the resampled items
    keys, cols = [], []
    for k, M in scores_rep.items():
        M = M if M.ndim == 2 else M[:, None]
        for j in range(M.shape[1]):
            keys.append((k, j)); cols.append(M[:, j])
    ref_keys = list(ref_scores)
    big = np.column_stack(cols + [ref_scores[k] for k in ref_keys])
    pos_of = {kj: i for i, kj in enumerate(keys)}

    def boot_matrix(nb, seed):
        Sb = np.empty((nb, big.shape[1]))
        for b, (idx, e) in enumerate(boot_sets(yr, qr, nr, gr, ar, nb, seed)):
            Sb[b] = e.S(big[idx])
        return Sb
    log(f"bootstrap: {N_BOOT_CURVE} resamples for the curves, {stat.N_BOOT} for the summaries, {big.shape[1]} columns")
    Sb_curve = boot_matrix(N_BOOT_CURVE, 1)
    Sb_sum = boot_matrix(stat.N_BOOT, 2)
    point = S_vec(e_rep, big)

    def ci(per_seed, boot_mean):
        """Section 3's interval: D +- t(df) x sqrt(seed variance / n + premise variance), the premise variance taken from
        the resamples of the seed mean (models fixed); one model: D +- 1.96 x premise SE. Not a percentile interval:
        resampling premises duplicates items and biases S itself downward, while its spread is what is needed."""
        x = np.atleast_1d(np.asarray(per_seed, float))
        D, v = float(x.mean()), float(np.nanvar(np.asarray(boot_mean, float), ddof=1))
        if len(x) >= 2:
            t, df = seedtest.t_and_df(D, x.var(ddof=1), len(x), None, 0, v)
            half = float(sps.t.ppf(0.975, float(df))) * (abs(D / float(t)) if float(t) != 0 else np.sqrt(x.var(ddof=1) / len(x) + v))
        else:
            half = 1.96 * np.sqrt(v)
        return [D - half, D + half]

    summary = {}
    for cond, runs in COND.items():
        out = {"runs": runs, "positions": {}}
        for pname in POSITIONS:
            L = len(per_run[runs[0]]["layers"])
            idx = np.array([[pos_of[((r_, pname), j)] for j in range(L)] for r_ in runs])      # runs x layers
            mean_pt = point[idx].mean(0)
            mean_b = Sb_curve[:, idx].mean(1)                                                 # resamples x layers
            lo, hi = np.array([ci(point[idx][:, j], mean_b[:, j]) for j in range(L)]).T
            above = [per_run[runs[0]]["layers"][j] for j in range(L) if lo[j] > 0.5]
            mx = float(np.max(mean_pt))
            reach90 = next(per_run[runs[0]]["layers"][j] for j in range(L) if mean_pt[j] >= 0.9 * mx)
            reach90_excess = next(per_run[runs[0]]["layers"][j] for j in range(L) if mean_pt[j] - 0.5 >= 0.9 * (mx - 0.5))
            # (c) at each run's own chosen layer, minus that run's own answer
            ch = [per_run[r_]["layers"].index(per_run[r_]["positions"][pname]["chosen_layer"]) for r_ in runs]
            pi = np.array([pos_of[((r_, pname), c)] for r_, c in zip(runs, ch)])
            ai = np.array([pos_of[((r_, "answer"), 0)] for r_ in runs])
            D_pt = float((point[pi] - point[ai]).mean())
            D_b = (Sb_sum[:, pi] - Sb_sum[:, ai]).mean(1)
            P_pt = float(point[pi].mean())
            out["positions"][pname] = {
                "layers": per_run[runs[0]]["layers"], "S_seed_mean": mean_pt.tolist(), "S_ci": [lo.tolist(), hi.tolist()],
                "S_seed_min": point[idx].min(0).tolist(), "S_seed_max": point[idx].max(0).tolist(),
                "first_layer_ci_above_0.5": above[0] if above else None,
                "first_layer_reaching_90pct_of_max_S": reach90, "first_layer_reaching_90pct_of_max_excess": reach90_excess,
                "max_S": mx, "chosen_layers": [per_run[r_]["positions"][pname]["chosen_layer"] for r_ in runs],
                "S_probe_at_chosen": P_pt, "S_probe_at_chosen_ci": ci(point[pi], Sb_sum[:, pi].mean(1)),
                "S_probe_minus_answer": D_pt, "S_probe_minus_answer_ci": ci(point[pi] - point[ai], D_b),
                "S_cells_at_chosen_seed_mean": float(np.mean([per_run[r_]["positions"][pname]["S_cells_at_chosen"] for r_ in runs])),
                "spearman_with_answer_seed_mean": float(np.mean([per_run[r_]["positions"][pname]["spearman_with_answer_within_string_at_chosen"] for r_ in runs]))}
        out["answer_S_seed_mean"] = float(np.mean([per_run[r_]["answer_S_report"] for r_ in runs]))
        ai_ = [pos_of[((r_, "answer"), 0)] for r_ in runs]
        out["answer_S_ci"] = ci(point[ai_], Sb_sum[:, ai_].mean(1))
        summary[cond] = out
    nk = len(keys)
    refs_out = {k: {"S": ref_S[k], "ci": ci([point[nk + i]], Sb_sum[:, nk + i])} for i, k in enumerate(ref_keys)}
    for cond in ("A", "B"):                     # each condition's probe against its own references (descriptive)
        for pname in POSITIONS:
            runs = COND[cond]
            ch = [per_run[r_]["layers"].index(per_run[r_]["positions"][pname]["chosen_layer"]) for r_ in runs]
            pi = np.array([pos_of[((r_, pname), c)] for r_, c in zip(runs, ch)])
            for rk in (f"memorisation_E5b|{cond}", "selection_half_fit|tuned_stack"):
                ri = nk + ref_keys.index(rk)
                summary[cond]["positions"][pname][f"S_probe_minus|{rk}"] = {
                    "D": float(point[pi].mean() - point[ri]), "ci": ci(point[pi] - point[ri], Sb_sum[:, pi].mean(1) - Sb_sum[:, ri])}
    res = {"meta": {"mode": mode, "amendment": "PREREG Amendment 2", "descriptive": True, "n_report": int(rep.sum()),
                    "n_selection": int(sel.sum()), "n_boot_curve": N_BOOT_CURVE, "n_boot_summary": stat.N_BOOT,
                    "tuned_stack_C_selection_half": C_stack, "layer_stride": a.layer_stride,
                    "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "references_report_half": refs_out, "conditions": summary, "runs": per_run}
    OUT.mkdir(exist_ok=True)
    p = OUT / f"e6_{mode}.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    (OUT / f"e6_{mode}_tables.md").write_text(tables(res))
    log("wrote", p.relative_to(ROOT))


def tables(res) -> str:
    f = lambda v: f"{v:.3f}"
    L = [f"# E6 ({res['meta']['mode']}): probe ladder on POOL-LB, report half (generated by finetune/scripts/e6_analysis.py)", "",
         "Descriptive (PREREG Amendment 2). S = within-string AUROC with prompt length removed; 95% premise-bootstrap intervals.", "",
         "## Summary at the layer chosen on the selection half", "",
         "| Condition | Position | chosen layers | S probe | S answer | probe − answer | first layer CI > 0.5 | 90% of max S | S cells | ρ(probe, answer) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c, o in res["conditions"].items():
        for p, d in o["positions"].items():
            L.append(f"| {c} | {p} | {d['chosen_layers']} | {f(d['S_probe_at_chosen'])} [{f(d['S_probe_at_chosen_ci'][0])}, {f(d['S_probe_at_chosen_ci'][1])}] "
                     f"| {f(o['answer_S_seed_mean'])} | {d['S_probe_minus_answer']:+.3f} [{d['S_probe_minus_answer_ci'][0]:+.3f}, {d['S_probe_minus_answer_ci'][1]:+.3f}] "
                     f"| {d['first_layer_ci_above_0.5']} | {d['first_layer_reaching_90pct_of_max_S']} | {f(d['S_cells_at_chosen_seed_mean'])} | {d['spearman_with_answer_seed_mean']:+.2f} |")
    L += ["", "## Probe against the condition's own references (descriptive)", "", "| Condition | Position | reference | D | 95% CI |", "|---|---|---|---|---|"]
    for c in ("A", "B"):
        for p, d in res["conditions"][c]["positions"].items():
            for k, v in d.items():
                if k.startswith("S_probe_minus|"):
                    L.append(f"| {c} | {p} | {k.split('|', 1)[1]} | {v['D']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |")
    L += ["", "## Reference rows (report half)", "", "| Reference | S | 95% CI |", "|---|---|---|"]
    for k, v in res["references_report_half"].items():
        L.append(f"| {k} | {f(v['S'])} | [{f(v['ci'][0])}, {f(v['ci'][1])}] |")
    L += ["", "## S by layer (seed mean [CI])", ""]
    for c, o in res["conditions"].items():
        for p, d in o["positions"].items():
            L.append(f"**{c}, {p}:** " + " · ".join(f"{l}: {f(m)} [{f(lo)}, {f(hi)}]" for l, m, lo, hi in
                                                     zip(d["layers"], d["S_seed_mean"], d["S_ci"][0], d["S_ci"][1])))
            L.append("")
    L += ["## Integrity", "", "| Run | largest |answer − stored score| |", "|---|---|"]
    for r_, v in res["runs"].items():
        L.append(f"| {r_} | {v['max_abs_diff_vs_stored_scores']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

"""The descriptive table registered beside E4 (PREREG Section 4, E4, last sentence; Section 1 set table: "Table 5b on
held-out premises; descriptive"): pooled F1 on ID, ID renamed, ID paraphrased and the authors' two perturbed sets,
every condition. Computed 23 Nov 2026 04:42 UTC (22 Nov 23:42 EST), AFTER the behaviour audit (owner's decision; listed as a deviation in the draft).
No GPU, no model: stored scores only.

  ../causal-probe/.venv/bin/python finetune/scripts/table5b.py
    -> finetune/results/table5b.json and table5b.md

Threshold: answer logit difference > 0, the registered threshold of E1 (the F1-maximising training-set threshold
was not computed: training-set scores were not stored; draft, Deviations). Valid-class F1 is the headline column
(the authors' Table 5b reports F1 for the valid class); macro-F1 and accuracy beside it. Per condition: the
registered seeds (A 5, B 5, A48 3, C 3, Bswap 3), seed mean and range, and a 95% interval in Section 3's form
(seed variance + premise variance from 2,000 premise resamples of the set, models fixed); base: premise interval.
The official test split (21 premises) is added for continuity with the paper.
Audit conditions (finetune/notes/audit_table5b_raw.md) met here:
  - trivial baselines per set: all-valid F1, all-invalid accuracy, macro-F1 of all-invalid;
  - intervals clipped to [0, 1], clipped cells marked "*";
  - a t(df = seeds - 1) multiplier (t(2) for 3-seed conditions), and intervals on 21 premises are rough;
  - outlying seeds flagged by PREREG's rule (more than 0.10 from the condition's median), plus seeds that predict a
    single class on a whole set;
  - rows restricted to the 1,162 authors' items that are our renaming / paraphrase of the official test
    ("matched"), beside the full files, which add 722 / 360 extra items on the same premises;
  - pooled AUROC beside F1, so that a threshold shift is not read as lost skill.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]
import numpy as np                                     # noqa: E402
from scipy import stats as sps                         # noqa: E402

from ftstudy import behaviour as BH, datasets as DS, seedtest, stat   # noqa: E402

SETS = ["id", "id_renamed", "id_paraphrased", "authors_refactorization", "authors_refactorization__matched_1162",
        "authors_paraphrasing", "authors_paraphrasing__matched_1162", "official_test"]
CONDS = ["base", "A", "B", "A48", "C", "Bswap"]
RUNS = ROOT / "finetune" / "runs"


def metrics(y, s):
    pred = (s > 0).astype(int)
    f1v = BH.f1_valid(y, pred)
    f1i = BH.f1_valid(1 - y, 1 - pred)
    return np.array([f1v, (f1v + f1i) / 2, float((pred == y).mean()), BH.pooled_auroc(y, s)])


METRICS = ("f1_valid", "macro_f1", "accuracy", "pooled_auroc")


def matched_mask(k, items):
    """Authors' items that are exactly our renaming / paraphrase of an official-test item (text and label)."""
    from causal_probe import data as D1
    t = DS.materialise("eval_official_test")
    f = D1.rename_variables if k.startswith("authors_refactorization") else DS.paraphrase
    keys = {(x["context"], x["query"], x["label"]) for x in map(f, t)}
    return np.array([(i["context"], i["query"], i["label"]) in keys for i in items])


def interval(per_seed, boot_mean):
    x = np.atleast_1d(np.asarray(per_seed, float))
    D, v = float(x.mean()), float(np.var(boot_mean, ddof=1))
    if len(x) >= 2:
        t, df = seedtest.t_and_df(D, x.var(ddof=1), len(x), None, 0, v)
        se = np.sqrt(x.var(ddof=1) / len(x) + v)
        half = float(sps.t.ppf(0.975, float(df))) * se
    else:
        half = 1.96 * np.sqrt(v)
    lo, hi = D - half, D + half
    return [max(0.0, lo), min(1.0, hi), bool(lo < 0 or hi > 1)]


def main():
    runs = {c: (["base"] if c == "base" else [f"{c}_s{s}" for s in range(BH.REGISTERED_SEEDS[c])]) for c in CONDS}
    out = {"threshold": "answer logit difference > 0", "computed": "23 Nov 2026 04:42 UTC (22 Nov 23:42 EST), after the behaviour audit; audit conditions met 23 Nov",
           "n_boot": stat.N_BOOT, "sets": {}}
    for k in SETS:
        base_k = k.split("__")[0]
        items = DS.materialise("eval_" + base_k)
        sub = matched_mask(k, items) if k.endswith("matched_1162") else np.ones(len(items), bool)
        idx = np.where(sub)[0]
        items = [items[i] for i in idx]
        y = np.array([i["label"] for i in items]); g = np.array([i["group"] for i in items])
        boots = stat.boot_indices(g, stat.N_BOOT, seed=0)
        rows = {}
        for c, rs in runs.items():
            per, bt = [], []
            for r in rs:
                z = np.load(RUNS / r / "scores.npz")
                assert [str(u) for u in z[f"eval_{base_k}__uid"][idx]] == [i["uid"] for i in items], (r, k)
                s = z[f"eval_{base_k}__score"][idx].astype(float)
                per.append(metrics(y, s))
                bt.append(np.array([metrics(y[b], s[b]) for b in boots]))
            per, bt = np.array(per), np.array(bt)                 # seeds x 4 ; seeds x boots x 4
            rows[c] = {m: {"mean": float(per[:, j].mean()), "min": float(per[:, j].min()), "max": float(per[:, j].max()),
                           "ci": interval(per[:, j], bt[:, :, j].mean(0)), "per_seed": per[:, j].tolist(),
                           "outlying_seeds": [rs[i] for i in seedtest.outlying_seeds(per[:, j])] if len(rs) > 2 else []}
                       for j, m in enumerate(METRICS)}
            single = []
            for r in rs:
                s = np.load(RUNS / r / "scores.npz")[f"eval_{base_k}__score"][idx]
                if (s > 0).all() or (s <= 0).all():
                    single.append(r)
            rows[c]["single_class_seeds"] = single
        pv = float(y.mean())
        trivial = {"all_valid_f1": 2 * pv / (1 + pv), "all_invalid_accuracy": 1 - pv, "all_invalid_macro_f1": (1 - pv) / (2 - pv)}
        out["sets"][k] = {"n": len(items), "premises": int(len(np.unique(g))), "share_valid": pv, "trivial": trivial,
                          "conditions": rows}
        print(k, {c: round(v["f1_valid"]["mean"], 3) for c, v in rows.items()}, flush=True)
    (ROOT / "finetune" / "results" / "table5b.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# Table 5b analogue: pooled F1 on held-out premises and the authors' perturbed sets (descriptive)", "",
         "Generated by `finetune/scripts/table5b.py` from `finetune/runs/*/scores.npz`. Computed 23 Nov 2026 04:42 UTC, after the",
         "behaviour audit; audited (`finetune/notes/audit_table5b_raw.md`). Threshold: answer logit difference > 0.",
         "Cells: seed mean [95% interval, Section 3 form, t(seeds - 1); clipped to [0, 1], * = clipped] (seed range).",
         "Intervals on the 21-premise sets (authors', official test) are rough. Outlying seeds (PREREG rule) are marked †",
         "and listed below the table. Bswap: no interval is to be interpreted (one seed predicts \"valid\" on every item).",
         "The authors' files hold the 1,162 perturbed test items plus 722 (refactorisation) / 360 (paraphrasing) extra items",
         "on the same 21 premises; the \"__matched_1162\" rows keep only the 1,162. Compare with the authors' Table 5b by",
         "direction only; no robust/fragile or between-condition claim is made from this table.", ""]
    heads = {"f1_valid": "Valid-class F1", "macro_f1": "Macro-F1", "accuracy": "Accuracy", "pooled_auroc": "Pooled AUROC"}
    for metric in METRICS:
        triv = {"f1_valid": "all-valid F1", "macro_f1": "all-invalid macro-F1", "accuracy": "all-invalid accuracy", "pooled_auroc": "chance"}[metric]
        L += [f"## {heads[metric]}", "", f"| Set | items (valid share) | trivial ({triv}) | " + " | ".join(CONDS) + " |",
              "|---|---|---|" + "---|" * len(CONDS)]
        for k, v in out["sets"].items():
            tv = {"f1_valid": v["trivial"]["all_valid_f1"], "macro_f1": v["trivial"]["all_invalid_macro_f1"],
                  "accuracy": v["trivial"]["all_invalid_accuracy"], "pooled_auroc": 0.5}[metric]
            cells = []
            for c in CONDS:
                d = v["conditions"][c][metric]
                rng = "" if c == "base" else f" ({d['min']:.3f}–{d['max']:.3f})"
                star = "*" if d["ci"][2] else ""
                dag = "†" if d["outlying_seeds"] else ""
                cells.append(f"{d['mean']:.3f}{dag} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]{star}{rng}")
            L.append(f"| {k} | {v['n']} ({v['share_valid']:.3f}) | {tv:.3f} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Outlying and single-class seeds", ""]
    for k, v in out["sets"].items():
        for c in CONDS:
            d = v["conditions"][c]
            ol = {m: d[m]["outlying_seeds"] for m in METRICS if d[m]["outlying_seeds"]}
            if ol or d["single_class_seeds"]:
                L.append(f"- {k}, {c}: outlying {ol or 'none'}; single-class predictions {d['single_class_seeds'] or 'none'}")
    (ROOT / "finetune" / "results" / "table5b.md").write_text("\n".join(L) + "\n")
    print("wrote finetune/results/table5b.json and table5b.md")


if __name__ == "__main__":
    main()

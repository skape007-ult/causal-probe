"""Label-changing vs label-preserving minimal pairs, compared within hypothesis template (CPU only).

The two kinds of pair have different mixes of hypothesis templates (round-2 code review, M8), so the
comparison of |D| (how much the answer logit differs between the two prompts of a pair) has to be made
within template. The label-changing runs made before 13 Nov did not save the pairs' templates; they
are recovered here by rebuilding the pair list exactly as scripts/08_patching.py does (same pool,
seed and tokenizer filter; no model needed) and checking it against the saved premise ids.

Writes results/patching_by_template_<model>.json.
"""
import argparse
import json

import numpy as np

from causal_probe import cache, data, models, patch, stats
from causal_probe.config import ALL_MODELS, RESULTS, SEED


def rebuild(tok, kind, n_pairs, label_mode, max_vars=5):
    pool = [i for i in data.load_c2c("train") if i["num_variables"] <= max_vars]
    pairs = patch.build_pairs(pool, kind, n_pairs * 2, seed=SEED, label_mode=label_mode)
    encs = [(p, patch.encode_pair(tok, p)) for p in pairs]
    return [p for p, e in encs if e is not None][:n_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    a = ap.parse_args()
    tok = models.load_tokenizer(a.model)
    out = {"model": a.model, "kinds": {}}
    for kind in ("hyp", "prem"):
        rows = {}
        for mode, suffix in (("change", ""), ("same", "_same")):
            f = cache.CACHE / a.model / f"patching{suffix}.npz"
            j = RESULTS / f"patching_{a.model}{suffix}.json"
            if not (f.exists() and j.exists()):
                continue
            z = np.load(f, allow_pickle=True)
            d = z[f"{kind}_ld_clean"] - z[f"{kind}_ld_cor"]
            grp = z[f"{kind}_groups"]
            if f"{kind}_template" in z.files and f"{kind}_swap" in z.files:
                tpl = np.array([str(t) for t in z[f"{kind}_template"]])
                later = np.array([str(w)[1] > str(w)[0] for w in z[f"{kind}_swap"]])
            else:
                pairs = rebuild(tok, kind, len(d), mode, json.loads(j.read_text())["max_vars"])
                assert [p["group"] for p in pairs] == [str(g) for g in grp] or [str(p["group"]) for p in pairs] == [str(g) for g in grp], \
                    "rebuilt pair list does not match the saved run"
                tpl = np.array([p["clean"]["template"] for p in pairs])
                later = np.array([p["swap"][1] > p["swap"][0] for p in pairs])      # the letter is replaced by a LATER one
            rows[mode] = (d, tpl, grp, later)
        if len(rows) < 2:
            continue
        (dc, tc, gc, lc), (ds, ts, gs, ls) = rows["change"], rows["same"]
        tc, ts = np.array([str(t) for t in tc]), np.array([str(t) for t in ts])
        common = sorted(set(tc.tolist()) & set(ts.tolist()))
        res = {"templates_in_both_kinds": common, "by_template": {}}
        for t in sorted(set(tc.tolist()) | set(ts.tolist())):
            res["by_template"][t] = {"label_changing": {"n_pairs": int((tc == t).sum()), "mean_abs_D": float(np.abs(dc[tc == t]).mean()) if (tc == t).any() else None},
                                     "label_preserving": {"n_pairs": int((ts == t).sum()), "mean_abs_D": float(np.abs(ds[ts == t]).mean()) if (ts == t).any() else None}}
        # template-standardised comparison: every template present in both kinds gets the same weight in both means
        w = {t: min((tc == t).sum(), (ts == t).sum()) for t in common}
        tot = sum(w.values())
        std_c = sum(w[t] * np.abs(dc[tc == t]).mean() for t in common) / tot
        std_s = sum(w[t] * np.abs(ds[ts == t]).mean() for t in common) / tot
        rng = np.random.default_rng(SEED)

        def boot(d, tpl, grp):
            ug = np.unique(grp); vals = []
            for _ in range(1000):
                pick = np.concatenate([np.where(grp == g)[0] for g in rng.choice(ug, len(ug))])
                dd, tt = d[pick], tpl[pick]
                if all((tt == t).any() for t in common):
                    vals.append(sum(w[t] * np.abs(dd[tt == t]).mean() for t in common) / tot)
            return np.array(vals)
        bc, bs = boot(dc, tc, gc), boot(ds, ts, gs)
        k = min(len(bc), len(bs))
        diff = bs[:k] - bc[:k]                         # the two kinds of pair are independent samples
        res["standardised_over_common_templates"] = {
            "weights": {t: int(v) for t, v in w.items()},
            "mean_abs_D_label_changing": {"est": float(std_c), "lo": float(np.quantile(bc, 0.025)), "hi": float(np.quantile(bc, 0.975))},
            "mean_abs_D_label_preserving": {"est": float(std_s), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))},
            "preserving_minus_changing": {"est": float(std_s - std_c), "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))},
            "ratio_preserving_over_changing": float(std_s / std_c)}
        res["unstandardised"] = {"mean_abs_D_label_changing": float(np.abs(dc).mean()), "mean_abs_D_label_preserving": float(np.abs(ds).mean()),
                                 "mean_D_label_changing": stats.bootstrap_ci(lambda v: float(v.mean()), (dc,), gc, n_boot=1000),
                                 "mean_D_label_preserving": stats.bootstrap_ci(lambda v: float(v.mean()), (ds,), gs, n_boot=1000)}
        # direction of the letter change (fifth audit, F6): label-preserving pairs are built with later-letter swaps only
        def sd(d, m, grp):
            return ({**stats.bootstrap_ci(lambda v: float(v.mean()), (d[m],), grp[m], n_boot=1000), "n_pairs": int(m.sum()),
                     "share_positive": float((d[m] > 0).mean())} if m.any() else {"n_pairs": 0})
        res["signed_D_by_direction_of_letter_change"] = {
            "label_changing_later_letter": sd(dc, lc, gc), "label_changing_earlier_letter": sd(dc, ~lc, gc),
            "label_preserving_later_letter": sd(ds, ls, gs), "label_preserving_earlier_letter": sd(ds, ~ls, gs),
            "note": "D = answer logit of the original prompt minus that of the edited one. The label-preserving pair list keeps only "
                    "edits in which the hypothesis letter is replaced by a later letter (patch.build_pairs counts each unordered pair once), "
                    "so its mean D must be compared with the later-letter half of the label-changing pairs."}
        if lc.any() and ls.any():
            a_, b_ = dc[lc], ds[ls]
            rng2 = np.random.default_rng(SEED + 1)
            boots = [a_[rng2.integers(0, len(a_), len(a_))].mean() - b_[rng2.integers(0, len(b_), len(b_))].mean() for _ in range(2000)]
            res["signed_D_by_direction_of_letter_change"]["label_changing_minus_label_preserving_later_letter"] = {
                "est": float(a_.mean() - b_.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
                "note": "pairs resampled individually; the two sets share some premises"}
        out["kinds"][kind] = res
        s = res["standardised_over_common_templates"]
        print(f"{a.model} {kind}: within-template |D| label-changing {s['mean_abs_D_label_changing']['est']:.3f}, label-preserving "
              f"{s['mean_abs_D_label_preserving']['est']:.3f}, difference {s['preserving_minus_changing']['est']:+.3f} "
              f"[{s['preserving_minus_changing']['lo']:+.3f}, {s['preserving_minus_changing']['hi']:+.3f}] over {common}")
    (RESULTS / f"patching_by_template_{a.model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

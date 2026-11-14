"""Is the steering slope of a class-mean-difference direction explained by its alignment with the answer direction? CPU only.

Added after the eleventh audit (notes/audit_runs_b_d_raw.md, F4), which found that the gold difference of class means ranks
above its 20 shuffled-label controls on average, and why. For every model with a 20-control steering run on the debiased
pool (cache/<model>/c2c_debiased/steering_controls[_hyp].npz) and the activations it was fitted on, at both tokens:

  1. the 20 shuffled-label class-mean differences of that run are regenerated from the run's seed (checked against the
     natural SDs the run saved; a site where they do not match is reported and not used);
  2. each direction's cosine with the ANSWER direction is computed (the difference of class means between items whose
     answer logit is above and below the median, on the same fitting half);
  3. over the 20 controls, slope = b x cosine is fitted through the origin, with the correlation r;
  4. the gold direction's slope is predicted from its cosine alone and compared with the slope the run measured.

If r is high and the prediction is close, a class-mean difference moves the answer to the extent that it is aligned with the
answer direction, and the gold one outranks its controls because the answers are correlated with the label inside a
hypothesis string (the shuffle within string removes that correlation from the controls). That is a fact about behaviour;
it does not show that a direction encoding validity is used.

Writes results/steering_slope_vs_answer_alignment.json.
"""
import json

import numpy as np

from causal_probe import cache, data, stats
from causal_probe.config import MODELS, REMOTE_MODELS, RESULTS

TAG = "c2c_debiased"


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    items = data.get_dataset(TAG); sel = data.selection_mask(items)
    strs_sel = np.array([i["query"] for i in items])[sel]
    out = {"sites": [], "note": "controls: class-mean differences of labels shuffled within hypothesis string (the 20 of the steering-controls run, regenerated); "
                                "answer direction: class-mean difference between items with the answer logit above and below its median, same fitting half"}
    for m in {**MODELS, **REMOTE_MODELS}:
        d = cache.cache_dir(m, TAG)
        if not (d / "resid.npy").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        if meta["uids"] != [i["uid"] for i in items]:
            continue
        y = np.array(meta["labels"]); ld = np.array(meta["logit_diff"]); mc = (ld > np.median(ld)).astype(int)
        resid = np.load(d / "resid.npy", mmap_mode="r")
        for suf, pn, token in (("", "final", "final"), ("_hyp", "content_end", "end of hypothesis")):
            f = d / f"steering_controls{suf}.npz"
            if not f.exists() or pn not in meta["positions"]:
                continue
            z = np.load(f, allow_pickle=True); fam = np.array([str(x) for x in z["families"]]); k = len(fam)
            o = z["out"].astype(np.float64); sl = ((o[:, 1:1 + k] - o[:, 1 + k:]) / (2 * float(z["alpha"]))).mean(0)
            layer = int(z["layer"])
            X = np.asarray(resid[:, layer, meta["positions"].index(pn), :], dtype=np.float32)
            mu = X[sel].mean(0); s_unit = float(np.sqrt(((X[sel] - mu) ** 2).sum(1).mean()))
            rng = np.random.default_rng(101)                      # the generator of 07c_steering_controls.py: 100 isotropic draws, then the shuffles
            for _ in range(100):
                rng.standard_normal(X.shape[1])

            def shuf(lab):
                o_ = lab.copy()
                for s_ in np.unique(strs_sel):
                    ix = np.where(strs_sel == s_)[0]; o_[ix] = lab[rng.permutation(ix)]
                return o_
            dm = lambda ys: unit(X[sel][ys == 1].mean(0) - X[sel][ys == 0].mean(0))       # noqa: E731
            D = np.stack([dm(y[sel])] + [dm(shuf(y[sel])) for _ in range(20)])
            own = ((X[sel] - mu) @ D.T).std(0) / s_unit
            saved = np.concatenate([z["own_sd_over_s"][fam == "gold_diffmeans"], z["own_sd_over_s"][fam == "shuffled_diffmeans"]])
            rel = float(np.abs(own - saved).max() / saved.max())
            s = np.concatenate([sl[fam == "gold_diffmeans"], sl[fam == "shuffled_diffmeans"]])
            cs = D @ dm(mc[sel])
            b = float((cs[1:] @ s[1:]) / (cs[1:] @ cs[1:])); r = float(np.corrcoef(cs[1:], s[1:])[0, 1])
            pc = float(sl[fam == "model_out"][0]); pcc = sl[fam == "shuffled_probe"]
            rec = {"model": m, "token": token, "layer": layer, "n_layers": int(resid.shape[1] - 1),
                   "regenerated_controls_match_the_run": rel < 1e-3, "max_relative_difference_of_natural_sds": rel,
                   "s_unit_matches": abs(s_unit - float(z["s_unit"])) < 1e-2 * max(1.0, s_unit),
                   "answers_vs_gold_within_string_auroc_fitting_half": float(np.mean([
                       stats.auroc(y[sel][strs_sel == u], ld[sel][strs_sel == u]) for u in np.unique(strs_sel)])),
                   "over_the_20_controls": {"r_cosine_vs_slope": r, "slope_per_unit_cosine": b},
                   "gold_difference_of_means": {"cosine_with_answer_direction": float(cs[0]), "rank_of_cosine_among_21_most_positive_first": int(1 + (cs[1:] > cs[0]).sum()),
                                                "slope_predicted_from_cosine": b * float(cs[0]), "slope_observed": float(s[0])},
                   "controls_cosine_sd": float(cs[1:].std(ddof=1)),
                   "positive_control_exceeds_all_controls": bool((np.abs(pcc) < abs(pc)).all())}
            out["sites"].append(rec)
            g = rec["gold_difference_of_means"]
            print(f"{m:12s} {token:17s} L{layer:2d} match {rec['regenerated_controls_match_the_run']} (rel {rel:.1e})  r {r:+.2f}  cos gold {g['cosine_with_answer_direction']:+.3f} "
                  f"(rank {g['rank_of_cosine_among_21_most_positive_first']})  predicted {g['slope_predicted_from_cosine']:+.2f}  observed {g['slope_observed']:+.2f}")
    out["by_site"] = {f"{r_['model']}.{'hyp' if r_['token'] != 'final' else 'final'}": r_ for r_ in out["sites"]}
    (RESULTS / "steering_slope_vs_answer_alignment.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

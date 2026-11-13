"""Layer-wise linear probes with controls, from the cache (CPU).

Writes  results/probes_<model>_<tag>.json   (summary per layer, traceable numbers)
        cache/<model>/<tag>/probe_oof.npz   (out-of-fold scores, reused by 05_confound.py)

Targets (final position, all layers, PROBE_SEEDS x N_FOLDS grouped folds):
  gold        the benchmark label
  model_out   the model's own answer (sign of the Yes-No answer logit difference)
Controls:
  shuffled    trained on permuted labels, scored against true labels
  control     Hewitt & Liang control task (random label per hypothesis/question type)
  random_init same probe on a randomly initialised copy of the model
  tfidf       bag-of-words on the input text; type_prior = label prior per template/query type
Other positions (content_end, premise_end): gold only.
"""
import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed

from causal_probe import cache, probes, stats
from causal_probe.config import MODELS, POSITIONS, PROBE_SEEDS, RESULTS


def run_layer(resid, layer, pos, jobs):
    """jobs: list of (name, y_eval, y_train, folds). Returns {name: oof scores}."""
    X = probes.layer_features(resid, layer, pos)
    return {name: probes.oof_scores(X, y, folds, y_train=yt) for name, y, yt, folds in jobs}


def summarise(y, S, groups, ci=True):
    """S [seeds, n] decision scores -> seed-averaged accuracy (with cluster-bootstrap CI) and AUROC."""
    def mean_acc(yy, *ss):
        return float(np.mean([stats.accuracy(yy, (s > 0).astype(int)) for s in ss]))
    out = {"auroc": float(np.mean([stats.auroc(y, s) for s in S])),
           "acc_per_seed": [stats.accuracy(y, (s > 0).astype(int)) for s in S]}
    out["accuracy"] = (stats.bootstrap_ci(mean_acc, (y, *S), groups, n_boot=500) if ci
                       else {"est": mean_acc(y, *S)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tag", required=True, choices=["c2c_probe", "cladder"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    resid, meta, items = cache.load_cache(a.model, a.tag)
    resid = np.asarray(resid)                       # into RAM once (float16)
    y = np.array(meta["labels"])
    m = (np.array(meta["logit_diff"]) > 0).astype(int)
    groups = np.array([i["group"] for i in items])
    L = resid.shape[1]
    minority = min(m.mean(), 1 - m.mean())
    do_model_out = minority >= 0.05
    # Median-split of the model's answer logit difference: its *relative* preference once a
    # constant Yes/No bias is removed (added after seeing that Qwen-0.5B answers "No" to
    # ~everything, which makes the raw model-output target degenerate; logged in notes/log.md).
    ldv = np.array(meta["logit_diff"])
    mcal = (ldv > np.median(ldv)).astype(int)
    ctrl = probes.control_task_labels(items)
    folds = {s: probes.folds_for(items, s) for s in PROBE_SEEDS}

    oof = {}
    # ---- final position: all targets
    jobs = []
    for s in PROBE_SEEDS:
        jobs.append((f"gold/{s}", y, None, folds[s]))
        jobs.append((f"shuffled/{s}", y, probes.shuffled_labels(y, s), folds[s]))
        jobs.append((f"control/{s}", ctrl, None, folds[s]))
        jobs.append((f"model_out_cal/{s}", mcal, None, folds[s]))
        if do_model_out:
            jobs.append((f"model_out/{s}", m, None, folds[s]))
    per_layer = Parallel(n_jobs=a.jobs)(delayed(run_layer)(resid, l, 0, jobs) for l in range(L))
    for name in ["gold", "shuffled", "control", "model_out_cal"] + (["model_out"] if do_model_out else []):
        oof[name] = np.stack([[per_layer[l][f"{name}/{s}"] for l in range(L)] for s in PROBE_SEEDS])
    print(f"final position done {time.time() - t0:.0f}s", flush=True)

    # ---- other positions: gold only
    gjobs = [(f"gold/{s}", y, None, folds[s]) for s in PROBE_SEEDS]
    for pi, pname in enumerate(POSITIONS[1:], start=1):
        pl = Parallel(n_jobs=a.jobs)(delayed(run_layer)(resid, l, pi, gjobs) for l in range(L))
        oof[f"gold@{pname}"] = np.stack([[pl[l][f"gold/{s}"] for l in range(L)] for s in PROBE_SEEDS])
    del resid

    # ---- random-init model
    has_rand = (cache.cache_dir(a.model, a.tag, True) / "meta.json").exists()
    if has_rand:
        rres, rmeta, _ = cache.load_cache(a.model, a.tag, random_init=True)
        assert rmeta["uids"] == meta["uids"]
        rres = np.asarray(rres)
        pl = Parallel(n_jobs=a.jobs)(delayed(run_layer)(rres, l, 0, gjobs) for l in range(L))
        oof["random_init"] = np.stack([[pl[l][f"gold/{s}"] for l in range(L)] for s in PROBE_SEEDS])
        del rres

    # ---- text baselines (no layers)
    oof["tfidf"] = np.stack([probes.tfidf_oof(items, y, folds[s]) for s in PROBE_SEEDS])
    oof["type_prior"] = np.stack([probes.query_only_oof(items, y, folds[s]) for s in PROBE_SEEDS])
    oof["tfidf_model_out_cal"] = np.stack([probes.tfidf_oof(items, mcal, folds[s]) for s in PROBE_SEEDS])
    if do_model_out:
        oof["tfidf_model_out"] = np.stack([probes.tfidf_oof(items, m, folds[s]) for s in PROBE_SEEDS])

    np.savez_compressed(cache.cache_dir(a.model, a.tag) / "probe_oof.npz", y=y, model_pred=m, model_pred_cal=mcal,
                        control_labels=ctrl, **{k.replace("@", "_at_"): v for k, v in oof.items()})

    # ---- summary
    res = {"model": MODELS[a.model], "tag": a.tag, "n": len(y), "n_groups": len(set(groups)),
           "n_layers_plus_embed": L, "probe_C": probes.PROBE_C, "seeds": list(PROBE_SEEDS),
           "model_pred_yes_frac": float(m.mean()), "model_out_probe_run": bool(do_model_out),
           "model_accuracy": stats.accuracy(y, m), "control_label_balance": float(ctrl.mean()),
           "layers": {}, "baselines": {}}
    targets = {"gold": y, "shuffled": y, "control": ctrl, "model_out": m, "model_out_cal": mcal, "random_init": y,
               "gold@content_end": y, "gold@premise_end": y}
    for name, S in oof.items():
        if S.ndim == 3:
            res["layers"][name] = [summarise(targets[name], S[:, l], groups) for l in range(L)]
        else:
            res["baselines"][name] = summarise({"tfidf_model_out": m, "tfidf_model_out_cal": mcal}.get(name, y), S, groups)
    g = np.array([r["accuracy"]["est"] for r in res["layers"]["gold"]])
    c = np.array([r["accuracy"]["est"] for r in res["layers"]["control"]])
    res["selectivity_by_layer"] = (g - c).tolist()
    res["best_layer_gold_final"] = int(g.argmax())
    res["note_best_layer"] = ("argmax over layers on the same held-out predictions is optimistically "
                              "biased; headline numbers use the split-half selection in 05_confound.py")
    (RESULTS / f"probes_{a.model}_{a.tag}.json").write_text(json.dumps(res, indent=1))
    bl = res["best_layer_gold_final"]
    print(f"{a.model}/{a.tag}: best layer {bl} gold acc {g[bl]:.3f} control {c[bl]:.3f} "
          f"shuffled {res['layers']['shuffled'][bl]['accuracy']['est']:.3f} "
          f"tfidf {res['baselines']['tfidf']['accuracy']['est']:.3f} "
          f"type_prior {res['baselines']['type_prior']['accuracy']['est']:.3f} "
          + (f"rand {res['layers']['random_init'][bl]['accuracy']['est']:.3f} " if has_rand else "")
          + f"model acc {res['model_accuracy']:.3f}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()

"""Behavioural evaluation from the cache (no model needed). Writes results/behaviour_<model>.json.

Zero-shot and few-shot; answer-token-logit scoring and generated-text scoring; baselines;
breakdowns; variable-renaming robustness. CIs are cluster bootstraps over premises / stories.
"""
import argparse
import json

import numpy as np

from causal_probe import cache, data, stats
from causal_probe.config import MODELS, RESULTS


def metric_block(y, pred, score, groups):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {
        "accuracy": stats.bootstrap_ci(stats.accuracy, (y, pred), groups),
        "macro_f1": stats.bootstrap_ci(stats.macro_f1, (y, pred), groups),
        "pos_f1": stats.bootstrap_ci(stats.pos_f1, (y, pred), groups),
        "frac_pred_yes": float(np.mean(pred == 1)),
        "frac_gold_yes": float(np.mean(y == 1)),
    }
    if score is not None:
        out["auroc_logit_diff"] = stats.bootstrap_ci(stats.auroc, (y, np.asarray(score)), groups)
    return out


def by_key(items, y, pred, key, groups):
    out = {}
    for v in sorted({i[key] for i in items}, key=str):
        m = np.array([i[key] == v for i in items])
        out[str(v)] = {"n": int(m.sum()), "frac_gold_yes": float(y[m].mean()),
                       "accuracy": stats.bootstrap_ci(stats.accuracy, (y[m], pred[m]), groups[m], n_boot=1000),
                       "macro_f1": stats.macro_f1(y[m], pred[m])}
    return out


def eval_tag(model, tag):
    _, meta, items = cache.load_cache(model, tag)
    y = np.array(meta["labels"])
    ld = np.array(meta["logit_diff"])
    pred = (ld > 0).astype(int)
    groups = np.array([i["group"] for i in items])
    ans_ids = set(meta["answer_tokens"]["yes"].values()) | set(meta["answer_tokens"]["no"].values())
    res = {"n": len(y), "n_groups": len(set(groups)), "label_balance": data.label_balance(items),
           "baselines": stats.baselines(y), "logit_scored": metric_block(y, pred, ld, groups),
           # None for models whose full-vocab logits were not computed during caching (Gemma)
           "top1_not_an_answer_token": (float(np.mean([t not in ans_ids for t in meta["top1_id"]]))
                                        if meta["top1_id"] is not None else None),
           "mean_logit_diff": float(ld.mean())}
    keys = ("template", "num_variables") if items[0]["dataset"] == "c2c" else ("rung", "query_type", "graph_id")
    res["by"] = {k: by_key(items, y, pred, k, groups) for k in keys}
    return res, (items, y, ld, pred, groups)


def gen_block(model, tag, y_by_uid, pred_by_uid, group_by_uid):
    p = cache.cache_dir(model, tag) / "gen.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    parsed = [cache.parse_answer(t) for t in g["text"]]
    y = np.array([y_by_uid[u] for u in g["uids"]])
    lp = np.array([pred_by_uid[u] for u in g["uids"]])
    groups = np.array([group_by_uid[u] for u in g["uids"]])
    gp = np.array([-1 if p is None else p for p in parsed])
    ok = gp >= 0
    examples = sorted({t.strip()[:40] for t, p in zip(g["text"], parsed) if p is None})[:8]
    return {"n": len(y), "unparseable_frac": float(1 - ok.mean()), "unparseable_examples": examples,
            # unparseable generations count as wrong
            "gen_scored": {"accuracy": stats.bootstrap_ci(stats.accuracy, (y, gp), groups),
                           "macro_f1": stats.macro_f1(y[ok], gp[ok]) if ok.any() else float("nan")},
            "logit_scored_same_items": {"accuracy": stats.bootstrap_ci(stats.accuracy, (y, lp), groups),
                                        "macro_f1": stats.macro_f1(y, lp)},
            "logit_vs_gen_disagree_frac": float(np.mean(gp != lp)),
            "logit_vs_gen_disagree_frac_parseable": float(np.mean(gp[ok] != lp[ok])) if ok.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    a = ap.parse_args()
    res = {"model": MODELS[a.model], "zero_shot": {}, "few_shot": {}, "generation": {}}
    cached = {}
    for tag in ["c2c_test", "c2c_probe", "cladder", "c2c_test_renamed"]:
        res["zero_shot"][tag], cached[tag] = eval_tag(a.model, tag)

    for base in ["c2c_test", "cladder"]:
        items, y, ld, pred, groups = cached[base]
        yb = {i["uid"]: int(v) for i, v in zip(items, y)}
        pb = {i["uid"]: int(v) for i, v in zip(items, pred)}
        gb = {i["uid"]: g for i, g in zip(items, groups)}
        res["generation"][f"{base}_sub"] = gen_block(a.model, f"{base}_sub", yb, pb, gb)
        fs_tag = f"{base}_sub_fewshot"
        if (cache.cache_dir(a.model, fs_tag) / "meta.json").exists():
            fs, (fitems, fy, fld, fpred, fgroups) = eval_tag(a.model, fs_tag)
            zs_pred = np.array([pb[i["uid"]] for i in fitems])
            fs["zero_shot_same_items"] = metric_block(fy, zs_pred, np.array(
                [ld[[j["uid"] for j in items].index(i["uid"])] for i in fitems]), fgroups)
            fs["delta_accuracy_fewshot_minus_zeroshot"] = stats.bootstrap_ci(
                lambda yy, a_, b_: stats.accuracy(yy, a_) - stats.accuracy(yy, b_), (fy, fpred, zs_pred), fgroups)
            fs["delta_macro_f1_fewshot_minus_zeroshot"] = stats.bootstrap_ci(
                lambda yy, a_, b_: stats.macro_f1(yy, a_) - stats.macro_f1(yy, b_), (fy, fpred, zs_pred), fgroups)
            res["few_shot"][fs_tag] = fs
            res["generation"][fs_tag] = gen_block(
                a.model, fs_tag, yb, {i["uid"]: int(v) for i, v in zip(fitems, fpred)}, gb)

    # robustness: variable renaming (paired, same items)
    items, y, ld, pred, groups = cached["c2c_test"]
    _, y2, ld2, pred2, _ = cached["c2c_test_renamed"]
    assert (y == y2).all()
    res["robustness_variable_renaming"] = {
        "original": res["zero_shot"]["c2c_test"]["logit_scored"],
        "renamed": res["zero_shot"]["c2c_test_renamed"]["logit_scored"],
        "prediction_flip_frac": stats.bootstrap_ci(lambda a_, b_: float(np.mean(a_ != b_)), (pred, pred2), groups),
        "delta_accuracy_renamed_minus_original": stats.bootstrap_ci(
            lambda yy, a_, b_: stats.accuracy(yy, b_) - stats.accuracy(yy, a_), (y, pred, pred2), groups),
        "delta_macro_f1_renamed_minus_original": stats.bootstrap_ci(
            lambda yy, a_, b_: stats.macro_f1(yy, b_) - stats.macro_f1(yy, a_), (y, pred, pred2), groups),
        "logit_diff_pearson_r": float(np.corrcoef(ld, ld2)[0, 1]),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"behaviour_{a.model}.json").write_text(json.dumps(res, indent=1))
    z = res["zero_shot"]
    for t in z:
        m = z[t]["logit_scored"]
        print(f"{a.model} {t}: acc {m['accuracy']['est']:.3f} [{m['accuracy']['lo']:.3f},{m['accuracy']['hi']:.3f}] "
              f"macroF1 {m['macro_f1']['est']:.3f} pred_yes {m['frac_pred_yes']:.2f} "
              f"auroc {m['auroc_logit_diff']['est']:.3f} maj_acc {z[t]['baselines']['majority']['accuracy']:.3f}")


if __name__ == "__main__":
    main()

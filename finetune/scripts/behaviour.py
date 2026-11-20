"""Behaviour analysis E1 to E5 (PREREG.md version 2, Amendment 1). Three modes:

  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/behaviour.py --synthetic planted
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/behaviour.py --synthetic null
      the real sets, references and resamples; every model's scores synthetic (no model file is opened).
      planted: B tracks validity much better than A (and above every reference); null: all conditions identical.
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/behaviour.py --base
      the base model only: S, S inside cells, pooled AUROC and F1 on every set, with premise-bootstrap intervals.
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/behaviour.py
      the registered analysis. REFUSES (exit 3) unless finetune/runs holds A_s0-4 and B_s0-4: no A or B score file
      is opened before that (owner's decision, 17 Nov; CLAUDE.md item 7).

Output: finetune/results/behaviour_<mode>.json. Uses the reference or the owner's part (b) through ftstudy.impl and
records which.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]
import numpy as np                                               # noqa: E402
from causal_probe import probes                                  # noqa: E402
from ftstudy import behaviour as BH, datasets as DS, impl, stat  # noqa: E402
from ftstudy.datasets import parse_premise                       # noqa: E402

RUNS, OUT = ROOT / "finetune" / "runs", ROOT / "finetune" / "results"
SETS = ["pool_lb", "id48", "id", "id48_renamed", "id48_renamed_keep_order"]
SUFFIX = {"id48_renamed": "-ren", "id48_renamed_keep_order": "-rko"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- sets, references
def build_sets(ntok_from: Path):
    """EvalSets aligned with the stored item order. Token counts are taken from a scores file (every model's are the
    same: same prompts, same tokenizer); renamed sets keep the original string identity and share ID48's resamples."""
    tok = np.load(ntok_from)
    items = {k: DS.materialise("eval_" + k) for k in SETS}
    sets = {}
    for k in SETS:
        uid = [str(u) for u in tok[f"eval_{k}__uid"]]
        if uid != [i["uid"] for i in items[k]]:                       # materialised items carry the -ren / -rko suffix
            raise RuntimeError(f"eval_{k}: token file not in the stored order")
        base = items["id48"] if k in SUFFIX else items[k]
        sets[k] = BH.EvalSet(k, np.array([i["label"] for i in items[k]]), np.array([i["query"] for i in base]),
                             tok[f"eval_{k}__n_tokens"].astype(float), np.array([i["group"] for i in items[k]]))
    sets["pool_lb"].adj = probes.structural_features(items["pool_lb"])[:, 2]
    for k in ("pool_lb", "id48", "id"):
        sets[k].boot = stat.boot_indices(sets[k].groups, stat.N_BOOT, seed=0)
    for k in SUFFIX:
        assert (sets[k].groups == sets["id48"].groups).all()
        sets[k].boot = sets["id48"].boot
    return sets, items


def memorisation_pair_only(train_items, test_items) -> dict:
    """Amendment 1, item 1: the reviewer's variant of the 'queried' members, counting a statement only when a queried
    variable is in its PAIR (not in its conditioning set). The 'all' members are unchanged."""
    import re
    st, lab = {}, {}
    for it in train_items:
        if it["group"] not in st:
            st[it["group"]] = parse_premise(it["context"])
        lab.setdefault(it["query"], {})[it["group"]] = it["label"]
    out = {f"queried_pair|{k}|{p}": np.full(len(test_items), 0.5) for k, ps in (("k", (1, 3, 5, 9)), ("exp", (1.0, 2.0, 4.0))) for p in ps}
    for i, it in enumerate(test_items):
        n, s = parse_premise(it["context"])
        qv = set(re.findall(r"\b[A-Z]\b", it["query"]))
        cands = sorted(g for g in lab.get(it["query"], {}) if st[g][0] == n)
        if not cands:
            continue
        y = np.array([lab[it["query"]][g] for g in cands], float)
        f = lambda x: {t for t in x if t[1] & qv}                              # noqa: E731
        mine = f(s)
        d = np.array([len(mine ^ f(st[g][1])) for g in cands], float)
        order = np.argsort(d, kind="stable")
        for k in (1, 3, 5, 9):
            out[f"queried_pair|k|{k}"][i] = y[order[:k]].mean()
        for tau in (1.0, 2.0, 4.0):
            w = np.exp(-(d - d.min()) / tau)
            out[f"queried_pair|exp|{tau}"][i] = float((w * y).sum() / w.sum())
    return out


def build_refs(sets, items, conds=("A", "B", "A48")):
    reg = json.loads((ROOT / "finetune" / "itemsets" / "references.json").read_text())["by_training_set"]
    tj = json.loads((ROOT / "finetune" / "itemsets" / "text_references.json").read_text())["conditions"]
    tz = np.load(ROOT / "finetune" / "itemsets" / "text_references.npz")
    e = sets["pool_lb"]
    refs = {"mem": {}, "mem_variant": {}, "text_best": {}, "mem_registered_S": {}, "near": {}, "checks": {}}
    for c in conds:
        tr = DS.materialise("train_" + c)
        refs["mem_registered_S"][c] = {k: reg[c][k]["memorisation_reference"]["S"] for k in ("pool_lb", "id48")}
        member = reg[c]["pool_lb"]["memorisation_reference"]["member"]
        fam = DS.memorisation_family(tr, items["pool_lb"])
        refs["mem"][c] = {"pool_lb": fam[member]}
        s_now = stat.S(e.y, fam[member], e.q, e.n)
        refs["checks"][f"mem_{c}"] = {"member": member, "S_registered": refs["mem_registered_S"][c]["pool_lb"], "S_recomputed": s_now}
        if abs(s_now - refs["mem_registered_S"][c]["pool_lb"]) > 1e-6:
            raise RuntimeError(f"memorisation reference of {c} recomputes to {s_now}, registered {refs['mem_registered_S'][c]['pool_lb']}")
        var = {**{k: v for k, v in fam.items() if k.startswith("all|")}, **memorisation_pair_only(tr, items["pool_lb"])}
        vS = {k: stat.S(e.y, v, e.q, e.n) for k, v in var.items()}
        vb = max(vS, key=vS.get)
        refs["mem_variant"][c] = {"pool_lb": var[vb]}
        refs["checks"][f"mem_variant_{c}"] = {"member": vb, "S": vS[vb]}
        if c in tj:
            best = tj[c]["pool_lb"]["best"]["name"]
            refs["text_best"][c] = (best, tz[f"{c}|pool_lb|{best}"])
            refs["checks"][f"text_{c}"] = {"best": best, "S_stored": tj[c]["pool_lb"]["best"]["S"], "S_recomputed": stat.S(e.y, tz[f"{c}|pool_lb|{best}"], e.q, e.n)}
        dist = DS.nearest_premise_distance(tr, items["pool_lb"])
        med = float(np.median(list(dist.values())))
        refs["near"][c] = np.array([dist[g] < med for g in e.groups])
        refs["checks"][f"near_{c}"] = {"median_distance": med, "n_near_items": int(refs["near"][c].sum())}
    return refs


def cached_refs(sets, items):
    """build_refs is slow (copy rules over every POOL-LB item); its result depends only on the stored sets and the two
    reference files, so it is cached next to the results and rebuilt when any of them is newer."""
    import pickle
    cache = OUT / ".refs_cache.pkl"
    deps = [ROOT / "finetune" / "itemsets" / n for n in ("references.json", "text_references.json", "text_references.npz")]
    deps += list((ROOT / "finetune" / "itemsets" / "sets").glob("*.json")) + [Path(DS.__file__), Path(__file__)]
    if cache.exists() and cache.stat().st_mtime > max(d.stat().st_mtime for d in deps):
        log("references from cache", cache.relative_to(ROOT))
        return pickle.loads(cache.read_bytes())
    log("building references")
    refs = build_refs(sets, items)
    OUT.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(refs))
    return refs


# --------------------------------------------------------------------------- synthetic scores
def synthetic(sets, design: str, seed: int = 0):
    """Scores that need no model: beta_model * (2y - 1) + a per-string offset + a quadratic in log tokens + noise.
    S rises with beta. planted: A 0.25, B 1.2, A48 0.6, C 0.0, Bswap 0.0 (B far above A), and a drop of 0.15 under the
    reversing renaming only; null: A, B, A48, C all 0.6, Bswap 0.0 (at chance), no renaming drop.
    Seed-to-seed spread of beta: 0.03."""
    rng = np.random.default_rng(seed)
    beta = {"planted": {"A": 0.25, "B": 1.2, "A48": 0.6, "C": 0.0, "Bswap": 0.0},
            "null": {**{c: 0.6 for c in BH.REGISTERED_SEEDS}, "Bswap": 0.0}}[design]
    scores = {}
    for c, nseed in BH.REGISTERED_SEEDS.items():
        for sd in range(nseed):
            b = beta[c] + 0.03 * rng.standard_normal()
            m = {}
            for k, e in sets.items():
                u, inv = np.unique(e.q, return_inverse=True)
                lx = np.log(e.n)
                drop = 0.15 if (k == "id48_renamed" and design == "planted") else 0.0
                m[k] = ((b - drop) * (2 * e.y - 1) + rng.standard_normal(len(u))[inv] + 0.8 * lx - 0.3 * lx ** 2
                        + rng.standard_normal(len(e.y)))
            scores[f"{c}_s{sd}"] = m
    return scores


# --------------------------------------------------------------------------- real scores (guarded)
def available_models():
    return sorted(p.name for p in RUNS.iterdir() if (p / "scores.npz").exists())


def load_scores(models, sets):
    BH.check_blinding(models)                       # before ANY A or B score file is opened
    out = {}
    for m in models:
        z = np.load(RUNS / m / "scores.npz")
        out[m] = {}
        for k, e in sets.items():
            uid = [str(u) for u in z[f"eval_{k}__uid"]]
            lab = z[f"eval_{k}__label"]
            if len(uid) != len(e.y) or not (np.asarray(lab).astype(int) == e.y).all():
                raise RuntimeError(f"{m}/{k}: labels or length differ from the stored set")
            if not np.allclose(z[f"eval_{k}__n_tokens"], e.n):
                raise RuntimeError(f"{m}/{k}: token counts differ from the other models'")
            out[m][k] = z[f"eval_{k}__score"].astype(float)
    return out


def base_report(sets, refs, scores):
    """The base model on every set: S, S inside cells (POOL-LB), classification, with premise-bootstrap intervals;
    beside its memorisation references. No test: one model, no seeds."""
    s = scores["base"]
    eng = BH.Engine(sets)
    r = {}
    for k, e in sets.items():
        pt = eng.point(k, "base", s[k])
        bt = eng.boot(k, "base", s[k])
        r[k] = {"S": pt, "S_premise_interval": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))],
                "eligible_strings": int(len(np.unique(e.q[stat.eligible(e.y, e.q)]))), **BH.classification(e.y, s[k])}
    pc = eng.point("pool_lb", "base", s["pool_lb"], "cells")
    bc = eng.boot("pool_lb", "base", s["pool_lb"], "cells")
    r["pool_lb"]["S_cells"] = pc
    r["pool_lb"]["S_cells_premise_interval"] = [float(np.quantile(bc, 0.025)), float(np.quantile(bc, 0.975))]
    r["max_discrepancy_fast_vs_registered_statistic"] = eng.max_discrepancy
    r["memorisation_references_pool_lb"] = {c: refs["mem_registered_S"][c]["pool_lb"] for c in refs["mem_registered_S"]}
    return r


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--synthetic", choices=["planted", "null"])
    g.add_argument("--base", action="store_true")
    a = ap.parse_args()
    if not (a.synthetic or a.base):                   # the registered run: refuse before anything heavy is loaded
        try:
            BH.check_blinding(available_models())
        except BH.Blinded as x:
            print("REFUSED:", x, file=sys.stderr)
            return 3
    t0 = time.time()
    log("building sets and resamples")
    sets, items = build_sets(RUNS / "base" / "scores.npz")
    refs = cached_refs(sets, items)
    _, which = impl.within_string_auroc_length_removed()
    meta = {"mode": a.synthetic and f"synthetic_{a.synthetic}" or ("base" if a.base else "registered"),
            "within_string_implementation": which, "n_boot": stat.N_BOOT, "reference_checks": refs["checks"],
            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.base:
        res = {"meta": meta, "base": base_report(sets, refs, load_scores(["base"], sets))}
    else:
        scores = synthetic(sets, a.synthetic) if a.synthetic else load_scores([m for m in available_models() if m != "base"], sets)
        log(f"analysing {len(scores)} models")
        res = {"meta": meta, **BH.analyse(sets, scores, refs)}
    OUT.mkdir(exist_ok=True)
    p = OUT / f"behaviour_{meta['mode']}.json"
    p.write_text(json.dumps(jsonable(res), indent=1) + "\n")
    log(f"wrote {p.relative_to(ROOT)} in {time.time() - t0:.0f} s")
    if "tests" in res:
        for k in ["E2c"] + BH.FAMILY:
            if k in res["tests"]:
                t = res["tests"][k]
                print(f"  {k:7s} D {t['D']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]  p {t['p_two_sided']:.4f}  "
                      f"holm {t.get('holm_p', float('nan')):.4f}  {t.get('direction', t.get('reading', ''))}")
        print("  E1 gate passed:", res["descriptive"]["E1"]["passed"], "| rungs:", res["descriptive"]["rung_A"], "/", res["descriptive"]["rung_B"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build every item set of the fine-tuning study on the laptop (CPU, about ten minutes, about 1.5 GB of memory).

  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/build_data.py                # build, verify, write finetune/itemsets/
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/build_data.py --check-only   # rebuild in memory, compare with the stored sets

Writes finetune/itemsets/sets/*.json (uids + transform + sha256), build_report.json (counts, the shortcut checks, the
text references that need no model, which implementation of the balancing function was used) and overlap_report.json
(no training set shares a premise or a graph class with any evaluation set). Exits non-zero if a requirement of
PREREG.md Section 1 fails. Design: PREREG.md (version 2), Section 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]

import numpy as np                                    # noqa: E402

from causal_probe import data as D, probes, prompts    # noqa: E402
from ftstudy import datasets as S, impl                # noqa: E402
from ftstudy.within_string_reference import within_string_auroc_length_removed as S_stat   # noqa: E402

SEED = 0
N_LENGTH_BINS = 4                     # fixed (PREREG v2): the build fails if the checks do not pass with it
PAIRS_TRAIN, PAIRS_EVAL = 8, 2        # pairs per premise in B and in POOL-LB
N_ID_PREMISES = 187                   # TRAIN-like premises held out at random as the in-distribution set
N_ID = 2000                           # items sampled from them
TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = ROOT / "finetune" / "itemsets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    balance, which = impl.balance_items()
    print("balancing function:", which)

    train, test = D.load_c2c("train"), D.load_c2c("test")
    dev_groups = {i["group"] for i in D.load_c2c("dev")}
    pool = D.get_dataset("c2c_debiased")
    pool_groups = {i["group"] for i in pool}
    by_uid = {i["uid"]: i for i in train}
    # ---- premises: the pool's 1,200 (selected by the v1.0 builder for holding valid items); of the rest, a random
    #      187 are the in-distribution evaluation premises and the others are the training premises
    rest = sorted({i["group"] for i in train} - pool_groups - dev_groups)
    id_groups = set(np.random.default_rng(SEED + 11).choice(rest, N_ID_PREMISES, replace=False))
    train_groups = set(rest) - id_groups
    TRAIN = [i for i in train if i["group"] in train_groups]
    IDALL = [i for i in train if i["group"] in id_groups]
    EVAL = [i for i in train if i["group"] in pool_groups]
    prem_tok = {}
    for i in train + test:
        if i["group"] not in prem_tok:
            prem_tok[i["group"]] = len(tok(i["context"], add_special_tokens=False)["input_ids"])

    def n_prompt(items):                               # token count of the full templated prompt, as the model sees it
        return np.array([len(prompts.encode_with_positions(tok, it)["input_ids"]) for it in items])

    rate = lambda items: round(float(np.mean([i["label"] for i in items])), 4)
    cells = Counter((i["query"], i["label"]) for i in TRAIN)
    two = {q for q, _ in cells if cells[(q, 0)] and cells[(q, 1)]}
    report = {"tokenizer": TOKENIZER, "balance_implementation": which, "seed": SEED,
              "premises": {"train_all": len({i["group"] for i in train}), "TRAIN": len(train_groups),
                           "ID_random_holdout": len(id_groups), "EVAL_pool": len(pool_groups), "test": len({i["group"] for i in test}),
                           "dev_unused": len(dev_groups)},
              "items": {"TRAIN": len(TRAIN), "ID_all_official_items": len(IDALL), "EVAL_pool_premises_all_official_items": len(EVAL)},
              "valid_rate": {"TRAIN": rate(TRAIN), "ID": rate(IDALL), "pool_premises": rate(EVAL), "official_test": rate(test),
                             "TRAIN_two_label_strings": rate([i for i in TRAIN if i["query"] in two]),
                             "ID_two_label_strings": rate([i for i in IDALL if i["query"] in two]),
                             "pool_premises_two_label_strings": rate([i for i in EVAL if i["query"] in two])},
              "two_label_strings_in_TRAIN": len(two)}

    # ---- B and POOL-LB, with the fixed number of length bins
    pt, pe = (np.array([prem_tok[i["group"]] for i in c]) for c in (TRAIN, EVAL))
    B = [TRAIN[j] for j in balance(TRAIN, pt, n_length_bins=N_LENGTH_BINS, max_pairs_per_premise=PAIRS_TRAIN, seed=SEED)]
    LB = [EVAL[j] for j in balance(EVAL, pe, n_length_bins=N_LENGTH_BINS, max_pairs_per_premise=PAIRS_EVAL, seed=SEED)]
    nB, nLB = n_prompt(B), n_prompt(LB)
    chk = {"B_in_sample": S.in_sample_balance(B, nB), "POOL_LB_in_sample": S.in_sample_balance(LB, nLB),
           "fit_B_test_POOL_LB": S.shortcut_transfer(B, nB, LB, nLB), "fit_POOL_LB_test_B": S.shortcut_transfer(LB, nLB, B, nB)}
    report["shortcut_checks"] = chk
    checks_ok = bool(chk["B_in_sample"]["passes"] and chk["POOL_LB_in_sample"]["passes"]
                     and chk["fit_B_test_POOL_LB"]["all_pass"] and chk["fit_POOL_LB_test_B"]["all_pass"])
    print("shortcut checks:", {d: {k: round(v["auroc"], 3) for k, v in chk[d].items() if isinstance(v, dict)}
                               for d in ("fit_B_test_POOL_LB", "fit_POOL_LB_test_B")}, "PASS" if checks_ok else "FAIL")

    # ---- the other conditions, all with N = len(B) items
    N = len(B)
    rng = np.random.default_rng(SEED)
    A = [TRAIN[j] for j in sorted(rng.choice(len(TRAIN), N, replace=False))]
    T48 = [i for i in TRAIN if i["query"] in two]
    A48 = [T48[j] for j in sorted(rng.choice(len(T48), N, replace=False))]
    first = {}
    for it in TRAIN:
        first.setdefault(it["group"], it)
    strata = S.swap_strata({g: (it["num_variables"], prem_tok[g]) for g, it in first.items()}, min_size=10)
    members = {}
    for g, sid in strata.items():
        members.setdefault(sid, []).append(g)

    def swapped(items):
        """Each item on the premise of another TRAIN premise of its stratum, drawn uniformly among premises."""
        donors = []
        for it in items:
            cand = [g for g in members[strata[it["group"]]] if g != it["group"]]
            donors.append(first[cand[int(rng.integers(len(cand)))]])
        return [S.swap_premise(it, d) for it, d in zip(items, donors)], donors

    C, donC = swapped(A)
    Bswap, donB = swapped(B)
    U = {(i["group"], i["query"]): i["label"] for i in TRAIN}

    def still_true(items):                             # share of swapped items whose label is the true label on the new premise
        v = [U.get((i["group"], i["query"])) for i in items]
        return {"share_label_correct_on_new_premise": float(np.mean([x == i["label"] for x, i in zip(v, items) if x is not None])),
                "share_with_that_hypothesis_on_new_premise": float(np.mean([x is not None for x in v]))}

    desc = lambda items: {**D.label_balance(items), "premises": len({i["group"] for i in items}), "strings": len({i["query"] for i in items}),
                          "items_in_two_label_strings": sum(i["query"] in two for i in items),
                          "valid_items_in_two_label_strings": sum(i["label"] for i in items if i["query"] in two)}
    report["conditions"] = {"N": N, "length_bins": N_LENGTH_BINS, "B_pairs_per_premise": PAIRS_TRAIN,
                            "A": desc(A), "A48": desc(A48), "B": {**desc(B), "by_template": dict(Counter(i["template"] for i in B))},
                            "C": {**desc(C), **still_true(C)}, "Bswap": {**desc(Bswap), **still_true(Bswap)},
                            "swap_strata": {"n": len(members), "smallest": min(map(len, members.values())), "largest": max(map(len, members.values()))}}

    # ---- evaluation sets
    ID = [IDALL[j] for j in sorted(np.random.default_rng(SEED + 1).choice(len(IDALL), N_ID, replace=False))]
    ID48_all = [i for i in IDALL if i["query"] in two]
    ID48 = [ID48_all[j] for j in sorted(np.random.default_rng(SEED + 2).choice(len(ID48_all), N_ID, replace=False))]
    pool_items = [by_uid[i["uid"]] for i in pool]
    sets = {"train_A": (A, "none", None), "train_A48": (A48, "none", None), "train_B": (B, "none", None),
            "train_C": (C, "premise_swap", donC), "train_Bswap": (Bswap, "premise_swap", donB),
            "eval_id": (ID, "none", None), "eval_pool": (pool_items, "none", None), "eval_pool_lb": (LB, "none", None),
            "eval_official_test": (test, "none", None),
            "eval_id_renamed": ([D.rename_variables(i) for i in ID], "rename", None),
            "eval_id_paraphrased": ([S.paraphrase(i) for i in ID], "paraphrase", None),
            "eval_id48": (ID48, "none", None),
            "eval_id48_renamed": ([D.rename_variables(i) for i in ID48], "rename", None),
            "eval_id48_renamed_keep_order": ([S.rename_keep_order(i) for i in ID48], "rename_keep_order", None),
            "eval_pool_lb_renamed": ([D.rename_variables(i) for i in LB], "rename", None)}
    authors = {k: S.load_authors(k) for k in ("paraphrasing", "refactorization")}

    # the paraphrase table and the renaming reproduce the authors' files on the official test items
    rep = {}
    for kind, fn in (("paraphrasing", S.paraphrase), ("refactorization", D.rename_variables)):
        au = {(i["context"], i["query"]): i["label"] for i in authors[kind]}
        mine = [fn(i) for i in test]
        rep[kind] = {"official_test_items": len(mine), "found_in_authors_file": sum((i["context"], i["query"]) in au for i in mine),
                     "same_label": sum(au.get((i["context"], i["query"])) == i["label"] for i in mine),
                     "distinct_inputs_in_authors_file": len(authors[kind])}
    report["our_transforms_against_authors_files"] = rep
    for nm, (items, _, _) in sets.items():
        report["items"][nm] = {**D.label_balance(items), "premises": len({i["group"] for i in items}),
                               "mean_prompt_tokens": float(n_prompt(items[:: max(1, len(items) // 300)]).mean())}
    for k, items in authors.items():
        report["items"]["eval_authors_" + k] = {**D.label_balance(items), "premises": len({i["group"] for i in items})}
    nper = Counter(i["group"] for i in LB)
    report["POOL_LB"] = {"n": len(LB), "premises": len(nper), "items_per_premise_max": max(nper.values()),
                         "pairs_per_string_min_median_max": [int(x) for x in np.percentile([c // 2 for c in Counter(i["query"] for i in LB).values()], [0, 50, 100])],
                         "by_template": dict(Counter(i["template"] for i in LB))}

    # ---- references that need no model (PREREG E1, E3, E5): fixed here, before any training
    nID, nID48 = n_prompt(ID), n_prompt(ID48)
    refs = {}
    y_id = np.array([i["label"] for i in ID48]); q_id = np.array([i["query"] for i in ID48])
    y_lb = np.array([i["label"] for i in LB]); q_lb = np.array([i["query"] for i in LB])
    for nm, tr_items in (("A", A), ("A48", A48), ("B", B), ("C", C)):
        r = {"shortcut_classifiers_fitted_on_it_tested_on_ID": S.shortcut_transfer(tr_items, n_prompt(tr_items), ID, nID, n_boot=200)}
        if nm != "C":
            for set_nm, items, y, q, n in (("ID48", ID48, y_id, q_id, nID48), ("POOL_LB", LB, y_lb, q_lb, nLB)):
                sc = S.nearest_premise_scores(tr_items, items)
                keep = np.array([x in two for x in q])
                r[f"nearest_premise_copy_on_{set_nm}"] = {
                    "S_two_label_strings": S_stat(y[keep], sc[keep], q[keep], n[keep])["mean_auroc"],
                    "pooled_auroc": S._auc(y, sc), "share_of_items_with_a_neighbour": float(np.mean(sc != 0.5))}
        refs[nm] = r
    refs["length_alone_raw_within_string_on_ID48_shorter_is_valid"] = float(np.mean(
        [S._auc(y_id[q_id == s], -nID48[q_id == s]) for s in np.unique(q_id) if 0 < y_id[q_id == s].mean() < 1]))
    for nm, tr_items in (("A", A), ("A48", A48), ("B", B)):       # the hypothesis-string prior of each training set, on ID
        rt = {}
        for it in tr_items:
            rt.setdefault(it["query"], []).append(it["label"])
        sc = np.array([(sum(rt[i["query"]]) + 1) / (len(rt[i["query"]]) + 2) if i["query"] in rt else 0.5 for i in ID])
        yy = np.array([i["label"] for i in ID])
        best = max(((2 * ((sc >= t) & (yy == 1)).sum()) / max(1, (sc >= t).sum() + yy.sum()) for t in np.unique(sc)))
        refs[nm]["hypothesis_string_prior_on_ID"] = {"pooled_auroc": S._auc(yy, sc), "valid_class_F1_at_its_best_threshold": float(best)}
    dist = S.nearest_premise_distance(TRAIN, LB)
    dv = np.array(list(dist.values()))
    refs["distance_of_POOL_LB_premises_to_nearest_TRAIN_premise"] = {"median": float(np.median(dv)), "p25": float(np.percentile(dv, 25)),
                                                                     "p75": float(np.percentile(dv, 75)), "max": int(dv.max())}
    # the E5 cell: hypothesis string x v1.0 adjacency (probes.structural_features[:, 2])
    adj = probes.structural_features(LB)[:, 2].astype(int)
    cell = Counter((q, a_, y) for q, a_, y in zip(q_lb, adj, y_lb))
    keys = {(q, a_) for q, a_, _ in cell}
    two_class = [(q, a_) for q, a_ in keys if min(cell[(q, a_, 0)], cell[(q, a_, 1)]) >= 3]
    refs["E5_cells_on_POOL_LB"] = {"cells": len(keys), "cells_with_at_least_3_per_class": len(two_class),
                                   "items_in_those_cells": int(sum(cell[(q, a_, 0)] + cell[(q, a_, 1)] for q, a_ in two_class)),
                                   "strings_with_such_a_cell": len({q for q, _ in two_class}),
                                   "valid_invalid_pairs_in_those_cells": int(sum(cell[(q, a_, 0)] * cell[(q, a_, 1)] for q, a_ in two_class)),
                                   "adjacency_alone_S_on_POOL_LB": S_stat(y_lb, adj.astype(float), q_lb, nLB)["mean_auroc"]}
    report["references"] = refs

    # ---- the proof of separation: premise text, premise text with any renaming undone, graph class
    undo = {"rename": {v: k for k, v in D.RENAME_MAP.items()}, "rename_keep_order": {v: k for k, v in S.KEEP_ORDER_MAP.items()}}

    def keys_of(items, how):
        prem = {i["context"] for i in items}
        orig = {re.sub(r"\b([U-Z])\b", lambda m: undo[how][m.group(1)], p) for p in prem} if how in undo else prem
        return prem, orig, {S.graph_class(p) for p in orig}

    tr_sets = {k: v for k, v in sets.items() if k.startswith("train_")}
    ev_sets = {**{k: v for k, v in sets.items() if k.startswith("eval_")},
               "eval_authors_paraphrasing": (authors["paraphrasing"], "none", None),
               "eval_authors_refactorization": (authors["refactorization"], "rename", None)}
    K = {k: keys_of(v[0], v[1]) for k, v in {**tr_sets, **ev_sets}.items()}
    all_prem = {i["context"] for i in TRAIN} | {i["context"] for i in IDALL} | {i["context"] for i in EVAL} | {i["context"] for i in test}
    overlap = {"definition": "shared premise texts as given / shared premise texts with any renaming undone / shared graph classes "
                             "(statement set canonicalised over every relabelling of the variables)",
               "premises_TRAIN_ID_pool_test": len(all_prem), "graph_classes_among_them": len({S.graph_class(p) for p in all_prem}), "pairs": {}}
    worst = 0
    for t in tr_sets:
        for e in ev_sets:
            o = [len(K[t][j] & K[e][j]) for j in range(3)]
            overlap["pairs"][f"{t} x {e}"] = {"premise_text": o[0], "renaming_undone": o[1], "graph_class": o[2],
                                             "train_premises": len(K[t][0]), "eval_premises": len(K[e][0])}
            worst = max(worst, *o)
    overlap["largest_overlap_of_any_pair"] = worst
    report["requirements"] = {
        "B and POOL-LB pass the shortcut checks with the fixed number of bins": checks_ok,
        "zero overlap of every training set with every evaluation set": worst == 0,
        "every premise is its own graph class": overlap["premises_TRAIN_ID_pool_test"] == overlap["graph_classes_among_them"],
        "paraphrase table and renaming reproduce the authors' files": all(v["official_test_items"] == v["found_in_authors_file"] == v["same_label"] for v in rep.values()),
        "all five conditions have N items": len({len(v[0]) for v in tr_sets.values()}) == 1,
        "C keeps A's hypotheses and labels, Bswap keeps B's": all(x["query"] == z["query"] and x["label"] == z["label"] for x, z in zip(A + B, C + Bswap)),
        "every swap stratum has at least two premises": min(map(len, members.values())) >= 2}
    for k, v in report["requirements"].items():
        print("ok  " if v else "FAIL", k)

    if a.check_only:
        bad = [nm for nm, (items, _, _) in sets.items()
               if not (S.SETS / f"{nm}.json").exists() or json.loads((S.SETS / f"{nm}.json").read_text())["sha256"] != S.digest(items)]
        print("check-only: sets that differ from the stored ones:", bad or "none")
        sys.exit(1 if bad or not all(report["requirements"].values()) else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in S.SETS.glob("*.json"):
        f.unlink()                                     # sets of an earlier design must not survive a rebuild
    for nm, (items, tf, dn) in sets.items():
        S.save_set(nm, items, tf, dn)
    for k, items in authors.items():
        S.save_set("eval_authors_" + k, items, "authors", kind=k)
        f = S.SETS / f"eval_authors_{k}.json"
        spec = json.loads(f.read_text()); spec["uids"] = []; f.write_text(json.dumps(spec) + "\n")
    (OUT / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")
    (OUT / "overlap_report.json").write_text(json.dumps(overlap, indent=1) + "\n")
    sys.exit(0 if all(report["requirements"].values()) else 1)


if __name__ == "__main__":
    main()

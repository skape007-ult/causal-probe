"""Build every item set of the fine-tuning study on the laptop (CPU, a few minutes, about 1.5 GB of memory).

  ../causal-probe/.venv/bin/python finetune/scripts/build_data.py            # build, verify, write finetune/itemsets/
  ../causal-probe/.venv/bin/python finetune/scripts/build_data.py --check-only   # rebuild in memory, compare with the stored sets

Writes finetune/itemsets/sets/*.json (uids + transform + sha256), finetune/itemsets/build_report.json (counts, the shortcut
checks of B and POOL-LB, which implementation of the balancing function was used) and
finetune/itemsets/overlap_report.json (the proof that no training set shares a premise or a graph class with any
evaluation set). Exits non-zero if a requirement of PREREG.md Section 1 fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]

import numpy as np                                    # noqa: E402

from causal_probe import data as D, prompts            # noqa: E402
from ftstudy import datasets as S, impl                # noqa: E402

SEED = 0
BINS_TO_TRY = (4, 8, 16, 32)          # the smallest number of length bins whose set passes the shortcut checks is used
PAIRS_TRAIN, PAIRS_EVAL = 8, 2        # pairs per premise in B and in POOL-LB
N_ID = 3000                           # ID-heldout
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
    TRAIN = [i for i in train if i["group"] not in pool_groups and i["group"] not in dev_groups]
    EVAL = [i for i in train if i["group"] in pool_groups]
    prem_tok = {}
    for i in train + test:
        if i["group"] not in prem_tok:
            prem_tok[i["group"]] = len(tok(i["context"], add_special_tokens=False)["input_ids"])

    def n_prompt(items):                               # token count of the full prompt, as the model sees it
        return np.array([len(prompts.encode_with_positions(tok, it)["input_ids"]) for it in items])

    report = {"tokenizer": TOKENIZER, "balance_implementation": which, "seed": SEED,
              "premises": {"train_all": len({i["group"] for i in train}), "TRAIN": len({i["group"] for i in TRAIN}),
                           "EVAL_pool": len(pool_groups), "test": len({i["group"] for i in test}), "dev_unused": len(dev_groups)},
              "items": {"TRAIN": len(TRAIN), "EVAL_pool_premises_all_official_items": len(EVAL)}}

    def balanced_pair():
        """B (from TRAIN premises) and POOL-LB (from the pool premises), with the smallest number of length bins for
        which every shortcut classifier fitted on one fails to predict the other, in both directions."""
        pt, pe = (np.array([prem_tok[i["group"]] for i in c]) for c in (TRAIN, EVAL))
        for nb in BINS_TO_TRY:
            Bs = [TRAIN[j] for j in balance(TRAIN, pt, n_length_bins=nb, max_pairs_per_premise=PAIRS_TRAIN, seed=SEED)]
            Ls = [EVAL[j] for j in balance(EVAL, pe, n_length_bins=nb, max_pairs_per_premise=PAIRS_EVAL, seed=SEED)]
            nB, nL = n_prompt(Bs), n_prompt(Ls)
            chk = {"B_in_sample": S.in_sample_balance(Bs, nB), "POOL_LB_in_sample": S.in_sample_balance(Ls, nL),
                   "fit_B_test_POOL_LB": S.shortcut_transfer(Bs, nB, Ls, nL), "fit_POOL_LB_test_B": S.shortcut_transfer(Ls, nL, Bs, nB)}
            ok = chk["B_in_sample"]["passes"] and chk["POOL_LB_in_sample"]["passes"] and chk["fit_B_test_POOL_LB"]["all_pass"] and chk["fit_POOL_LB_test_B"]["all_pass"]
            print(f"{nb} bins: B {len(Bs)} items, POOL-LB {len(Ls)} items;",
                  {d: {k: round(v["auroc"], 3) for k, v in chk[d].items() if isinstance(v, dict)} for d in ("fit_B_test_POOL_LB", "fit_POOL_LB_test_B")},
                  "within-string token AUROC (mean, worst):", round(chk["B_in_sample"]["token_count_within_string"]["mean_auroc"], 3),
                  round(chk["B_in_sample"]["token_count_within_string"]["largest_deviation_of_a_string"], 3), "PASS" if ok else "fail")
            report.setdefault("bins_tried", {})[str(nb)] = {"n_B": len(Bs), "n_POOL_LB": len(Ls), "passes": ok, "checks": chk}
            if ok:
                return Bs, Ls, nb
        raise SystemExit(f"no bin count in {BINS_TO_TRY} passes the shortcut checks; no model may be trained")

    # ---- training conditions
    B, LB, nb_B = balanced_pair()
    nb_LB = nb_B
    N = len(B)
    rng = np.random.default_rng(SEED)
    A = [TRAIN[j] for j in sorted(rng.choice(len(TRAIN), N, replace=False))]
    # C: A's items on the premise of another TRAIN item with the same number of variables and the same length bin
    edges = np.quantile(sorted(prem_tok[g] for g in {i["group"] for i in TRAIN}), np.arange(1, 16) / 16)
    cell = lambda it: (it["num_variables"], int(np.searchsorted(edges, prem_tok[it["group"]], side="right")))
    first = {}
    for it in TRAIN:
        first.setdefault(it["group"], it)
    donors_by_cell = {}
    for g, it in first.items():
        donors_by_cell.setdefault(cell(it), []).append(it)
    donors, no_donor = [], 0
    for it in A:
        cand = [d for d in donors_by_cell[cell(it)] if d["group"] != it["group"]]
        if not cand:                                   # a cell with one premise: take the nearest length with the same variable count
            cand = sorted((d for d in first.values() if d["num_variables"] == it["num_variables"] and d["group"] != it["group"]),
                          key=lambda d: abs(prem_tok[d["group"]] - prem_tok[it["group"]]))[:5]
            no_donor += 1
        donors.append(cand[int(rng.integers(len(cand)))])
    C = [S.swap_premise(it, d) for it, d in zip(A, donors)]
    report["conditions"] = {
        "N": N, "B_length_bins": nb_B, "B_pairs_per_premise": PAIRS_TRAIN,
        "A": {**D.label_balance(A), "premises": len({i["group"] for i in A}), "strings": len({i["query"] for i in A})},
        "B": {**D.label_balance(B), "premises": len({i["group"] for i in B}), "strings": len({i["query"] for i in B}),
              "by_template": dict(Counter(i["template"] for i in B))},
        "C": {**D.label_balance(C), "premises": len({i["group"] for i in C}), "items_whose_cell_had_no_other_premise": no_donor,
              "mean_abs_token_difference_to_own_premise": float(np.mean([abs(prem_tok[c["group"]] - prem_tok[c["own_group"]]) for c in C]))}}
    ID = [EVAL[j] for j in sorted(np.random.default_rng(SEED + 1).choice(len(EVAL), N_ID, replace=False))]
    nID = n_prompt(ID)
    for nm, items in (("A", A), ("C", C)):          # for comparison: on the official distribution the same classifiers do predict
        report["conditions"][nm]["shortcut_classifiers_fitted_here_tested_on_ID_heldout"] = S.shortcut_transfer(items, n_prompt(items), ID, nID, n_boot=200)

    # ---- evaluation sets
    pool_items = [by_uid[i["uid"]] for i in pool]
    ev = {"id_heldout": ID, "pool": pool_items, "pool_lb": LB, "official_test": test}
    report["evaluation"] = {"POOL_LB_length_bins": nb_LB, "POOL_LB_pairs_per_premise": PAIRS_EVAL}
    sets = {"train_A": (A, "none", None), "train_B": (B, "none", None), "train_C": (C, "premise_swap", donors)}
    for nm, items in ev.items():
        sets["eval_" + nm] = (items, "none", None)
        if nm != "pool":
            sets[f"eval_{nm}_renamed"] = ([D.rename_variables(i) for i in items], "rename", None)
            sets[f"eval_{nm}_paraphrased"] = ([S.paraphrase(i) for i in items], "paraphrase", None)
    authors = {k: S.load_authors(k) for k in ("paraphrasing", "refactorization")}

    # the paraphrase table reproduces the authors' file: every paraphrased official test item is in it, same label
    au = {(i["context"], i["query"]): i["label"] for i in authors["paraphrasing"]}
    mine = sets["eval_official_test_paraphrased"][0]
    found = sum((i["context"], i["query"]) in au for i in mine)
    agree = sum(au.get((i["context"], i["query"])) == i["label"] for i in mine)
    report["paraphrase_table_against_authors_file"] = {"official_test_items": len(mine), "found_in_authors_file": found, "same_label": agree}
    ar = {(i["context"], i["query"]): i["label"] for i in authors["refactorization"]}
    mine_r = sets["eval_official_test_renamed"][0]
    report["renaming_against_authors_file"] = {"official_test_items": len(mine_r), "found_in_authors_file": sum((i["context"], i["query"]) in ar for i in mine_r),
                                               "same_label": sum(ar.get((i["context"], i["query"])) == i["label"] for i in mine_r)}

    for nm, (items, _, _) in sets.items():
        report["items"][nm] = {**D.label_balance(items), "premises": len({i["group"] for i in items}),
                               "mean_prompt_tokens": float(n_prompt(items[:: max(1, len(items) // 400)]).mean())}
    for k, items in authors.items():
        report["items"]["eval_authors_" + k] = {**D.label_balance(items), "premises": len({i["group"] for i in items})}

    # ---- the proof of separation: premise text, premise text after renaming, graph class
    def keys(items):
        prem = {i["context"] for i in items}
        ren = {D._VAR_RE.sub(lambda m: {**D.RENAME_MAP, **{v: k for k, v in D.RENAME_MAP.items()}}.get(m.group(1), m.group(1)), p) for p in prem}
        return prem, prem | ren, {S.graph_class(_unrename(p)) for p in prem}

    def _unrename(p):                                   # the authors' refactored premises use Z..U: map back before parsing
        back = {v: k for k, v in D.RENAME_MAP.items()}
        import re
        return re.sub(r"\b([U-Z])\b", lambda m: back[m.group(1)], p)

    tr_sets = {k: v[0] for k, v in sets.items() if k.startswith("train_")}
    ev_sets = {**{k: v[0] for k, v in sets.items() if k.startswith("eval_")}, **{"eval_authors_" + k: v for k, v in authors.items()}}
    K = {k: keys(v) for k, v in {**tr_sets, **ev_sets}.items()}
    overlap = {"definition": "shared premise texts / shared after renaming either way / shared graph classes (statement set up to relabelling)",
               "graph_classes_among_all_train_premises": len({S.graph_class(i["context"]) for i in first.values()} | {S.graph_class(p) for p in {i["context"] for i in EVAL}}),
               "pairs": {}}
    worst = 0
    for t in tr_sets:
        for e in ev_sets:
            o = [len(K[t][j] & K[e][j]) for j in range(3)]
            overlap["pairs"][f"{t} x {e}"] = {"premise_text": o[0], "after_renaming": o[1], "graph_class": o[2],
                                             "train_premises": len(K[t][0]), "eval_premises": len(K[e][0])}
            worst = max(worst, *o)
    overlap["largest_overlap_of_any_pair"] = worst
    report["requirements"] = {
        "B and POOL-LB pass the shortcut checks": report["bins_tried"][str(nb_B)]["passes"],
        "zero overlap of every training set with every evaluation set": worst == 0,
        "paraphrase table reproduces the authors' file": found == len(mine) == agree,
        "A, B and C have the same number of items": len(A) == len(B) == len(C),
        "C keeps A's hypotheses and labels": all(x["query"] == z["query"] and x["label"] == z["label"] for x, z in zip(A, C))}
    for k, v in report["requirements"].items():
        print("ok  " if v else "FAIL", k)

    if a.check_only:
        bad = [nm for nm, (items, _, _) in sets.items()
               if not (S.SETS / f"{nm}.json").exists() or json.loads((S.SETS / f"{nm}.json").read_text())["sha256"] != S.digest(items)]
        print("check-only: sets that differ from the stored ones:", bad or "none")
        sys.exit(1 if bad or not all(report["requirements"].values()) else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for nm, (items, tf, dn) in sets.items():
        S.save_set(nm, items, tf, dn)
    for k, items in authors.items():
        S.save_set("eval_authors_" + k, items, "authors", kind=k)
        json_path = S.SETS / f"eval_authors_{k}.json"
        spec = json.loads(json_path.read_text()); spec["uids"] = []; json_path.write_text(json.dumps(spec) + "\n")
    (OUT / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")
    (OUT / "overlap_report.json").write_text(json.dumps(overlap, indent=1) + "\n")
    print(json.dumps(report["conditions"], indent=1)[:1500])
    print({k: (v["n"], v["premises"]) for k, v in report["items"].items() if isinstance(v, dict)})
    sys.exit(0 if all(report["requirements"].values()) else 1)


if __name__ == "__main__":
    main()

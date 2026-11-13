"""Dataset loaders, stratified subsampling, grouped splits and the variable-renaming perturbation.

Every item is a dict with at least:
  uid      unique string id
  dataset  "c2c" | "cladder"
  context  the premise (Corr2Cause) or background + given info (CLadder)
  query    the hypothesis (Corr2Cause) or the question (CLadder)
  label    1 = "Yes"/valid, 0 = "No"/invalid
  group    the leakage unit: premise id (Corr2Cause) or story_id (CLadder)
plus dataset-specific metadata (template, num_variables, rung, query_type, graph_id ...).
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np

from .config import (C2C_DEBIASED_PER_CELL, C2C_PROBE_PER_PREMISE, C2C_PROBE_PREMISES, CLADDER_N,
                     DATA_RAW, N_SUB, SEED)

csv.field_size_limit(10**9)
HYP_SEP = "\nHypothesis: "


def _h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- Corr2Cause
def load_c2c(split: str) -> list[dict]:
    """Load a Corr2Cause split ("train" | "dev" | "test") from the raw CSV."""
    out = []
    with open(DATA_RAW / "corr2cause" / f"{split}.csv") as f:
        for i, r in enumerate(csv.DictReader(f)):
            assert r["input"].startswith("Premise: ") and r["input"].count(HYP_SEP) == 1
            prem, hyp = r["input"][len("Premise: "):].split(HYP_SEP)
            out.append(dict(
                uid=f"c2c-{split}-{i}", dataset="c2c", split=split,
                context=prem, query=hyp, label=int(r["label"]),
                group=f"prem-{_h(prem)}", template=r["template"],
                num_variables=int(r["num_variables"]),
            ))
    return out


def c2c_probe_pool(train: list[dict], seed: int = SEED,
                   n_premises: int = C2C_PROBE_PREMISES,
                   per_premise: int = C2C_PROBE_PER_PREMISE) -> list[dict]:
    """Label-balanced probe pool: sample premises, then per premise an equal number of
    positive and negative hypotheses (half of `per_premise` each). Premises with too few
    positives are skipped, so the pool is exactly 50/50 by construction."""
    rng = np.random.default_rng(seed)
    by_prem = defaultdict(list)
    for it in train:
        by_prem[it["group"]].append(it)
    half = per_premise // 2
    prems = sorted(g for g, its in by_prem.items()
                   if sum(i["label"] for i in its) >= half
                   and sum(1 - i["label"] for i in its) >= half)
    chosen = rng.choice(prems, size=n_premises, replace=False)
    pool = []
    for g in chosen:
        its = by_prem[g]
        for lab in (1, 0):
            cand = [i for i in its if i["label"] == lab]
            idx = rng.choice(len(cand), size=half, replace=False)
            pool.extend(cand[j] for j in sorted(idx))
    return pool


def c2c_debiased_pool(train: list[dict], seed: int = SEED,
                      per_cell: int = C2C_DEBIASED_PER_CELL) -> list[dict]:
    """Corr2Cause pool in which the hypothesis string carries no label information.

    In the official data most hypothesis strings ("D directly causes C.") have the same label for
    every premise, so the string alone predicts the label. Here we keep only the strings that
    occur with BOTH labels and sample exactly `per_cell` valid and `per_cell` invalid items for
    each of them. Every hypothesis string (and therefore every template and every variable pair)
    is then exactly 50/50, and a classifier that sees only the hypothesis is at chance by
    construction. Premises still repeat across items, so evaluation must hold out by premise
    (`group`), as everywhere else. Items are returned sorted by (hypothesis, label, uid)."""
    rng = np.random.default_rng(seed)
    cells = defaultdict(list)
    for it in train:
        cells[(it["query"], it["label"])].append(it)
    strings = sorted({q for q, _ in cells})
    both = [q for q in strings if len(cells.get((q, 0), [])) >= per_cell and len(cells.get((q, 1), [])) >= per_cell]
    pool = []
    for q in both:
        for lab in (0, 1):
            cand = cells[(q, lab)]
            idx = rng.choice(len(cand), size=per_cell, replace=False)
            pool.extend(cand[j] for j in sorted(idx))
    return pool


def c2c_debiased_stats(items: list[dict], per_cell: int = C2C_DEBIASED_PER_CELL) -> dict:
    """How much of a split survives the debiasing rule (for the write-up)."""
    cells = Counter((i["query"], i["label"]) for i in items)
    strings = sorted({q for q, _ in cells})
    both = [q for q in strings if cells[(q, 0)] > 0 and cells[(q, 1)] > 0]
    return {"n_items": len(items), "n_hypothesis_strings": len(strings),
            "n_strings_with_both_labels": len(both),
            "n_strings_single_label": len(strings) - len(both),
            "n_items_in_single_label_strings": sum(cells[(q, 0)] + cells[(q, 1)] for q in strings if q not in set(both)),
            "max_items_balanced_within_string": sum(2 * min(cells[(q, 0)], cells[(q, 1)]) for q in both),
            "min_cell_size_among_both": min((min(cells[(q, 0)], cells[(q, 1)]) for q in both), default=0),
            "n_strings_with_cells_at_least_per_cell": sum(min(cells[(q, 0)], cells[(q, 1)]) >= per_cell for q in both)}


_VAR_RE = re.compile(r"\b([A-F])\b")
RENAME_MAP = dict(zip("ABCDEF", "ZYXWVU"))


def rename_variables(item: dict) -> dict:
    """Corr2Cause 'variable refactorisation' robustness check (paper Sec. on perturbations):
    A,B,C,... -> Z,Y,X,...  Applied to premise and hypothesis; label unchanged."""
    def sub(s):  # "A" as an article never occurs as a standalone capital in this template text
        return _VAR_RE.sub(lambda m: RENAME_MAP[m.group(1)], s)
    new = dict(item)
    new["uid"] = item["uid"] + "-ren"
    new["context"], new["query"] = sub(item["context"]), sub(item["query"])
    return new


# ----------------------------------------------------------------------------- CLadder
def _split_cladder_prompt(prompt: str) -> tuple[str, str]:
    """Split the CLadder prompt into (background + given info, question).
    The question is the final sentence; we cut after the last '. ' (or '.' + space)
    that precedes it."""
    p = prompt.strip()
    cut = p.rfind(". ")
    assert cut > 0, prompt
    return p[:cut + 1], p[cut + 2:]


def load_cladder() -> list[dict]:
    out = []
    with open(DATA_RAW / "cladder" / "full_v1.5_default.csv") as f:
        for r in csv.DictReader(f):
            ctx, q = _split_cladder_prompt(r["prompt"])
            out.append(dict(
                uid=f"cladder-{r['id']}", dataset="cladder", split="full",
                context=ctx, query=q, label=int(r["label"] == "yes"),
                group=r["story_id"], rung=int(r["rung"]), query_type=r["query_type"],
                graph_id=r["graph_id"], question_property=r["question_property"],
            ))
    return out


def stratified_sample(items: list[dict], n: int, keys: tuple[str, ...], seed: int = SEED) -> list[dict]:
    """Proportional stratified sample of n items over the cross of `keys`
    (largest-remainder allocation, so the total is exactly n)."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for it in items:
        strata[tuple(it[k] for k in keys)].append(it)
    names = sorted(strata)
    quota = np.array([len(strata[s]) * n / len(items) for s in names])
    alloc = np.floor(quota).astype(int)
    for j in np.argsort(-(quota - alloc))[: n - alloc.sum()]:
        alloc[j] += 1
    out = []
    for s, a in zip(names, alloc):
        idx = rng.choice(len(strata[s]), size=min(a, len(strata[s])), replace=False)
        out.extend(strata[s][j] for j in sorted(idx))
    return out


def cladder_sample(seed: int = SEED, n: int = CLADDER_N) -> list[dict]:
    return stratified_sample(load_cladder(), n, ("label", "query_type"), seed)


# ----------------------------------------------------------------------------- shared
def label_balance(items: list[dict]) -> dict:
    c = Counter(i["label"] for i in items)
    return {"n": len(items), "n_pos": c[1], "n_neg": c[0], "frac_pos": c[1] / max(1, len(items))}


def grouped_folds(items: list[dict], n_folds: int, seed: int):
    """Stratified (by label) K-fold where no `group` (premise / story) is in both train and test."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([i["label"] for i in items])
    g = np.array([i["group"] for i in items])
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, g))


def _from_train(tag: str, build) -> list[dict]:
    """Pools sampled from the 383 MB train file are stored once as a small JSON under
    data/derived/, so later scripts do not each parse the whole file (about 1 GB of Python
    objects, which matters on an 8 GB machine). The file name carries every sampling parameter."""
    import json
    name = f"{tag}_seed{SEED}_{C2C_PROBE_PREMISES}x{C2C_PROBE_PER_PREMISE}_{C2C_DEBIASED_PER_CELL}.json"
    path = DATA_RAW.parent / "derived" / name
    if path.exists():
        return json.loads(path.read_text())
    items = build(load_c2c("train"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items))
    tmp.rename(path)
    return items


def get_dataset(tag: str) -> list[dict]:
    """The named item sets that get cached. One place, so every script agrees on order."""
    if tag == "c2c_test":
        return load_c2c("test")
    if tag == "c2c_probe":
        return _from_train(tag, c2c_probe_pool)
    if tag == "c2c_debiased":
        return _from_train(tag, c2c_debiased_pool)
    if tag == "c2c_test_renamed":
        return [rename_variables(i) for i in load_c2c("test")]
    if tag == "cladder":
        return cladder_sample()
    if tag in ("c2c_test_sub", "cladder_sub"):   # evenly spaced subsets for generation / few-shot
        full = get_dataset(tag.removesuffix("_sub"))
        idx = np.unique(np.linspace(0, len(full) - 1, N_SUB).round().astype(int))
        return [full[i] for i in idx]
    raise KeyError(tag)

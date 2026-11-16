"""Part (a): the same tests run on the owner's implementation and on the reference.
The owner's is skipped (not failed) while it still raises NotImplementedError."""
from collections import Counter, defaultdict

import numpy as np
import pytest

from ftstudy import balance, balance_reference

IMPLS = {"owner": balance, "reference": balance_reference}


def synthetic(seed=0, n_prem=300, templates=("parent", "has_collider"), strings_per_template=6, both_share=0.7):
    """Premises with random lengths; shorter premises make more hypotheses valid (the leak of the real data).
    Some strings never occur valid (single-label strings, as in the real data)."""
    rng = np.random.default_rng(seed)
    items, ntok = [], []
    for g in range(n_prem):
        L = int(rng.integers(80, 900))
        for t in templates:
            for k in range(strings_per_template):
                if rng.random() < 0.25:
                    continue                                   # not every premise has every string
                single = k >= both_share * strings_per_template
                p_valid = 0.0 if single else float(np.clip(0.75 - L / 1200 + 0.08 * k, 0.05, 0.95))
                items.append({"uid": f"u{len(items)}", "group": f"prem-{g}", "template": t, "query": f"{t}-string-{k}",
                              "label": int(rng.random() < p_valid)})
                ntok.append(L)
    return items, np.array(ntok)


def run(mod, items, ntok, n_bins=4, k=3, seed=0):
    try:
        return mod.balance_items(items, ntok, n_length_bins=n_bins, max_pairs_per_premise=k, seed=seed)
    except NotImplementedError:
        pytest.skip("owner's part (a) is not implemented yet")


def spec_bins(items, ntok, n_bins):
    """Length bin of every item exactly as MY_PARTS.md defines it (None for items of single-label strings)."""
    lab = defaultdict(set)
    for it in items:
        lab[it["query"]].add(it["label"])
    out = [None] * len(items)
    for t in {it["template"] for it in items}:
        idx = [i for i, it in enumerate(items) if it["template"] == t and lab[it["query"]] == {0, 1}]
        prem = {items[i]["group"]: int(ntok[i]) for i in idx}
        edges = np.quantile(np.array(sorted(prem.values()), float), np.arange(1, n_bins) / n_bins) if n_bins > 1 and prem else []
        for i in idx:
            out[i] = int(np.searchsorted(edges, ntok[i], side="right"))
    return out


@pytest.mark.parametrize("name", IMPLS)
@pytest.mark.parametrize("n_bins,k,seed", [(1, 1, 0), (4, 3, 0), (6, 2, 1), (3, 8, 2)])
def test_balance_properties(name, n_bins, k, seed):
    items, ntok = synthetic(seed)
    sel = run(IMPLS[name], items, ntok, n_bins, k, seed)
    assert len(sel) > 0, "an empty selection is balanced but useless"
    assert sel == sorted(set(sel)) and all(0 <= i < len(items) for i in sel), "sorted indices into items, no repeats"
    S = [items[i] for i in sel]
    bins = spec_bins(items, ntok, n_bins)

    def balanced(key):
        c = defaultdict(Counter)
        for i in sel:
            c[key(i)][items[i]["label"]] += 1
        return [k_ for k_, v in c.items() if v[0] != v[1]]

    assert balanced(lambda i: items[i]["query"]) == [], "P1: valid = invalid inside every hypothesis string"
    assert balanced(lambda i: items[i]["group"]) == [], "P2: valid = invalid inside every premise"
    assert balanced(lambda i: (items[i]["template"], bins[i])) == [], "P3: ... inside every type x length bin"
    assert balanced(lambda i: (items[i]["query"], bins[i])) == [], "P4: ... inside every string x length bin"
    assert max(Counter(it["group"] for it in S).values()) <= 2 * k, "P5: at most max_pairs_per_premise pairs per premise"


@pytest.mark.parametrize("name", IMPLS)
def test_deterministic_and_input_untouched(name):
    items, ntok = synthetic(3)
    before = [dict(it) for it in items]
    a = run(IMPLS[name], items, ntok, seed=5)
    b = run(IMPLS[name], items, ntok, seed=5)
    assert a == b and items == before


@pytest.mark.parametrize("name", IMPLS)
def test_length_no_longer_predicts_the_label_within_string(name):
    from sklearn.metrics import roc_auc_score
    items, ntok = synthetic(4, n_prem=1500)
    sel = run(IMPLS[name], items, ntok, n_bins=8, k=3, seed=0)

    def length_auc(idx):
        by = defaultdict(list)
        for i in idx:
            by[items[i]["query"]].append(i)
        a = [roc_auc_score([items[i]["label"] for i in v], ntok[v]) for v in by.values() if len({items[i]["label"] for i in v}) == 2]
        return float(np.mean(a))
    assert abs(length_auc(range(len(items))) - 0.5) > 0.08, "the synthetic data must leak, or this test shows nothing"
    assert abs(length_auc(sel) - 0.5) < 0.03


@pytest.mark.parametrize("name", IMPLS)
def test_keeps_a_useful_share(name):
    """Not optimality: only that the function does not throw most of the data away."""
    items, ntok = synthetic(5, n_prem=600)
    n = len(run(IMPLS[name], items, ntok, n_bins=4, k=3, seed=0))
    ref = len(balance_reference.balance_items(items, ntok, n_length_bins=4, max_pairs_per_premise=3, seed=0))
    assert n >= 0.5 * ref


@pytest.mark.parametrize("name", IMPLS)
def test_edge_cases(name):
    assert run(IMPLS[name], [], np.array([], int)) == []
    one_label = [{"uid": "a", "group": "p", "template": "t", "query": "q", "label": 1},
                 {"uid": "b", "group": "p", "template": "t", "query": "r", "label": 1}]
    assert run(IMPLS[name], one_label, np.array([10, 10])) == []

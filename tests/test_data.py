"""Loader tests: label balance, leakage, subsample stratification, renaming."""
import re
from collections import Counter

import numpy as np
import pytest

from causal_probe import data
from causal_probe.config import C2C_PROBE_PER_PREMISE, C2C_PROBE_PREMISES, CLADDER_N


@pytest.fixture(scope="module")
def c2c_test():
    return data.load_c2c("test")


@pytest.fixture(scope="module")
def c2c_train():
    return data.load_c2c("train")


@pytest.fixture(scope="module")
def pool(c2c_train):
    return data.c2c_probe_pool(c2c_train)


@pytest.fixture(scope="module")
def cl():
    return data.cladder_sample()


def test_c2c_sizes_and_imbalance(c2c_test, c2c_train):
    assert len(c2c_test) == 1162 and len(c2c_train) == 205734
    bt, br = data.label_balance(c2c_test), data.label_balance(c2c_train)
    assert bt["n_pos"] == 180 and abs(bt["frac_pos"] - 0.155) < 0.001   # imbalanced!
    assert abs(br["frac_pos"] - 0.185) < 0.001


def test_c2c_no_leakage_train_test(c2c_test, c2c_train):
    train_inputs = {(i["context"], i["query"]) for i in c2c_train}
    assert not any((i["context"], i["query"]) in train_inputs for i in c2c_test)
    assert not ({i["group"] for i in c2c_test} & {i["group"] for i in c2c_train})  # premise level


def test_c2c_probe_pool_balanced_and_disjoint_from_test(pool, c2c_test):
    assert len(pool) == C2C_PROBE_PREMISES * C2C_PROBE_PER_PREMISE
    assert data.label_balance(pool)["frac_pos"] == 0.5
    per = Counter((i["group"], i["label"]) for i in pool)
    assert set(per.values()) == {C2C_PROBE_PER_PREMISE // 2}
    assert len({i["uid"] for i in pool}) == len(pool)
    assert not ({i["group"] for i in pool} & {i["group"] for i in c2c_test})


def test_subsample_is_deterministic(pool, c2c_train, cl):
    assert [i["uid"] for i in data.c2c_probe_pool(c2c_train)] == [i["uid"] for i in pool]
    assert [i["uid"] for i in data.cladder_sample()] == [i["uid"] for i in cl]


def test_cladder_full_balance_and_strata(cl):
    full = data.load_cladder()
    assert len(full) == 10112 and data.label_balance(full)["frac_pos"] == 0.5
    assert len(cl) == CLADDER_N and len({i["uid"] for i in cl}) == CLADDER_N
    assert abs(data.label_balance(cl)["frac_pos"] - 0.5) < 0.01
    f, s = Counter(i["query_type"] for i in full), Counter(i["query_type"] for i in cl)
    for qt in f:  # proportional allocation within +-1 item per (label, query_type) cell
        assert abs(s[qt] - f[qt] * CLADDER_N / len(full)) <= 2
    assert {i["rung"] for i in cl} == {1, 2, 3}
    # rung is a function of query type
    assert all(len({i["rung"] for i in full if i["query_type"] == qt}) == 1 for qt in f)


def test_cladder_prompt_split_is_lossless():
    import csv
    rows = list(csv.DictReader(open(data.DATA_RAW / "cladder" / "full_v1.5_default.csv")))
    for r, it in zip(rows, data.load_cladder()):
        assert it["context"] + " " + it["query"] == r["prompt"].strip()
        assert it["query"].endswith("?") and it["context"].endswith(".")


@pytest.mark.parametrize("tag", ["c2c_probe", "cladder"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_folds_have_no_group_leakage(tag, seed):
    items = data.get_dataset(tag)
    g = np.array([i["group"] for i in items])
    seen = np.zeros(len(items), int)
    for tr, te in data.grouped_folds(items, 5, seed):
        assert not (set(g[tr]) & set(g[te]))      # no premise / story on both sides
        seen[te] += 1
    assert (seen == 1).all()                      # every item held out exactly once


def test_rename_variables(c2c_test):
    for it in c2c_test[::37]:
        r = data.rename_variables(it)
        assert r["label"] == it["label"] and r["uid"] != it["uid"]
        assert not re.search(r"\b[A-F]\b", r["context"] + r["query"])
        # inverse map restores the original text exactly
        inv = {v: k for k, v in data.RENAME_MAP.items()}
        back = re.sub(r"\b([U-Z])\b", lambda m: inv[m.group(1)], r["context"] + "|" + r["query"])
        assert back == it["context"] + "|" + it["query"]


def test_debiased_pool_has_no_hypothesis_shortcut():
    from collections import Counter
    pool = data.get_dataset("c2c_debiased")
    cells = Counter((i["query"], i["label"]) for i in pool)
    strings = {q for q, _ in cells}
    assert all(cells[(q, 0)] == cells[(q, 1)] > 0 for q in strings)     # every string exactly 50/50
    assert len({i["uid"] for i in pool}) == len(pool)
    tpl = Counter((i["template"], i["label"]) for i in pool)
    assert all(tpl[(t, 0)] == tpl[(t, 1)] for t in {t for t, _ in tpl})  # hence every template too
    test_prems = {i["group"] for i in data.load_c2c("test")}
    assert not ({i["group"] for i in pool} & test_prems)
    assert [i["uid"] for i in data.get_dataset("c2c_debiased")] == [i["uid"] for i in pool]   # deterministic

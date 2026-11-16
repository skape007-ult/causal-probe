"""Part (b): the same tests on the owner's implementation and on the reference, and one against the other."""
import numpy as np
import pytest

from ftstudy import within_string, within_string_reference

IMPLS = {"owner": within_string, "reference": within_string_reference}


def call(name, *a):
    try:
        return IMPLS[name].within_string_auroc_length_removed(*a)
    except NotImplementedError:
        pytest.skip("owner's part (b) is not implemented yet")


def data(seed=0, n_strings=12, per=40, signal=0.8, length_effect=0.0):
    rng = np.random.default_rng(seed)
    strings = np.repeat([f"s{k}" for k in range(n_strings)], per)
    y = rng.integers(0, 2, len(strings))
    n = rng.integers(300, 900, len(strings)) - 200 * y           # valid items are shorter: the leak
    s = signal * y + rng.standard_normal(len(y)) + length_effect * np.log(n)
    return y, s, strings, n


@pytest.mark.parametrize("name", IMPLS)
def test_perfect_and_inverted(name):
    y, _, strings, n = data()
    assert call(name, y, y + 0.0, strings, np.full(len(y), 300))["mean_auroc"] == pytest.approx(1.0)
    assert call(name, y, -y + 0.0, strings, np.full(len(y), 300))["mean_auroc"] == pytest.approx(0.0)


@pytest.mark.parametrize("name", IMPLS)
def test_a_score_that_is_only_length_is_removed_entirely(name):
    """score = any quadratic in log(tokens) + noise  must give exactly what the noise alone gives."""
    y, _, strings, n = data(1)
    lx = np.log(n)
    e = np.random.default_rng(9).standard_normal(len(y))
    raw = call(name, y, -lx, strings, np.full(len(y), 300))["mean_auroc"]      # length NOT removed (constant token count given)
    assert raw > 0.7, "the test data must leak through length, or this test shows nothing"
    want = call(name, y, e, strings, n)["mean_auroc"]
    for quad in (-5 * lx, 3 - 2 * lx + 0.5 * lx ** 2):
        assert call(name, y, quad + e, strings, n)["mean_auroc"] == pytest.approx(want, abs=1e-9)
    assert abs(want - 0.5) < 0.1


@pytest.mark.parametrize("name", IMPLS)
def test_adding_a_per_string_function_of_length_changes_nothing(name):
    y, s, strings, n = data(2)
    lx = np.log(n)
    shift = np.array([hash(u) % 7 for u in strings]) * 1.0
    a = call(name, y, s, strings, n)["mean_auroc"]
    b = call(name, y, s + shift + (1 + shift) * lx - 0.3 * shift * lx ** 2, strings, n)["mean_auroc"]
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("name", IMPLS)
def test_one_class_strings_are_skipped_and_counted(name):
    y, s, strings, n = data(3)
    y = y.copy(); y[strings == "s0"] = 1
    r = call(name, y, s, strings, n)
    assert r["n_strings"] == 11 and "s0" not in r["per_string"] and len(r["per_string"]) == 11
    assert r["mean_auroc"] == pytest.approx(np.mean(list(r["per_string"].values())))


@pytest.mark.parametrize("name", IMPLS)
def test_ties_count_one_half(name):
    y = np.array([1, 0, 1, 0]); s = np.zeros(4)
    assert call(name, y, s, np.array(["a"] * 4), np.array([10, 20, 30, 40]))["mean_auroc"] == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(5))
def test_owner_equals_reference(seed):
    y, s, strings, n = data(seed, signal=0.4, length_effect=0.7)
    a = call("owner", y, s, strings, n); b = call("reference", y, s, strings, n)
    assert a["mean_auroc"] == pytest.approx(b["mean_auroc"], abs=1e-9)
    assert a["per_string"].keys() == b["per_string"].keys()
    assert all(a["per_string"][k] == pytest.approx(b["per_string"][k], abs=1e-9) for k in b["per_string"])

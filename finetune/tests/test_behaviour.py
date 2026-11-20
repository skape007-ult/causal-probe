"""The behaviour analysis on synthetic data only (no model score is opened): the blinding guard, the fast statistic
against the registered one, the outlier rule, Holm, the rungs, and a whole analysis with a planted and a null
effect."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "finetune"), str(ROOT / "finetune" / "scripts")]
from ftstudy import behaviour as BH, stat          # noqa: E402

TEN = [f"{c}_s{s}" for c in ("A", "B") for s in range(5)]


# ---------------------------------------------------------------- blinding
def test_blinding_refuses_partial_A_or_B():
    for models in (TEN[:4] + TEN[5:], TEN[:-1], ["A_s0"], ["B_s3", "C_s0"]):
        with pytest.raises(BH.Blinded):
            BH.check_blinding(models)


def test_blinding_allows_complete_set_and_non_AB_models():
    BH.check_blinding(TEN + ["A48_s0", "C_s1", "base"])
    BH.check_blinding(["base"]); BH.check_blinding(["C_s0", "Bswap_s1", "A48_s2"]); BH.check_blinding([])


def test_script_refuses_before_loading_anything(monkeypatch, capsys):
    import behaviour as script
    monkeypatch.setattr(script, "available_models", lambda: ["base"] + [f"{c}_s{s}" for c in ("A", "B") for s in range(4)])
    monkeypatch.setattr(script, "build_sets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded data while blinded")))
    monkeypatch.setattr(sys, "argv", ["behaviour.py"])
    assert script.main() == 3
    assert "A_s4" in capsys.readouterr().err


def test_load_scores_checks_blinding_first(monkeypatch):
    import behaviour as script
    monkeypatch.setattr(script.np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened a score file")))
    with pytest.raises(BH.Blinded):
        script.load_scores(["A_s0", "A_s1"], {})


# ---------------------------------------------------------------- toy data
def toy(n_str=6, per=48, seed=0):
    rng = np.random.default_rng(seed)
    q = np.repeat([f"s{i}" for i in range(n_str)], per)
    y = np.tile(np.r_[np.ones(per // 2), np.zeros(per // 2)], n_str).astype(int)
    rng.shuffle(y)
    n = rng.integers(150, 700, len(q)).astype(float)
    g = np.array([f"p{i // 4}" for i in rng.permutation(len(q))])
    adj = rng.integers(0, 2, len(q))
    return q, y, n, g, adj


def test_fast_statistic_equals_registered_one():
    q, y, n, g, adj = toy()
    rng = np.random.default_rng(1)
    for trial in range(5):
        s = rng.standard_normal(len(y)) + 0.7 * y + (np.round(rng.standard_normal(len(y))) if trial == 4 else 0)
        if trial >= 3:                                # many exact ties, as in real logit differences (float32)
            s = np.round(s, 1).astype(np.float32).astype(float); n = np.round(n, -2)
        idx = rng.integers(0, len(y), len(y)) if trial else np.arange(len(y))            # resamples have duplicates
        pl = BH.Plan(y[idx], q[idx], n[idx], adj[idx])
        assert abs(pl.S(s[idx]) - stat.S(y[idx], s[idx], q[idx], n[idx])) < 1e-9
        assert abs(pl.S_cells(s[idx]) - stat.S_cells(y[idx], s[idx], q[idx], adj[idx], n[idx])) < 1e-9
        M = np.column_stack([s, -s, s ** 3])[idx]
        assert np.allclose(pl.S(M), [pl.S(M[:, j]) for j in range(3)], atol=1e-12)
        assert np.allclose(pl.S_cells(M), [pl.S_cells(M[:, j]) for j in range(3)], atol=1e-12)


# ---------------------------------------------------------------- test assembly
def fake_boot(v, sd=0.01, n=2000, seed=0):
    return v + sd * np.random.default_rng(seed).standard_normal(n)


def test_outlier_rule_reports_both_versions():
    px = [0.70, 0.71, 0.69, 0.72, 0.45]            # the last seed failed to train
    bx = [fake_boot(v, seed=i) for i, v in enumerate(px)]
    py = [0.60, 0.61, 0.59, 0.60, 0.62]
    by = [fake_boot(v, seed=10 + i) for i, v in enumerate(py)]
    t = BH.run_test(px, bx, py, by)
    assert t["outlying_seeds_x"] == [4] and "without_outliers" in t
    assert t["without_outliers"]["n_x"] == 4


def test_holm_claims_only_if_both_versions_survive():
    tests = {k: {"p_two_sided": 0.5, "D": 0.01, "seed_sd_x": 0.01, "seed_sd_y": None} for k in BH.FAMILY}
    tests["E2a"].update(p_two_sided=1e-6, D=0.2)
    tests["E2b"].update(p_two_sided=1e-6, D=0.2, without_outliers={"p_two_sided": 0.2})
    BH.holm_with_outlier_rule(tests)
    assert tests["E2a"]["claimed"] and tests["E2a"]["direction"] == "above"
    assert not tests["E2b"]["claimed"] and tests["E2b"]["holm_p"] < 0.05


def test_rungs():
    up, no = {"claimed": True, "D": 0.2}, {"claimed": False, "D": 0.2}
    assert BH.rung(no, no, no) == "L0" and BH.rung(no, up, no).startswith("L0 (E5b")
    assert BH.rung(up, no, no) == "L1" and BH.rung(up, up, no) == "L2" and BH.rung(up, up, up) == "L3"
    assert BH.rung({"claimed": True, "D": -0.2}, no, no) == "L0"            # claimed BELOW 0.5 is not a rung


# ---------------------------------------------------------------- the whole analysis on toy sets
def toy_world(planted: bool):
    q, y, n, g, adj = toy(n_str=5, per=40, seed=3)
    sets = {}
    for k in ("pool_lb", "id48", "id", "id48_renamed", "id48_renamed_keep_order"):
        sets[k] = BH.EvalSet(k, y, q, n, g, adj if k == "pool_lb" else None)
    boot = stat.boot_indices(g, stat.N_BOOT, seed=0)
    for e in sets.values():
        e.boot = boot
    rng = np.random.default_rng(5)
    beta = {"A": 0.3, "B": 1.5, "A48": 0.6, "C": 0.0, "Bswap": 0.0} if planted else {c: 0.6 for c in BH.REGISTERED_SEEDS}
    scores = {}
    for c, k_ in BH.REGISTERED_SEEDS.items():
        for sd in range(k_):
            b = beta[c] + 0.02 * rng.standard_normal()
            scores[f"{c}_s{sd}"] = {k: b * (2 * y - 1) + rng.standard_normal(len(y)) for k in sets}
    ref = lambda b: b * (2 * y - 1) + rng.standard_normal(len(y))              # noqa: E731
    refs = {"mem": {c: {"pool_lb": ref(0.4)} for c in ("A", "B", "A48")},
            "text_best": {c: ("lookup", ref(0.3)) for c in ("A", "B")},
            "mem_registered_S": {c: {"pool_lb": 0.7, "id48": 0.7} for c in ("A", "B", "A48")}}
    return sets, scores, refs


@pytest.mark.parametrize("planted", [True, False])
def test_whole_analysis(planted):
    sets, scores, refs = toy_world(planted)
    out = BH.analyse(sets, scores, refs)
    T = out["tests"]
    assert set(BH.FAMILY) <= set(T) and all("holm_p" in T[k] for k in BH.FAMILY)
    if planted:
        assert T["E2a"]["claimed"] and T["E2a"]["D"] > 0.2
        assert T["E5b_B"]["claimed"] and out["descriptive"]["rung_B"] in ("L2", "L3")
    else:
        assert not T["E2a"]["claimed"] and abs(T["E2a"]["D"]) < 0.1
    assert out["seeds"]["A"] == [f"A_s{i}" for i in range(5)]


def test_analysis_refuses_without_all_ten():
    sets, scores, refs = toy_world(True)
    del scores["B_s4"]
    with pytest.raises(BH.Blinded):
        BH.analyse(sets, scores, refs)


def test_holm_counts_missing_tests_as_p_one():
    tests = {k: {"p_two_sided": 0.5, "D": 0.01, "seed_sd_x": 0.01, "seed_sd_y": None} for k in BH.FAMILY if k != "E2b"}
    tests["E2a"]["p_two_sided"] = 0.004                        # 0.004 * 11 = 0.044 < 0.05; with 10 tests 0.04
    BH.holm_with_outlier_rule(tests)
    assert abs(tests["E2a"]["holm_p"] - 0.044) < 1e-12 and tests["_not_run"] == ["E2b"]

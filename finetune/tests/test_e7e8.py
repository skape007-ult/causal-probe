"""E7/E8 pieces (PREREG Amendment 3), on synthetic input."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "finetune")]
from ftstudy import e7e8 as E  # noqa: E402


def test_site_layer_returns_planted_layer_with_nan_layers():
    L = 29
    S = np.full((5, L), 0.5)
    S[:, 10:] = 0.5 + 0.2 * np.minimum(1, (np.arange(10, L) - 9) / 4)     # rises to 0.70 at layer 13
    S[:, 3] = np.nan
    S[2, 5] = np.nan
    assert E.site_layer(S, list(range(L))) == 13          # 0.9 x 0.2 = 0.18 first reached at layer 13 (0.2 >= 0.18; 12: 0.15)


def test_site_layer_none_without_signal():
    assert E.site_layer(np.full((5, 29), 0.51), list(range(29))) is None
    assert E.site_layer(np.full((5, 29), np.nan), list(range(29))) is None


def test_projection_removes_answer_and_text():
    rng = np.random.default_rng(0)
    n, d = 600, 16
    y = rng.integers(0, 2, n); own = rng.integers(0, 2, n); txt = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)); X[:, 0] += 2 * y; X[:, 1] += 2 * own; X[:, 2] += 2 * txt
    V, sig, diag = E.directions(X, y, own, txt, np.array(["s"] * n))
    assert abs(V["g"] @ V["d_ans"]) < 1e-8
    assert abs(V["g"] @ (V["d_text"] - (V["d_text"] @ V["d_ans"]) * V["d_ans"])) < 1e-8
    assert abs(V["g"][0]) > 0.9                             # the planted gold axis survives
    assert len(V["controls"]) == 20 and np.allclose(np.linalg.norm(V["controls"], axis=1), 1)
    assert diag["median_abs_cos_g_control"] < 0.6


def test_shuffle_keeps_single_class_cells():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 1])
    cells = np.array(["a", "a", "a", "b", "b", "b", "c", "c"])
    for s in range(5):
        ys = E.shuffle_within_cells(y, cells, s)
        assert (ys[:3] == 1).all()
        for c in "abc":
            assert ys[cells == c].sum() == y[cells == c].sum()


def test_take_premises_truncates_at_boundary():
    groups = np.repeat(np.arange(10), 7)
    idx = E.take_premises(groups, np.arange(70), 30)
    assert len(idx) == 28 and len(np.unique(groups[idx])) == 4
    ex = E.take_premises(groups, np.arange(70), 30, exclude=set(groups[idx]))
    assert not set(groups[ex]) & set(groups[idx])


def test_donor_rules():
    y = np.array([0, 1, 0, 1, 1]); s = np.array(["x"] * 5); adj = np.array([0, 0, 0, 0, 1])
    grp = np.array([0, 1, 2, 0, 3]); n = np.array([100, 105, 108, 100, 100])
    D = E.donors([0], np.arange(5), y, s, adj, grp, n)
    assert D[0][0] == 1 and D[0][1] == 2                    # 3 is the same premise, 4 another adjacency


def test_slope_and_e8_signs():
    LD = np.array([[-2, -1, 0, 1, 2], [4, 2, 0, -2, -4]], float)
    assert np.allclose(E.slopes(LD), [1, -2])
    e = E.e8_items(np.array([0.0]), np.array([1.0]), np.array([0.2]), np.array([1]), 1.0)
    assert np.isclose(e[0], 0.8)
    e = E.e8_items(np.array([0.0]), np.array([-1.0]), np.array([-0.2]), np.array([0]), 1.0)
    assert np.isclose(e[0], 0.8)


def test_seed_sd_bound_and_power():
    b = E.seed_sd_bound({"A": [0.0, 0.02], "B": [0.1, 0.12]})
    assert b["df"] == 2 and np.isclose(b["upper80"], b["pooled_sd"] * 2.117, rtol=1e-3)
    assert b["used"] >= E.SEED_SD_FLOOR
    assert E.power(0.10, 0.02, 0.01) > 0.9
    assert E.power(0.0, 0.02, 0.01) < 0.03                  # size at alpha 0.05/4
    r = E.choose_n_items({"E7(A)": 0.05, "E7(B)": 0.5}, 2000, 7.9, 25, 0.03)
    assert "E7(B)" in r["dropped"] and r["N_items"] is not None

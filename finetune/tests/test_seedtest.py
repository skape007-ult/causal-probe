"""The registered test: arithmetic, and that its size is what PREREG.md says (a small re-simulation)."""
import numpy as np
import pytest
from scipy import stats

from ftstudy import seedtest


def test_reduces_to_welch_when_premise_variance_is_zero():
    x, y = [0.61, 0.64, 0.60, 0.66, 0.63], [0.55, 0.52, 0.58]
    r = seedtest.test(x, y, boot_D=np.zeros(300))
    w = stats.ttest_ind(x, y, equal_var=False)
    assert r["t"] == pytest.approx(w.statistic) and r["p_two_sided"] == pytest.approx(w.pvalue) and r["df"] == pytest.approx(w.df)


def test_one_sample_and_premise_term():
    x = [0.52, 0.49, 0.51]
    r = seedtest.test(x, constant=0.5, boot_D=np.zeros(300))
    assert r["p_two_sided"] == pytest.approx(stats.ttest_1samp(x, 0.5).pvalue)
    b = np.random.default_rng(0).normal(0, 0.02, 2000)
    r2 = seedtest.test(x, constant=0.5, boot_D=b)
    assert r2["se"] > r["se"] and r2["se"] == pytest.approx(np.sqrt(np.var(x, ddof=1) / 3 + b.var(ddof=1)))
    assert r2["lo"] < r2["D"] < r2["hi"]


def test_resamples_are_required():
    with pytest.raises(ValueError):
        seedtest.test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])


@pytest.mark.parametrize("nx,ny,sd,tau", [(5, 5, 0.02, 0.01), (5, 3, 0.03, 0.005), (3, 0, 0.02, 0.01), (5, 5, 0.005, 0.03)])
def test_size_is_at_most_nominal(nx, ny, sd, tau):
    rng = np.random.default_rng(1); reps = 40_000
    x = sd * rng.standard_normal((reps, nx)); shift = tau * rng.standard_normal(reps)
    if ny:
        y = sd * rng.standard_normal((reps, ny))
        t, df = seedtest.t_and_df(x.mean(1) - y.mean(1) + shift, x.var(1, ddof=1), nx, y.var(1, ddof=1), ny, tau ** 2)
    else:
        t, df = seedtest.t_and_df(x.mean(1) + shift, x.var(1, ddof=1), nx, None, 0, tau ** 2)
    assert np.mean(seedtest.p_two_sided(t, df) < 0.05) < 0.056

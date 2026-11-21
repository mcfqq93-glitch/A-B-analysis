"""Cross-check the hand-written formulas against scipy / statsmodels."""
import sys
from pathlib import Path

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest import stats as S  # noqa: E402

X_C, N_C, X_T, N_T = 17489, 145274, 17264, 145310


def test_ztest_matches_statsmodels():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T)
    z, p = proportions_ztest([X_T, X_C], [N_T, N_C])
    assert r.z == pytest.approx(z)
    assert r.p_value == pytest.approx(p)


def test_one_sided():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T, alternative="larger")
    _, p = proportions_ztest([X_T, X_C], [N_T, N_C], alternative="larger")
    assert r.p_value == pytest.approx(p)


def test_chi2_equals_z_squared():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T)
    chi2, p, _ = S.chi2_independence(X_C, N_C, X_T, N_T)
    assert chi2 == pytest.approx(r.z ** 2)
    assert p == pytest.approx(r.p_value)


def test_sample_size_close_to_statsmodels():
    # statsmodels uses Cohen's h (arcsine transform); results agree within ~1%
    n = S.sample_size_per_group(0.12, 0.01)
    h = proportion_effectsize(0.13, 0.12)
    n_sm = NormalIndPower().solve_power(h, alpha=0.05, power=0.8)
    assert abs(n - n_sm) / n_sm < 0.01


def test_power_at_required_n_is_target():
    n = S.sample_size_per_group(0.12, 0.01, power=0.8)
    assert S.power_two_proportions(0.12, 0.01, n) == pytest.approx(0.8, abs=0.002)


def test_mde_inverts_power():
    mde = S.minimum_detectable_effect(0.12, 145_000)
    assert S.power_two_proportions(0.12, mde, 145_000) == pytest.approx(0.8, abs=1e-6)


def test_srm():
    assert not S.srm_check([145274, 145310]).srm_detected
    assert S.srm_check([50000, 51500]).srm_detected


@pytest.mark.parametrize("method,sm", [("bonferroni", "bonferroni"),
                                       ("holm", "holm"), ("bh", "fdr_bh")])
def test_multiple_testing(method, sm):
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.5])
    assert np.allclose(S.adjust_pvalues(p, method), multipletests(p, method=sm)[1])


def test_bootstrap_binary_close_to_normal_ci():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T)
    b = S.bootstrap_diff_binary(X_C, N_C, X_T, N_T, n_boot=50_000)
    assert b["ci"][0] == pytest.approx(r.ci_abs[0], abs=2e-4)
    assert b["ci"][1] == pytest.approx(r.ci_abs[1], abs=2e-4)


def test_cuped_reduces_variance_and_keeps_mean_diff():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 3, 20_000)
    t = rng.integers(0, 2, 20_000)
    y = 0.8 * x + rng.normal(0, 2, 20_000) + 0.3 * t
    y_adj, theta = S.cuped_adjust(y, x)
    assert theta == pytest.approx(0.8, abs=0.03)
    assert y_adj.var() < 0.5 * y.var()
    raw = y[t == 1].mean() - y[t == 0].mean()
    adj = y_adj[t == 1].mean() - y_adj[t == 0].mean()
    assert adj == pytest.approx(raw, abs=0.1)


# --- cross-checks for the remaining hand-written routines --------------------
from scipy import stats as sps  # noqa: E402
from statsmodels.stats.proportion import confint_proportions_2indep  # noqa: E402


def test_abs_ci_matches_statsmodels_wald():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T)
    lo, hi = confint_proportions_2indep(X_T, N_T, X_C, N_C, method="wald", compare="diff")
    assert r.ci_abs == pytest.approx((lo, hi))


def test_rel_ci_matches_statsmodels_log_ratio():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T)
    lo, hi = confint_proportions_2indep(X_T, N_T, X_C, N_C, method="log", compare="ratio")
    assert r.ci_rel == pytest.approx((lo - 1, hi - 1))


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.exponential(5, 3_000), rng.exponential(5.4, 2_500)
    r = S.welch_ttest(a, b)
    res = sps.ttest_ind(b, a, equal_var=False)
    assert r["t"] == pytest.approx(res.statistic)
    assert r["p_value"] == pytest.approx(res.pvalue)
    ci = res.confidence_interval(0.95)
    assert r["ci"] == pytest.approx((ci.low, ci.high))


def test_permutation_close_to_ztest():
    r = S.two_proportion_ztest(X_C, N_C, X_T, N_T)
    perm = S.permutation_test_binary(X_C, N_C, X_T, N_T, n_perm=50_000)
    assert perm["p_value"] == pytest.approx(r.p_value, abs=0.01)


def test_bootstrap_means_close_to_welch_ci():
    rng = np.random.default_rng(2)
    a, b = rng.normal(10, 3, 4_000), rng.normal(10.2, 3, 4_000)
    boot = S.bootstrap_diff_in_means(a, b, n_boot=4_000)
    ci = S.welch_ttest(a, b)["ci"]
    assert boot["ci"][0] == pytest.approx(ci[0], abs=0.02)
    assert boot["ci"][1] == pytest.approx(ci[1], abs=0.02)


@pytest.mark.parametrize("p,n", [(1.0, 1_000), (0.0, 1_000), (0.995, 1_000)])
def test_mde_returns_nan_when_unreachable(p, n):
    assert np.isnan(S.minimum_detectable_effect(p, n))

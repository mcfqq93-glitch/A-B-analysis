"""Statistical toolkit for A/B test analysis.

Every function is deliberately small and written from first principles
(closed-form formulas) so that it can be read, explained in an interview,
and cross-checked against scipy / statsmodels in ``tests/``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "srm_check", "two_proportion_ztest", "chi2_independence", "sample_size_per_group",
    "power_two_proportions", "minimum_detectable_effect", "bootstrap_diff_in_means",
    "bootstrap_diff_binary", "permutation_test_binary", "cuped_adjust", "welch_ttest",
    "adjust_pvalues",
]


# ---------------------------------------------------------------------------
# 1. Sample Ratio Mismatch (SRM)
# ---------------------------------------------------------------------------
@dataclass
class SRMResult:
    observed: tuple
    expected: tuple
    chi2: float
    p_value: float
    srm_detected: bool

    def to_dict(self):
        return asdict(self)


def srm_check(counts: Sequence[int], expected_ratios: Sequence[float] | None = None,
              threshold: float = 0.001) -> SRMResult:
    """Chi-square goodness-of-fit test for Sample Ratio Mismatch.

    A very small threshold (0.001) is the industry convention because SRM is a
    data-quality alarm: with large samples even tiny imbalances are "significant",
    and we only want to fire when the split is clearly broken.
    """
    counts = np.asarray(counts, dtype=float)
    if expected_ratios is None:
        expected_ratios = np.repeat(1 / len(counts), len(counts))
    expected_ratios = np.asarray(expected_ratios, dtype=float)
    expected_ratios = expected_ratios / expected_ratios.sum()
    expected = counts.sum() * expected_ratios
    chi2, p = stats.chisquare(counts, expected)
    return SRMResult(tuple(int(v) for v in counts), tuple(round(float(v), 1) for v in expected),
                     float(chi2), float(p), bool(p < threshold))


# ---------------------------------------------------------------------------
# 2. Two-proportion z-test + confidence intervals
# ---------------------------------------------------------------------------
@dataclass
class ProportionTestResult:
    p_control: float
    p_treatment: float
    abs_diff: float          # p_t - p_c
    rel_lift: float          # (p_t - p_c) / p_c
    z: float
    p_value: float
    ci_abs: tuple            # CI for abs_diff (unpooled Wald)
    ci_rel: tuple            # CI for rel_lift (delta method)
    alpha: float

    def to_dict(self):
        return asdict(self)


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int,
                         alpha: float = 0.05,
                         alternative: str = "two-sided") -> ProportionTestResult:
    """Two-proportion z-test.

    * The **test statistic** uses the *pooled* proportion, because under H0
      (p_c == p_t) both groups share one conversion rate.
    * The **confidence interval** uses the *unpooled* standard error, because
      we no longer assume H0 when estimating how big the difference is.
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = diff / se_pool

    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":      # H1: p_t > p_c
        p_value = stats.norm.sf(z)
    elif alternative == "smaller":     # H1: p_t < p_c
        p_value = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger' or 'smaller'")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_unpool = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    ci_abs = (diff - z_crit * se_unpool, diff + z_crit * se_unpool)

    # Delta method for the ratio p_t / p_c - 1:
    # Var(log(p_t/p_c)) ≈ (1-p_t)/(n_t p_t) + (1-p_c)/(n_c p_c)
    se_log = np.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
    log_ratio = np.log(p_t / p_c)
    ci_rel = (np.exp(log_ratio - z_crit * se_log) - 1,
              np.exp(log_ratio + z_crit * se_log) - 1)

    return ProportionTestResult(float(p_c), float(p_t), float(diff), float(diff / p_c), float(z), float(p_value),
                                tuple(map(float, ci_abs)), tuple(map(float, ci_rel)),
                                alpha)


def chi2_independence(x_c: int, n_c: int, x_t: int, n_t: int):
    """2x2 chi-square test of independence (no continuity correction).

    For a 2x2 table, chi2 == z**2 and the two-sided p-values are identical.
    """
    table = np.array([[x_c, n_c - x_c], [x_t, n_t - x_t]])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof)


# ---------------------------------------------------------------------------
# 3. Power analysis / sample size
# ---------------------------------------------------------------------------
def sample_size_per_group(p_baseline: float, mde_abs: float, alpha: float = 0.05,
                          power: float = 0.8, ratio: float = 1.0,
                          two_sided: bool = True) -> int:
    """Required sample size for the *control* group (treatment = ratio * n).

    n_c = ( z_a * sqrt(p̄q̄(1 + 1/k)) + z_b * sqrt(p1q1 + p2q2/k) )^2 / δ^2
    where k = ratio = n_t / n_c, δ = mde_abs.
    """
    p1 = p_baseline
    p2 = p_baseline + mde_abs
    if not 0 < p2 < 1:
        raise ValueError("baseline + MDE must lie in (0, 1)")
    k = ratio
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + k * p2) / (1 + k)
    num = (z_a * np.sqrt(p_bar * (1 - p_bar) * (1 + 1 / k))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / k)) ** 2
    return int(np.ceil(num / mde_abs ** 2))


def power_two_proportions(p_baseline: float, mde_abs: float, n_c: int,
                          alpha: float = 0.05, ratio: float = 1.0) -> float:
    """Power of a two-sided two-proportion z-test for a given sample size."""
    p1, p2 = p_baseline, p_baseline + mde_abs
    n_t = n_c * ratio
    z_a = stats.norm.ppf(1 - alpha / 2)
    p_bar = (p1 * n_c + p2 * n_t) / (n_c + n_t)
    se0 = np.sqrt(p_bar * (1 - p_bar) * (1 / n_c + 1 / n_t))   # SE under H0
    se1 = np.sqrt(p1 * (1 - p1) / n_c + p2 * (1 - p2) / n_t)  # SE under H1
    d = abs(mde_abs)
    return float(stats.norm.cdf((d - z_a * se0) / se1)
                 + stats.norm.cdf((-d - z_a * se0) / se1))


def minimum_detectable_effect(p_baseline: float, n_c: int, alpha: float = 0.05,
                              power: float = 0.8, ratio: float = 1.0) -> float:
    """Smallest absolute lift detectable with the given sample (solved numerically)."""
    from scipy.optimize import brentq
    f = lambda d: power_two_proportions(p_baseline, d, n_c, alpha, ratio) - power
    return float(brentq(f, 1e-6, 1 - p_baseline - 1e-6))


# ---------------------------------------------------------------------------
# 4. Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_diff_in_means(y_c: np.ndarray, y_t: np.ndarray, n_boot: int = 10_000,
                            alpha: float = 0.05, seed: int = 42,
                            chunk: int = 500) -> dict:
    """Percentile bootstrap for mean(y_t) - mean(y_c).

    Resamples each group independently with replacement (groups are
    independent by randomisation). Done in chunks to keep memory bounded.
    """
    rng = np.random.default_rng(seed)
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    diffs = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        mc = y_c[rng.integers(0, len(y_c), (b, len(y_c)))].mean(axis=1)
        mt = y_t[rng.integers(0, len(y_t), (b, len(y_t)))].mean(axis=1)
        diffs[start:start + b] = mt - mc
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(y_t.mean() - y_c.mean()), "ci": (float(lo), float(hi)),
            "se": float(diffs.std(ddof=1)), "samples": diffs}


def bootstrap_diff_binary(x_c: int, n_c: int, x_t: int, n_t: int, n_boot: int = 100_000,
                          alpha: float = 0.05, seed: int = 42) -> dict:
    """Fast bootstrap for 0/1 outcomes.

    Resampling n Bernoulli values with replacement and summing them is
    exactly a Binomial(n, p_hat) draw, so we can skip building huge arrays.
    """
    rng = np.random.default_rng(seed)
    pc = rng.binomial(n_c, x_c / n_c, n_boot) / n_c
    pt = rng.binomial(n_t, x_t / n_t, n_boot) / n_t
    diffs = pt - pc
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": x_t / n_t - x_c / n_c, "ci": (float(lo), float(hi)),
            "se": float(diffs.std(ddof=1)), "samples": diffs}


def permutation_test_binary(x_c: int, n_c: int, x_t: int, n_t: int,
                            n_perm: int = 20_000, seed: int = 42) -> dict:
    """Permutation test for a difference in proportions.

    Under H0 the group labels are exchangeable: shuffling labels among the
    pooled users and re-counting conversions in the 'treatment' slots is a
    hypergeometric draw.
    """
    rng = np.random.default_rng(seed)
    total_conv, N = x_c + x_t, n_c + n_t
    xt_perm = rng.hypergeometric(total_conv, N - total_conv, n_t, n_perm)
    xc_perm = total_conv - xt_perm
    diffs = xt_perm / n_t - xc_perm / n_c
    observed = x_t / n_t - x_c / n_c
    p = (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n_perm + 1)
    return {"observed": observed, "p_value": float(p), "samples": diffs}


# ---------------------------------------------------------------------------
# 5. CUPED
# ---------------------------------------------------------------------------
def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: Y_adj = Y - θ (X - mean(X)),  θ = Cov(X, Y) / Var(X).

    X must be a *pre-experiment* covariate (unaffected by treatment). θ is
    estimated on the pooled data of both groups. The adjusted metric has the
    same expectation difference between groups but variance multiplied by
    (1 - ρ²), where ρ = Corr(X, Y).
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def welch_ttest(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch t-test + CI for mean(y_t) - mean(y_c) (continuous metrics)."""
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    diff = y_t.mean() - y_c.mean()
    vc, vt = y_c.var(ddof=1) / len(y_c), y_t.var(ddof=1) / len(y_t)
    se = np.sqrt(vc + vt)
    df = (vc + vt) ** 2 / (vc ** 2 / (len(y_c) - 1) + vt ** 2 / (len(y_t) - 1))
    t = diff / se
    p = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return {"diff": float(diff), "se": float(se), "t": float(t), "df": float(df),
            "p_value": float(p), "ci": (float(diff - tc * se), float(diff + tc * se))}


# ---------------------------------------------------------------------------
# 6. Multiple comparisons
# ---------------------------------------------------------------------------
def adjust_pvalues(p_values: Sequence[float], method: str = "holm") -> np.ndarray:
    """Bonferroni / Holm (control FWER) and Benjamini-Hochberg (control FDR)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    if method == "bonferroni":
        adj = np.minimum(p * m, 1)
        return adj
    if method == "holm":
        adj_sorted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method in ("bh", "fdr_bh"):
        adj_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
    else:
        raise ValueError("method must be 'bonferroni', 'holm' or 'bh'")
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1)
    return adj

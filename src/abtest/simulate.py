"""Simulated experiments used for teaching (CUPED, peeking, A/A tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_revenue_experiment(n_users: int = 60_000, true_lift: float = 0.03,
                                seed: int = 7) -> pd.DataFrame:
    """Revenue experiment with a pre-period covariate (for the CUPED demo).

    The public Udacity dataset has no pre-experiment metric, so CUPED cannot be
    applied to it honestly. We simulate a realistic scenario instead:

    * ``revenue``    : revenue per user during the experiment (skewed, many zeros)
    * ``pre_revenue``: the same user's revenue in the equally long period *before*
      the experiment (cannot be affected by the treatment)
    * true effect    : ``true_lift`` relative lift in revenue for treatment
    """
    rng = np.random.default_rng(seed)
    # latent spending propensity drives both periods -> pre/post correlation
    propensity = rng.lognormal(mean=2.0, sigma=0.9, size=n_users)
    active_pre = rng.random(n_users) < 0.55
    active_post = rng.random(n_users) < 0.15 + 0.75 * active_pre  # habit persistence
    group = rng.choice(["control", "treatment"], size=n_users)
    pre = active_pre * propensity * rng.gamma(10, 0.1, n_users)
    lift = np.where(group == "treatment", 1 + true_lift, 1.0)
    post = active_post * propensity * rng.gamma(10, 0.1, n_users) * lift
    return pd.DataFrame({"user_id": np.arange(1, n_users + 1), "group": group,
                         "pre_revenue": pre.round(2), "revenue": post.round(2)})


def simulate_peeking(n_experiments: int = 2_000, n_per_look: int = 1_000, n_looks: int = 20,
                     p: float = 0.12, alpha: float = 0.05, seed: int = 0) -> dict:
    """A/A experiments (no true effect) analysed after every batch of users.

    Returns the false-positive rate when (a) we only test once at the end and
    (b) we stop as soon as any interim look has p < alpha ("peeking").
    """
    from scipy import stats
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n_per_look, p, (n_experiments, n_looks)).cumsum(axis=1)
    conv_b = rng.binomial(n_per_look, p, (n_experiments, n_looks)).cumsum(axis=1)
    n = n_per_look * np.arange(1, n_looks + 1)
    pa, pb = conv_a / n, conv_b / n
    pool = (conv_a + conv_b) / (2 * n)
    z = (pb - pa) / np.sqrt(pool * (1 - pool) * 2 / n)
    pvals = 2 * stats.norm.sf(np.abs(z))
    sig = pvals < alpha
    return {"fpr_final_look": float(sig[:, -1].mean()),
            "fpr_any_look": float(sig.any(axis=1).mean()),
            "fpr_by_n_looks": sig.cumsum(axis=1).astype(bool).mean(axis=0),
            "pvalue_paths": pvals[:20]}

"""Finite-difference Dupire local volatility approximation from total variance."""

import numpy as np


def dupire_local_vol(k_grid, t_grid, total_variance, *, min_denominator=1e-8):
    """Compute sigma_loc on an (expiry, log-moneyness) grid from w(k,T).

    Uses the Gatheral total-variance formula. Returns NaN where numerical inputs
    are invalid or the Dupire denominator is non-positive.
    """
    k = np.asarray(k_grid, float)
    t = np.asarray(t_grid, float)
    w = np.asarray(total_variance, float)
    if w.shape != (len(t), len(k)) or len(t) < 2 or len(k) < 3:
        raise ValueError("expected w with shape (len(t_grid), len(k_grid)) and >=2x3 grid")
    wk = np.gradient(w, k, axis=1)
    wkk = np.gradient(wk, k, axis=1)
    wt = np.gradient(w, t, axis=0)
    denom = (
        (1 - k[None, :] * wk / (2 * np.maximum(w, 1e-12))) ** 2
        - (wk**2 / 4) * (1 / np.maximum(w, 1e-12) + 0.25)
        + wkk / 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = wt / denom
    return np.sqrt(np.where((denom > min_denominator) & (variance >= 0), variance, np.nan))

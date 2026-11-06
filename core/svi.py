"""Raw SVI and SSVI total-variance parameterizations and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def variance(self, k):
        """Evaluate raw SVI total variance at log-moneyness k."""
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma))


def fit_svi(k, total_variance) -> SVIParams:
    """Fit raw SVI by bounded nonlinear least squares; bounds enforce basic parameter sanity."""
    k, w = np.asarray(k, float), np.asarray(total_variance, float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if len(k) < 5:
        raise ValueError("at least five valid smile observations are needed")
    scale = max(float(np.median(w)), 1e-4)
    x0 = [max(scale * 0.35, 1e-5), max(scale * 0.8, 1e-4), -0.2, float(np.median(k)), 0.15]
    lo = [0.0, 1e-7, -0.999, float(k.min() - 2), 1e-5]
    hi = [max(2.0, float(w.max() * 20)), 10.0, 0.999, float(k.max() + 2), 5.0]

    def residual(p):
        a, b, rho, m, sigma = p
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return (model - w) / max(scale, 0.01)

    result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=30000, loss="soft_l1")
    if not result.success and not np.isfinite(result.cost):
        raise RuntimeError(f"SVI fit failed: {result.message}")
    return SVIParams(*map(float, result.x))


def ssvi_variance(k, theta, rho, eta, gamma):
    """Gatheral–Jacquier SSVI total variance; theta is ATM total variance."""
    if theta <= 0 or eta <= 0 or not (-1 < rho < 1) or not (0 < gamma <= 1):
        raise ValueError("invalid SSVI parameters")
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
    x = phi * np.asarray(k, dtype=float)
    return theta / 2 * (1 + rho * x + np.sqrt((x + rho) ** 2 + 1 - rho**2))


def fit_ssvi(expiries, k_by_expiry, w_by_expiry):
    """Jointly fit SSVI shape parameters and monotone ATM variance term structure."""
    t = np.asarray(expiries, float)
    if np.any(t <= 0) or len(t) < 2:
        raise ValueError("SSVI needs at least two positive expiries")
    data = [(np.asarray(k, float), np.asarray(w, float)) for k, w in zip(k_by_expiry, w_by_expiry)]
    theta0 = np.maximum.accumulate([max(float(np.median(w)), 1e-4) for _, w in data])
    n = len(t)
    x0 = np.r_[theta0, 0.0, 0.8, 0.5]
    bounds = [(1e-6, 5.0)] * n + [(-0.95, 0.95), (0.01, 3.0), (0.05, 1.0)]

    def objective(x):
        theta = np.maximum.accumulate(x[:n])
        rho, eta, gamma = x[n:]
        err = []
        for i, (k, w) in enumerate(data):
            err.extend((ssvi_variance(k, theta[i], rho, eta, gamma) - w).tolist())
        return float(np.mean(np.square(err)))

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 5000})
    return {
        "theta": np.maximum.accumulate(res.x[:n]).tolist(),
        "rho": float(res.x[n]),
        "eta": float(res.x[n + 1]),
        "gamma": float(res.x[n + 2]),
        "rmse": float(np.sqrt(res.fun)),
        "success": bool(res.success),
    }


def arbitrage_checks(params_by_expiry, expiries, grid=None, tolerance=1e-7):
    """Approximate calendar (monotone w) and butterfly (g(k)>=0) checks on a k-grid.

    The butterfly diagnostic uses g=(1-k*w'/2w)^2 - (w'^2/4)(1/w+1/4) + w''/2.
    It is a numerical diagnostic, not a formal proof outside the tested grid.
    """
    grid = np.asarray(grid if grid is not None else np.linspace(-1.5, 1.5, 301), float)
    expiries = np.asarray(expiries, float)
    surfaces = np.array([p.variance(grid) for p in params_by_expiry])
    calendar = []
    for i in range(len(expiries) - 1):
        bad = np.where(surfaces[i + 1] < surfaces[i] - tolerance)[0]
        if len(bad):
            calendar.append(
                {
                    "earlier_expiry": float(expiries[i]),
                    "later_expiry": float(expiries[i + 1]),
                    "count": len(bad),
                    "k_min": float(grid[bad].min()),
                    "k_max": float(grid[bad].max()),
                }
            )
    butterfly = []
    for i, (p, t) in enumerate(zip(params_by_expiry, expiries)):
        w = np.maximum(p.variance(grid), 1e-12)
        d1 = np.gradient(w, grid)
        d2 = np.gradient(d1, grid)
        g = (1 - grid * d1 / (2 * w)) ** 2 - d1**2 / 4 * (1 / w + 0.25) + d2 / 2
        bad = np.where(g < -tolerance)[0]
        if len(bad):
            butterfly.append(
                {
                    "expiry": float(t),
                    "count": len(bad),
                    "minimum_density_factor": float(g.min()),
                    "k_min": float(grid[bad].min()),
                    "k_max": float(grid[bad].max()),
                }
            )
    return {
        "calendar_violations": calendar,
        "butterfly_violations": butterfly,
        "calendar_ok": not calendar,
        "butterfly_ok": not butterfly,
    }

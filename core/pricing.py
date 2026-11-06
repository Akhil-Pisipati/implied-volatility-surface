"""Black–Scholes pricing, Greeks, and bounded implied-volatility inversion."""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.special import ndtr


def _d1d2(spot: float, strike: float, time: float, rate: float, vol: float, dividend: float = 0.0):
    if min(spot, strike, time, vol) <= 0:
        raise ValueError("spot, strike, time, and volatility must be positive")
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol**2) * time) / (
        vol * math.sqrt(time)
    )
    return d1, d1 - vol * math.sqrt(time)


def price(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    vol: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> float:
    """Price a European option with continuous dividend yield."""
    d1, d2 = _d1d2(spot, strike, time, rate, vol, dividend)
    dq, dr = math.exp(-dividend * time), math.exp(-rate * time)
    if kind.lower() == "call":
        return spot * dq * ndtr(d1) - strike * dr * ndtr(d2)
    if kind.lower() == "put":
        return strike * dr * ndtr(-d2) - spot * dq * ndtr(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def greeks(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    vol: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> dict[str, float]:
    """Return per-share delta, gamma, vega, and theta (calendar-year units)."""
    d1, d2 = _d1d2(spot, strike, time, rate, vol, dividend)
    pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    dq, dr = math.exp(-dividend * time), math.exp(-rate * time)
    call = kind.lower() == "call"
    if not call and kind.lower() != "put":
        raise ValueError("kind must be 'call' or 'put'")
    delta = dq * (ndtr(d1) if call else ndtr(d1) - 1)
    gamma = dq * pdf / (spot * vol * math.sqrt(time))
    vega = spot * dq * pdf * math.sqrt(time)
    theta_call = (
        -(spot * dq * pdf * vol / (2 * math.sqrt(time)))
        - rate * strike * dr * ndtr(d2)
        + dividend * spot * dq * ndtr(d1)
    )
    theta = theta_call if call else theta_call + rate * strike * dr - dividend * spot * dq
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


def price_bounds(
    spot: float, strike: float, time: float, rate: float, kind: str, dividend: float = 0.0
) -> tuple[float, float]:
    """European no-arbitrage lower and upper bounds."""
    s, k = spot * math.exp(-dividend * time), strike * math.exp(-rate * time)
    if kind.lower() == "call":
        return max(s - k, 0.0), s
    if kind.lower() == "put":
        return max(k - s, 0.0), k
    raise ValueError("kind must be 'call' or 'put'")


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    kind: str = "call",
    dividend: float = 0.0,
    *,
    tolerance: float = 1e-8,
) -> float:
    """Invert Black–Scholes with Brent's method; reject prices outside theoretical bounds."""
    if time <= 0 or spot <= 0 or strike <= 0 or market_price < 0:
        raise ValueError("invalid option inputs")
    lower, upper = price_bounds(spot, strike, time, rate, kind, dividend)
    eps = max(1e-10, tolerance * 0.1)
    if market_price < lower - eps or market_price > upper + eps:
        raise ValueError(
            f"price {market_price:.6g} outside no-arbitrage bounds [{lower:.6g}, {upper:.6g}]"
        )
    if abs(market_price - lower) <= eps:
        raise ValueError("price at intrinsic bound has no finite positive implied volatility")
    f = lambda v: price(spot, strike, time, rate, v, kind, dividend) - market_price
    hi = 2.0
    while f(hi) < 0 and hi < 20:
        hi *= 2
    if f(hi) < 0:
        raise ValueError("could not bracket implied volatility")
    return float(brentq(f, 1e-8, hi, xtol=tolerance, maxiter=200))

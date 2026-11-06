"""Pluggable option-chain providers, synthetic demo data, and disk caching."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.pricing import price
from core.svi import SVIParams

CACHE_DIR = Path(".cache/chains")


class ChainProvider(ABC):
    @abstractmethod
    def get_chain(self, ticker: str) -> pd.DataFrame: ...


class YFinanceProvider(ChainProvider):
    """Fetch calls and puts from yfinance (delayed quotes may be returned)."""

    def get_chain(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf

        obj = yf.Ticker(ticker.upper())
        spot = float(obj.fast_info.get("last_price") or obj.info["regularMarketPrice"])
        rows = []
        for expiry in obj.options:
            t = max(
                (
                    datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    - datetime.now(timezone.utc)
                ).total_seconds()
                / 31557600,
                1e-5,
            )
            for kind, chain in (
                ("call", obj.option_chain(expiry).calls),
                ("put", obj.option_chain(expiry).puts),
            ):
                for _, r in chain.iterrows():
                    rows.append(
                        {
                            "ticker": ticker.upper(),
                            "expiry": expiry,
                            "T": t,
                            "strike": float(r.strike),
                            "kind": kind,
                            "bid": float(r.bid or 0),
                            "ask": float(r.ask or 0),
                            "lastPrice": float(r.lastPrice or 0),
                            "volume": float(r.volume or 0),
                            "openInterest": float(r.openInterest or 0),
                            "impliedVolatility_vendor": float(r.impliedVolatility or 0),
                            "spot": spot,
                            "source": "yfinance",
                        }
                    )
        if not rows:
            raise ValueError(f"No option chains returned for {ticker}")
        return pd.DataFrame(rows)


class SyntheticProvider(ChainProvider):
    """Create reproducible option quotes generated from a known SVI surface."""

    def get_chain(self, ticker: str = "SYNTH") -> pd.DataFrame:
        spot = 100.0
        rate = 0.04
        params = [
            SVIParams(0.012, 0.09, -0.35, 0.0, 0.18),
            SVIParams(0.035, 0.14, -0.35, 0.0, 0.22),
            SVIParams(0.065, 0.19, -0.35, 0.0, 0.28),
        ]
        expiries = ["2026-12-18", "2027-03-19", "2027-09-17"]
        times = [0.12, 0.37, 0.87]
        rows = []
        for exp, t, p in zip(expiries, times, params):
            for k in np.linspace(-0.35, 0.35, 17):
                F = spot * np.exp(rate * t)
                K = F * np.exp(k)
                w = float(p.variance(k))
                vol = np.sqrt(w / t)
                kind = "put" if k < 0 else "call"
                mid = price(spot, K, t, rate, vol, kind)
                spread = max(0.03, mid * 0.025)
                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "expiry": exp,
                        "T": t,
                        "strike": K,
                        "kind": kind,
                        "bid": max(mid - spread / 2, 0.01),
                        "ask": mid + spread / 2,
                        "lastPrice": mid,
                        "volume": 250,
                        "openInterest": 1800,
                        "spot": spot,
                        "source": "synthetic",
                        "true_iv": vol,
                    }
                )
        return pd.DataFrame(rows)


def cached_chain(
    ticker: str, provider: ChainProvider, *, refresh=False
) -> tuple[pd.DataFrame, str]:
    """Fetch chain, falling back to cached CSV on upstream errors."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(ticker.upper().encode()).hexdigest()[:12]
    path = CACHE_DIR / f"{key}.csv"
    if not refresh and path.exists():
        try:
            return pd.read_csv(path), "cache"
        except Exception:
            pass
    try:
        df = provider.get_chain(ticker)
        df.to_csv(path, index=False)
        return df, "live"
    except Exception:
        if path.exists():
            return pd.read_csv(path), "cache-fallback"
        raise


def fetch_risk_free_rate(default=0.04) -> tuple[float, str]:
    """Read annualized 13-week Treasury proxy (^IRX); fall back to supplied rate."""
    try:
        import yfinance as yf

        hist = yf.Ticker("^IRX").history(period="5d")
        value = float(hist["Close"].dropna().iloc[-1]) / 100
        if 0 < value < 0.3:
            return value, "^IRX"
    except Exception:
        pass
    return float(default), "configured fallback"

"""FastAPI routes for Indian NSE implied-volatility surfaces."""

import logging
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from core.market import IST
from core.pipeline import fit_surface, prepare_chain
from data.provider import (
    DEFAULT_INDIA_RISK_FREE_RATE,
    MAX_CHAIN_AGE_SECONDS,
    NSEProvider,
    get_chain,
)

logger = logging.getLogger(__name__)
app = FastAPI(
    title="NSE Implied Volatility Surface API",
    version="0.2.0",
    description="Indian options IV analytics with parity forwards and SVI fitting.",
)


@app.get("/health")
def health():
    return {"status": "ok", "timezone": "Asia/Kolkata"}


@app.get("/surface")
def surface(
    symbol: str = Query("NIFTY", min_length=1, max_length=30),
    source: str = Query("auto", pattern="^(auto|nse|broker|synthetic)$"),
    rate: float = Query(DEFAULT_INDIA_RISK_FREE_RATE, gt=-0.1, lt=0.5),
    refresh: bool = False,
):
    try:
        raw, actual_source = get_chain(symbol, source, refresh)
        points, quality, forwards = prepare_chain(raw, rate)
        if len(points) < 5:
            raise ValueError("Too few clean OTM option quotes to fit a surface")
        fitted, models, checks = fit_surface(points)
        fetched = pd.to_datetime(raw["fetched_at"].iloc[0])
        fetched = fetched.tz_localize(IST) if fetched.tzinfo is None else fetched.tz_convert(IST)
        timestamp = fetched.isoformat()
        age = max(0.0, (datetime.now(IST) - fetched.to_pydatetime()).total_seconds())
        vix = raw.attrs.get("india_vix")
        if source != "synthetic" and actual_source in {"nse", "broker", "cache"} and vix is None:
            try:
                vix, _ = NSEProvider().get_india_vix()
            except Exception as exc:
                logger.info("India VIX not available: %s", exc)
        lot_size = (
            next((int(x) for x in raw["lot_size"].dropna() if int(x) > 0), None)
            if "lot_size" in raw
            else None
        )
        provider_name = {
            "broker": "FYERS",
            "nse": "NSE",
            "cache": str(raw.attrs.get("cache_origin") or "unknown").upper() + " cache",
            "synthetic": "Synthetic demo",
        }.get(actual_source, actual_source)
        rows = fitted.copy()
        rows["expiry"] = [pd.Timestamp(x).isoformat() for x in rows.expiry]
        rows = rows.astype(object).where(pd.notna(rows), None)
        return {
            "symbol": symbol.upper(),
            "requested_source": source,
            "source": actual_source,
            "provider": provider_name,
            "timestamp_ist": timestamp,
            "freshness_seconds": age,
            "freshness_limit_seconds": MAX_CHAIN_AGE_SECONDS,
            "is_stale": age > MAX_CHAIN_AGE_SECONDS,
            "lot_size": lot_size,
            "currency": "INR",
            "risk_free_rate": rate,
            "risk_free_source": "configured Indian rate",
            "india_vix": vix,
            "quality": quality,
            "forwards": forwards,
            "arbitrage": checks,
            "points": rows.to_dict(orient="records"),
            "svi": {
                e: {"a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma}
                for e, p in models.items()
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

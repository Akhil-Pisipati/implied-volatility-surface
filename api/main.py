"""FastAPI service exposing chain analysis and surface diagnostics."""

from fastapi import FastAPI, HTTPException, Query

from core.pipeline import fit_surface, prepare_chain
from data.provider import SyntheticProvider, YFinanceProvider, cached_chain, fetch_risk_free_rate

app = FastAPI(
    title="Implied Volatility Surface API",
    version="0.1.0",
    description="Option-chain IV analytics with SVI fitting and static-arbitrage diagnostics.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/surface")
def surface(
    ticker: str = Query("SPY", min_length=1, max_length=15),
    source: str = Query("yfinance", pattern="^(yfinance|synthetic)$"),
    rate: float | None = None,
    refresh: bool = False,
):
    try:
        provider = SyntheticProvider() if source == "synthetic" else YFinanceProvider()
        raw, origin = cached_chain(ticker, provider, refresh=refresh)
        rf, rf_source = fetch_risk_free_rate() if rate is None else (rate, "request")
        points, quality, forwards = prepare_chain(raw, rf)
        if len(points) < 5:
            raise ValueError("Too few clean OTM option quotes to fit a surface")
        fitted, models, checks = fit_surface(points)
        return {
            "ticker": ticker.upper(),
            "source": origin,
            "risk_free_rate": rf,
            "risk_free_source": rf_source,
            "quality": quality,
            "forwards": forwards,
            "arbitrage": checks,
            "points": fitted.to_dict(orient="records"),
            "svi": {
                e: {"a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma}
                for e, p in models.items()
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

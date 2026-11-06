"""End-to-end market quote cleaning, IV computation, and surface fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pricing import implied_volatility
from core.svi import arbitrage_checks, fit_svi

REQUIRED = {"expiry", "T", "strike", "kind", "bid", "ask", "spot"}


def prepare_chain(
    raw: pd.DataFrame,
    rate: float,
    *,
    min_volume=1,
    min_open_interest=10,
    max_spread_pct=0.5,
    min_mid=0.03,
):
    """Clean quotes, estimate expiry forwards from parity, and solve IVs.

    Returns (valid points, quality summary, forward-by-expiry). Rejection reasons
    are counted in the quality panel rather than silently discarded.
    """
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"chain missing columns: {sorted(missing)}")
    df = raw.copy()
    df["reason"] = ""
    df["mid"] = (df.bid + df.ask) / 2

    def reject(mask, reason):
        ix = df.index[mask & df.reason.eq("")]
        df.loc[ix, "reason"] = reason

    reject((df.bid <= 0) | (~np.isfinite(df.bid)), "zero/invalid bid")
    reject((df.ask < df.bid) | (~np.isfinite(df.ask)), "invalid ask")
    reject((df.mid < min_mid), "mid below minimum")
    reject(((df.ask - df.bid) / df.mid.clip(lower=1e-9) > max_spread_pct), "wide spread")
    if "volume" in df:
        reject(df.volume.fillna(0) < min_volume, "low volume")
    if "openInterest" in df:
        reject(df.openInterest.fillna(0) < min_open_interest, "low open interest")
    forwards = {}
    for expiry, g in df[df.reason.eq("")].groupby("expiry"):
        calls = g[g.kind.str.lower().eq("call")].set_index("strike")
        puts = g[g.kind.str.lower().eq("put")].set_index("strike")
        common = calls.index.intersection(puts.index)
        estimates = []
        for strike in common:
            c, p = calls.loc[strike], puts.loc[strike]
            if isinstance(c, pd.DataFrame):
                continue
            t = float(c["T"])
            estimates.append(float(strike + (c.mid - p.mid) * np.exp(rate * t)))
        forwards[str(expiry)] = (
            float(np.median(estimates))
            if estimates
            else float(g.spot.iloc[0] * np.exp(rate * float(g["T"].median())))
        )
    for ix, row in df[df.reason.eq("")].iterrows():
        try:
            F = forwards[str(row.expiry)]
            k = float(np.log(row.strike / F))
            # Retain only OTM options, as specified; ATM convention uses calls.
            if (row.kind.lower() == "put" and k >= 0) or (row.kind.lower() == "call" and k < 0):
                df.loc[ix, "reason"] = "not OTM under forward"
                continue
            iv = implied_volatility(
                float(row.mid),
                float(row.spot),
                float(row.strike),
                float(row["T"]),
                rate,
                str(row.kind).lower(),
            )
            df.loc[ix, "iv"] = iv
            df.loc[ix, "forward"] = F
            df.loc[ix, "k"] = k
            df.loc[ix, "total_variance"] = iv * iv * float(row["T"])
        except Exception as exc:
            df.loc[ix, "reason"] = f"invalid price/IV: {str(exc)[:100]}"
    valid = df[df.reason.eq("")].copy()
    rejects = df[df.reason.ne("")].reason.value_counts().to_dict()
    quality = {
        "received": len(df),
        "kept": len(valid),
        "rejected": int(len(df) - len(valid)),
        "reasons": rejects,
    }
    return valid, quality, forwards


def fit_surface(points: pd.DataFrame):
    """Fit per-expiry SVI models; attach fitted variance and arbitrage checks."""
    models = {}
    ordered = points.sort_values("T").groupby("expiry", sort=False)
    for expiry, g in ordered:
        models[str(expiry)] = fit_svi(g.k.to_numpy(), g.total_variance.to_numpy())
    fitted = points.copy()
    fitted["fitted_total_variance"] = [
        models[str(e)].variance(float(k)) for e, k in zip(fitted.expiry, fitted.k)
    ]
    fitted["fitted_iv"] = np.sqrt(np.maximum(fitted.fitted_total_variance / fitted["T"], 0))
    times = [float(g["T"].median()) for _, g in ordered]
    checks = (
        arbitrage_checks(list(models.values()), times)
        if len(models) > 0
        else {
            "calendar_violations": [],
            "butterfly_violations": [],
            "calendar_ok": True,
            "butterfly_ok": True,
        }
    )
    return fitted, models, checks

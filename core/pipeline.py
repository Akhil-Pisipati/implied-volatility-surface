"""India-aware quote cleaning, parity-forward estimation, and IV calibration."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.market import now_ist, time_to_expiry
from core.pricing import implied_volatility
from core.svi import arbitrage_checks, fit_svi

logger = logging.getLogger(__name__)
REQUIRED = {
    "symbol",
    "spot",
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "ltp",
    "volume",
    "open_interest",
}
FILTERS = {
    "index": {"min_volume": 1, "min_open_interest": 25, "max_spread_pct": 0.35, "min_mid": 0.5},
    "stock": {"min_volume": 1, "min_open_interest": 10, "max_spread_pct": 0.45, "min_mid": 0.5},
}


def estimate_forward_parity(
    chain: pd.DataFrame, rate: float, *, t: float | None = None, near_atm_width=0.12
):
    """Estimate F by least-squares C-P = DF(F-K) on liquid near-ATM matched strikes."""
    cols = {"strike", "option_type", "bid", "ask"}
    if not cols.issubset(chain.columns):
        raise ValueError(f"parity data missing {sorted(cols-set(chain.columns))}")
    d = chain.copy()
    d["mid"] = (d.bid + d.ask) / 2
    c = (
        d[d.option_type.str.lower().isin(["call", "ce"])]
        .drop_duplicates("strike")
        .set_index("strike")
    )
    p = (
        d[d.option_type.str.lower().isin(["put", "pe"])]
        .drop_duplicates("strike")
        .set_index("strike")
    )
    common = c.index.intersection(p.index)
    if len(common) < 3:
        raise ValueError("at least three matched call/put strikes are needed for parity regression")
    spot = (
        float(d.spot.iloc[0])
        if "spot" in d and pd.notna(d.spot.iloc[0])
        else float(np.median(common))
    )
    near = [k for k in common if abs(np.log(float(k) / spot)) <= near_atm_width]
    if len(near) < 3:
        near = sorted(common, key=lambda k: abs(np.log(float(k) / spot)))[
            : max(3, min(9, len(common)))
        ]
    strikes = np.asarray(near, dtype=float)
    diff = np.asarray([float(c.loc[k, "mid"] - p.loc[k, "mid"]) for k in near])
    slope, intercept = np.polyfit(strikes, diff, 1)
    discount = -float(slope)
    if not np.isfinite(discount) or not 0.5 < discount < 1.2:
        raise ValueError("put-call parity regression returned invalid discount factor")
    forward = float(intercept / discount)
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("parity regression returned invalid forward")
    return forward, discount, len(near)


def prepare_chain(
    raw: pd.DataFrame,
    rate: float,
    *,
    min_volume=None,
    min_open_interest=None,
    max_spread_pct=None,
    min_mid=None,
    filters=None,
    now=None,
):
    """Clean normalized NSE quotes, infer each forward, select OTM options, and solve IVs.

    India filter thresholds are selected by symbol family and can be overridden
    explicitly. T is recalculated using actual expiry time at 15:30 IST and ACT/365.
    """
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"chain missing columns: {sorted(missing)}")
    current = now or now_ist()
    df = raw.copy()
    df["reason"] = ""
    df["option_type"] = df.option_type.astype(str).str.lower().replace({"ce": "call", "pe": "put"})
    df["mid"] = (df.bid + df.ask) / 2
    sym = str(df.symbol.iloc[0]).upper()
    config = {
        **FILTERS[
            (
                "index"
                if sym in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
                else "stock"
            )
        ],
        **(filters or {}),
    }
    config.update(
        {
            k: v
            for k, v in {
                "min_volume": min_volume,
                "min_open_interest": min_open_interest,
                "max_spread_pct": max_spread_pct,
                "min_mid": min_mid,
            }.items()
            if v is not None
        }
    )

    def reject(mask, reason):
        ix = df.index[mask & df.reason.eq("")]
        df.loc[ix, "reason"] = reason

    reject((df.bid <= 0) | (~np.isfinite(df.bid)), "zero/invalid bid")
    reject((df.ask < df.bid) | (~np.isfinite(df.ask)), "invalid ask")
    reject(df.mid < config["min_mid"], "mid below minimum")
    reject(((df.ask - df.bid) / df.mid.clip(lower=1e-9) > config["max_spread_pct"]), "wide spread")
    reject(df.volume.fillna(0) < config["min_volume"], "low volume")
    reject(df.open_interest.fillna(0) < config["min_open_interest"], "low open interest")
    df["T"] = [time_to_expiry(x, current) for x in df.expiry]
    forwards = {}
    parity_info = {}
    for expiry, g in df[df.reason.eq("")].groupby("expiry"):
        try:
            t = float(g["T"].median())
            if t <= 0:
                raise ValueError("expiry has passed")
            forward, discount, n = estimate_forward_parity(g, rate, t=t)
            forwards[expiry_at_text(expiry)] = forward
            parity_info[expiry_at_text(expiry)] = {
                "method": "put-call parity regression",
                "discount_factor": discount,
                "matched_strikes": n,
            }
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            t = float(g["T"].median())
            spot = float(g.spot.iloc[0])
            forward = spot * np.exp(rate * t)
            key = expiry_at_text(expiry)
            forwards[key] = forward
            parity_info[key] = {"method": "carry fallback", "warning": str(exc)}
            logger.warning("Parity forward failed for %s; using carry forward: %s", key, exc)
    for ix, row in df[df.reason.eq("")].iterrows():
        try:
            key = expiry_at_text(row.expiry)
            F = forwards[key]
            T = float(row["T"])
            if T <= 0:
                raise ValueError("expiry has passed")
            k = float(np.log(float(row.strike) / F))
            kind = row.option_type
            if (kind == "put" and k >= 0) or (kind == "call" and k < 0):
                df.loc[ix, "reason"] = "not OTM relative to parity forward"
                continue
            q = rate - np.log(F / float(row.spot)) / T if row.spot > 0 else 0.0
            iv = implied_volatility(
                float(row.mid), float(row.spot), float(row.strike), T, rate, kind, dividend=float(q)
            )
            df.loc[ix, "forward"] = F
            df.loc[ix, "k"] = k
            df.loc[ix, "dividend_yield_inferred"] = q
            df.loc[ix, "computed_iv"] = iv
            df.loc[ix, "iv"] = iv
            df.loc[ix, "total_variance"] = iv * iv * T
        except (ValueError, TypeError, OverflowError) as exc:
            df.loc[ix, "reason"] = f"invalid price/IV: {str(exc)[:100]}"
    valid = df[df.reason.eq("")].copy()
    quality = {
        "received": len(df),
        "kept": len(valid),
        "rejected": int(len(df) - len(valid)),
        "reasons": df[df.reason.ne("")].reason.value_counts().to_dict(),
        "filters": config,
        "forwards": parity_info,
    }
    return valid, quality, forwards


def expiry_at_text(value):
    from core.market import expiry_at_ist

    return expiry_at_ist(value).isoformat()


def fit_surface(points: pd.DataFrame):
    """Fit one raw SVI smile per listed expiry and run finite-grid arbitrage checks."""
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
        if models
        else {
            "calendar_violations": [],
            "butterfly_violations": [],
            "calendar_ok": True,
            "butterfly_ok": True,
        }
    )
    return fitted, models, checks

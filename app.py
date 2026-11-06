"""Streamlit dashboard for Indian F&O volatility surfaces."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.market import IST, expiry_kind, is_market_open, now_ist
from core.pipeline import fit_surface, prepare_chain
from core.svi import fit_ssvi
from data.provider import (
    DEFAULT_INDIA_RISK_FREE_RATE,
    MAX_CHAIN_AGE_SECONDS,
    NSEProvider,
    get_chain,
)

st.set_page_config(page_title="NSE Volatility Surface", page_icon="N", layout="wide")
st.title("Indian Options Implied Volatility Surface")
st.caption("NSE F&O · Black–Scholes–Merton · parity forwards · SVI / SSVI")
SYMBOLS = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "RELIANCE",
    "HDFCBANK",
    "INFY",
    "TCS",
    "ICICIBANK",
]
SOURCE_OPTIONS = ["auto", "nse", "broker", "synthetic"]
DEFAULT_SOURCE = os.getenv("DEFAULT_DATA_SOURCE", "auto").lower()
with st.sidebar:
    symbol = st.selectbox("Underlying", SYMBOLS)
    requested = st.selectbox(
        "Source preference",
        SOURCE_OPTIONS,
        index=SOURCE_OPTIONS.index(DEFAULT_SOURCE) if DEFAULT_SOURCE in SOURCE_OPTIONS else 0,
        help="Auto tries FYERS, then NSE, local cache, then synthetic demo.",
    )
    rate = st.number_input(
        "Indian annual risk-free assumption",
        min_value=-0.02,
        max_value=0.3,
        value=float(DEFAULT_INDIA_RISK_FREE_RATE),
        step=0.0025,
        format="%.4f",
        help="Set to a current 91-day T-bill or RBI repo assumption; configured default is named in INDIA_RISK_FREE_RATE.",
    )
    refresh = st.button("Refresh data", type="primary")
    st.caption(
        "FYERS Standard API access has free usage limits. Add its client ID and daily access token to .env."
    )
now = now_ist()
if not is_market_open(now):
    st.warning(
        "Market closed — showing the last available data (or synthetic demo if no snapshot is cached)."
    )
try:
    raw, actual = get_chain(symbol, requested, refresh)
    points, quality, forwards = prepare_chain(raw, rate)
    if len(points) < 5:
        st.error("Too few clean OTM options to fit a reliable surface.")
        st.stop()
    fitted, models, checks = fit_surface(points)
except Exception as exc:
    st.error(f"Unable to build the surface: {exc}")
    st.info("Choose Synthetic for an offline demo, or try Auto when NSE/FYERS data is reachable.")
    st.stop()
fetch = pd.to_datetime(raw.fetched_at.iloc[0])
fetch = fetch.tz_localize(IST) if fetch.tzinfo is None else fetch.tz_convert(IST)
age = max(0, (now - fetch.to_pydatetime()).total_seconds())
lot = (
    next((int(x) for x in raw.lot_size.dropna() if int(x) > 0), None) if "lot_size" in raw else None
)
vix = raw.attrs.get("india_vix")
if actual in {"nse", "broker", "cache"} and vix is None:
    try:
        vix, _ = NSEProvider().get_india_vix()
    except Exception:
        pass
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Underlying", symbol)
c2.metric("Valid quotes", f"{quality['kept']} / {quality['received']}")
source_label = {
    "broker": "FYERS",
    "nse": "NSE",
    "cache": f"CACHE ({str(raw.attrs.get('cache_origin') or 'unknown').upper()})",
    "synthetic": "SYNTHETIC",
}.get(actual, actual.upper())
c3.metric("Data source", source_label)
c4.metric("Lot size", str(lot) if lot else "Not supplied")
c5.metric("Snapshot age", f"{age/60:.1f} min")
if actual != "synthetic" and age > MAX_CHAIN_AGE_SECONDS:
    st.error(
        f"Stale option-chain snapshot: {age/60:.1f} min old; freshness limit is "
        f"{MAX_CHAIN_AGE_SECONDS/60:.1f} min. Refresh or switch sources."
    )
st.caption(
    f"Snapshot: {fetch.strftime('%d %b %Y %H:%M:%S IST')} · INR · Risk-free rate {rate:.2%} · Forward estimation: put-call parity regression"
)
# 30-calendar-day ATM IV from SVI ATM total variance, interpolated in maturity.
ordered = sorted(
    models, key=lambda e: float(fitted.loc[fitted.expiry.astype(str).eq(e), "T"].median())
)
ts = np.array([float(fitted.loc[fitted.expiry.astype(str).eq(e), "T"].median()) for e in ordered])
atm_w = np.array([float(models[e].variance(0)) for e in ordered])
target = 30 / 365
w30 = float(np.interp(target, ts, atm_w)) if len(ts) >= 2 else float(atm_w[0])
atm30 = float(np.sqrt(max(w30 / target, 0)) * 100)
vix_col1, vix_col2, vix_col3 = st.columns(3)
vix_col1.metric("30-day ATM IV", f"{atm30:.2f}%")
vix_col2.metric("India VIX", f"{vix:.2f}%" if vix is not None else "Unavailable")
vix_col3.metric("ATM IV − India VIX", f"{atm30-vix:+.2f} pp" if vix is not None else "Unavailable")
left, right = st.columns([2, 1])
with left:
    st.subheader("Fitted volatility surface")
    grid_k = np.linspace(float(fitted.k.min()) - 0.08, float(fitted.k.max()) + 0.08, 60)
    exp_names = ordered
    expiry_t = np.array(
        [float(fitted.loc[fitted.expiry.astype(str).eq(e), "T"].median()) for e in exp_names]
    )
    grid_iv = np.array(
        [
            np.sqrt(np.maximum(models[e].variance(grid_k) / t, 0)) * 100
            for e, t in zip(exp_names, expiry_t)
        ]
    )
    fig = go.Figure(
        [
            go.Surface(
                x=grid_k,
                y=expiry_t,
                z=grid_iv,
                colorscale="Viridis",
                colorbar={"title": "IV %"},
                name="SVI",
            ),
            go.Scatter3d(
                x=fitted.k,
                y=fitted["T"],
                z=fitted.computed_iv * 100,
                mode="markers",
                marker={"size": 3, "color": "#ff785a"},
                name="Market points",
            ),
        ]
    )
    fig.update_layout(
        scene={
            "xaxis_title": "Log-moneyness ln(K/F)",
            "yaxis_title": "Maturity (years)",
            "zaxis_title": "Implied volatility (%)",
        },
        height=560,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
    )
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Data quality")
    st.write(
        f"Kept **{quality['kept']}** of **{quality['received']}** rows; rejected **{quality['rejected']}**."
    )
    if quality["reasons"]:
        st.dataframe(
            pd.DataFrame([{"Reason": k, "Count": v} for k, v in quality["reasons"].items()]),
            hide_index=True,
            width="stretch",
        )
    st.subheader("Static-arbitrage checks")
    st.write(
        f"Calendar: {'PASS' if checks['calendar_ok'] else 'VIOLATIONS'} · Butterfly: {'PASS' if checks['butterfly_ok'] else 'VIOLATIONS'}"
    )
    if checks["calendar_violations"]:
        st.json(checks["calendar_violations"])
    if checks["butterfly_violations"]:
        st.json(checks["butterfly_violations"])
st.subheader("Smile slices by listed expiry")
smile = go.Figure()
for expiry, g in fitted.groupby("expiry"):
    label = expiry_kind(expiry, list(fitted.expiry.unique()))
    g = g.sort_values("k")
    name = pd.Timestamp(expiry).strftime("%d %b %Y")
    smile.add_scatter(x=g.k, y=g.computed_iv * 100, mode="markers", name=f"{name} market")
    smile.add_scatter(x=g.k, y=g.fitted_iv * 100, mode="lines", name=f"{name} SVI · {label}")
smile.update_layout(xaxis_title="Log-moneyness ln(K/F)", yaxis_title="IV (%)", height=390)
st.plotly_chart(smile, width="stretch")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Computed IV vs exchange IV")
    cross = fitted.dropna(subset=["exchange_iv"])
    cross = cross[cross.exchange_iv > 0]
    if len(cross):
        mad = float(np.mean(np.abs(cross.computed_iv - cross.exchange_iv)) * 100)
        st.metric("Mean absolute difference", f"{mad:.2f} percentage points")
        crossfig = go.Figure(
            go.Scatter(
                x=cross.exchange_iv * 100,
                y=cross.computed_iv * 100,
                mode="markers",
                marker={
                    "color": cross["T"],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "T"},
                },
                text=cross.strike,
                hovertemplate="Strike %{text}<br>Exchange %{x:.2f}%<br>Computed %{y:.2f}%<extra></extra>",
            )
        )
        crossfig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max(cross.exchange_iv.max(), cross.computed_iv.max()) * 100,
            y1=max(cross.exchange_iv.max(), cross.computed_iv.max()) * 100,
            line={"dash": "dash"},
        )
        crossfig.update_layout(
            xaxis_title="Exchange IV (%)", yaxis_title="Computed IV (%)", height=350
        )
        st.plotly_chart(crossfig, width="stretch")
    else:
        st.info("This source did not supply exchange IV; it is not used as an input to the fit.")
with col2:
    st.subheader("Term structure and smile heatmap")
    atm = go.Figure()
    for expiry in ordered:
        group = fitted[fitted.expiry.astype(str).eq(expiry)]
        t = float(group["T"].median())
        atm.add_scatter(
            x=[t],
            y=[np.sqrt(float(models[expiry].variance(0)) / max(t, 1e-12)) * 100],
            mode="markers+text",
            text=[f"{pd.Timestamp(expiry):%d %b} · {expiry_kind(expiry,ordered)}"],
            textposition="top center",
            showlegend=False,
        )
    atm.update_layout(xaxis_title="Years to expiry", yaxis_title="ATM IV (%)", height=350)
    st.plotly_chart(atm, width="stretch")
with st.expander("Optional global SSVI fit"):
    try:
        groups = list(fitted.sort_values("T").groupby("expiry", sort=False))
        st.json(
            fit_ssvi(
                [float(g["T"].median()) for _, g in groups],
                [g.k.to_numpy() for _, g in groups],
                [g.total_variance.to_numpy() for _, g in groups],
            )
        )
    except Exception as exc:
        st.warning(f"SSVI fit needs enough data across two or more expiries: {exc}")
st.caption(
    "Educational use only, not investment advice. NSE web endpoints are unofficial, may rate-limit/change, and should be used in compliance with NSE data terms. Surface checks are numerical and finite-grid approximations."
)

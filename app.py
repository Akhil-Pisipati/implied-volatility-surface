"""Interactive dashboard; run with `streamlit run app.py`."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pipeline import fit_surface, prepare_chain
from core.svi import fit_ssvi
from data.provider import SyntheticProvider, YFinanceProvider, cached_chain, fetch_risk_free_rate

st.set_page_config(page_title="Volatility Surface Builder", page_icon="📈", layout="wide")
st.title("Implied Volatility Surface Builder")
st.caption("Market option chains · SVI / SSVI · static-arbitrage diagnostics")
with st.sidebar:
    ticker = st.text_input("Ticker", "SPY").strip().upper() or "SPY"
    source = st.selectbox("Data source", ["Synthetic demo", "Yahoo Finance"])
    default_rate, rate_source = fetch_risk_free_rate()
    rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=-0.05,
        max_value=0.3,
        value=float(default_rate),
        step=0.0025,
        format="%.4f",
    )
    refresh = st.button("Refresh data", type="primary")
    st.caption(f"Rate source: {rate_source}. Synthetic mode works offline.")
try:
    provider = SyntheticProvider() if source.startswith("Synthetic") else YFinanceProvider()
    raw, origin = cached_chain(ticker, provider, refresh=refresh)
    points, quality, forwards = prepare_chain(raw, rate)
    if len(points) < 5:
        st.error("Too few clean OTM quotes for surface fitting.")
        st.stop()
    fitted, models, checks = fit_surface(points)
except Exception as exc:
    st.error(f"Unable to build surface: {exc}")
    st.info(
        "Try Synthetic demo for an offline sample or retry later if Yahoo Finance is unavailable."
    )
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid quotes", f"{quality['kept']} / {quality['received']}")
m2.metric("Expiries", str(len(models)))
m3.metric("Underlying", ticker)
m4.metric("Data", origin)
left, right = st.columns([2, 1])
with left:
    st.subheader("Fitted implied-volatility surface")
    all_k = np.linspace(float(fitted.k.min()) - 0.08, float(fitted.k.max()) + 0.08, 55)
    expiry_names = list(models)
    ts = np.array([float(fitted.loc[fitted.expiry.eq(e), "T"].median()) for e in expiry_names])
    grid_w = np.array([models[e].variance(all_k) for e in expiry_names])
    grid_iv = np.sqrt(np.maximum(grid_w / ts[:, None], 0)) * 100
    fig = go.Figure(
        data=[
            go.Surface(
                x=all_k,
                y=ts,
                z=grid_iv,
                colorscale="Viridis",
                colorbar={"title": "IV %"},
                name="SVI surface",
            ),
            go.Scatter3d(
                x=fitted.k,
                y=fitted["T"],
                z=fitted.iv * 100,
                mode="markers",
                marker={"size": 3, "color": "#ff6b6b"},
                name="Market points",
            ),
        ]
    )
    fig.update_layout(
        scene={
            "xaxis_title": "Log-moneyness k=ln(K/F)",
            "yaxis_title": "Time to expiry (years)",
            "zaxis_title": "Implied volatility (%)",
        },
        height=570,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Data quality")
    st.write(f"**Kept:** {quality['kept']}  ·  **Rejected:** {quality['rejected']}")
    if quality["reasons"]:
        st.dataframe(
            pd.DataFrame([{"Reason": k, "Count": v} for k, v in quality["reasons"].items()]),
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Arbitrage checks")
    st.write(f"Calendar: {'PASS' if checks['calendar_ok'] else 'VIOLATIONS'}")
    st.write(f"Butterfly: {'PASS' if checks['butterfly_ok'] else 'VIOLATIONS'}")
    if checks["calendar_violations"]:
        st.json(checks["calendar_violations"])
    if checks["butterfly_violations"]:
        st.json(checks["butterfly_violations"])

st.subheader("Smile slices by expiry")
smile = go.Figure()
for expiry, g in fitted.groupby("expiry"):
    g = g.sort_values("k")
    smile.add_scatter(x=g.k, y=g.iv * 100, mode="markers", name=f"{expiry} market")
    smile.add_scatter(x=g.k, y=g.fitted_iv * 100, mode="lines", name=f"{expiry} SVI")
smile.update_layout(xaxis_title="Log-moneyness", yaxis_title="IV (%)", height=400)
st.plotly_chart(smile, use_container_width=True)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Term structure")
    atm = go.Figure()
    for expiry, g in fitted.groupby("expiry"):
        t = float(g["T"].median())
        p = models[str(expiry)]
        atm.add_scatter(
            x=[t],
            y=[np.sqrt(float(p.variance(0)) / t) * 100],
            mode="markers+text",
            text=[str(expiry)],
            textposition="top center",
            name=str(expiry),
        )
    atm.update_layout(
        xaxis_title="Time (years)", yaxis_title="ATM IV (%)", height=350, showlegend=False
    )
    st.plotly_chart(atm, use_container_width=True)
with col2:
    st.subheader("Market IV heatmap")
    heat = (
        fitted.pivot_table(
            index="expiry", columns=pd.cut(fitted.k, 12), values="iv", aggfunc="mean"
        )
        * 100
    )
    heat.columns = [f"{x.left:.2f}" for x in heat.columns]
    st.plotly_chart(
        go.Figure(
            go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorscale="Viridis",
                colorbar={"title": "IV %"},
            )
        ).update_layout(xaxis_title="Log-moneyness bucket", yaxis_title="Expiry", height=350),
        use_container_width=True,
    )
with st.expander("SSVI global fit (optional)"):
    try:
        grouped = list(fitted.sort_values("T").groupby("expiry", sort=False))
        ss = fit_ssvi(
            [float(g["T"].median()) for _, g in grouped],
            [g.k.to_numpy() for _, g in grouped],
            [g.total_variance.to_numpy() for _, g in grouped],
        )
        st.json(ss)
    except Exception as exc:
        st.warning(f"SSVI fit requires sufficient observations across at least two expiries: {exc}")
st.caption(
    "Caveat: yfinance chains may be delayed; Black–Scholes assumes European exercise. SVI checks are numerical over a finite grid, not a formal global-arbitrage guarantee."
)

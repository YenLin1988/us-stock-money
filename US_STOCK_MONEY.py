from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from us_stock_money.alerts import evaluate_alerts
from us_stock_money.market_data import benchmark_table, build_sector_table, download_prices
from us_stock_money.scoring import broad_flow_score, classify_regime, flow_delta, group_scores
from us_stock_money.storage import HistoryStore


st.set_page_config(page_title="US STOCK MONEY", page_icon="$", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0b0f14; color: #e6edf3; }
    [data-testid="stMetricValue"] { color: #f0f6fc; }
    .flow-card {
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 16px;
        background: #111820;
    }
    .small-label { color: #8b949e; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = download_prices()
    return build_sector_table(data), benchmark_table(data)


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def main() -> None:
    st.title("US STOCK MONEY")
    st.caption("US equity sector money-flow radar powered by public Yahoo Finance market data.")

    try:
        sector_df, bench_df = load_data()
    except Exception as exc:  # pragma: no cover - Streamlit runtime display
        st.error(f"Could not load market data: {exc}")
        return

    sector_scores = dict(zip(sector_df["ticker"], sector_df["flow_score"], strict=False))
    groups = group_scores(sector_scores)
    broad = broad_flow_score(sector_scores)
    risk_on = groups.get("Risk-On", 0.0)
    defensive = groups.get("Defensive", 0.0)
    regime = classify_regime(broad, risk_on, defensive)

    store = HistoryStore(Path("data/flow_history.sqlite3"))
    now = dt.datetime.now().replace(second=0, microsecond=0)
    record = {
        "time": now.strftime("%Y-%m-%d %H:%M"),
        "broad_flow_score": round(broad, 2),
        "risk_on_score": round(risk_on, 2),
        "defensive_score": round(defensive, 2),
        "regime": regime.name,
        "leaders": sector_df.head(3)[["ticker", "sector", "flow_score"]].to_dict("records"),
        "laggards": sector_df.tail(3)[["ticker", "sector", "flow_score"]].to_dict("records"),
    }
    store.upsert_record(record)
    history = store.load_history()
    delta_24h = flow_delta(history, broad, 24, now)

    alerts = evaluate_alerts(
        {
            "broad_flow_score": broad,
            "risk_on_score": risk_on,
            "defensive_score": defensive,
            "delta_24h": delta_24h,
            "regime": regime.name,
        }
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime", regime.name)
    col2.metric("Broad Flow", f"{broad:.1f}/100", None if delta_24h is None else f"{delta_24h:+.1f} 24H")
    col3.metric("Risk-On", f"{risk_on:.1f}/100")
    col4.metric("Defensive", f"{defensive:.1f}/100")

    st.divider()

    left, right = st.columns([1.4, 1])
    with left:
        fig = px.bar(
            sector_df,
            x="ticker",
            y="flow_score",
            color="flow_score",
            hover_data=["sector", "return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_m"],
            color_continuous_scale=["#f85149", "#d29922", "#2ea043"],
            range_color=[0, 100],
            title="Sector Money Flow Score",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", height=430)
        fig.update_yaxes(range=[0, 100], title="Flow Score")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=list(groups.values()),
                theta=list(groups.keys()),
                fill="toself",
                name="Group Flow",
                line_color="#58a6ff",
            )
        )
        radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f14",
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
            title="Rotation Map",
            height=430,
        )
        st.plotly_chart(radar, use_container_width=True)

    st.subheader("Sector Table")
    display_df = sector_df.copy()
    for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
        display_df[column] = display_df[column].map(fmt_pct)
    display_df["flow_score"] = display_df["flow_score"].map(lambda x: f"{x:.1f}")
    display_df["dollar_volume_m"] = display_df["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.subheader("Benchmark Pulse")
        if not bench_df.empty:
            bench_display = bench_df.copy()
            for column in ["return_1d", "return_5d", "return_20d"]:
                bench_display[column] = bench_display[column].map(fmt_pct)
            st.dataframe(bench_display, use_container_width=True, hide_index=True)

    with lower_right:
        st.subheader("Alerts")
        if alerts:
            for alert in alerts:
                st.warning(f"{alert.title}: {alert.message}")
        else:
            st.success("No active flow alerts.")

    if history:
        hist_df = pd.DataFrame(history)
        if "broad_flow_score" in hist_df:
            st.subheader("Flow History")
            hist_fig = px.line(hist_df.tail(200), x="time", y="broad_flow_score", title="Broad Flow Score History")
            hist_fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", height=320)
            hist_fig.update_yaxes(range=[0, 100])
            st.plotly_chart(hist_fig, use_container_width=True)

    st.caption("Research tool only. Sector flow scores are proxies derived from price and volume, not official fund-flow data.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from us_stock_money.alerts import evaluate_alerts
from us_stock_money.market_data import benchmark_table, build_component_table, build_sector_table, build_theme_table, download_prices
from us_stock_money.model_config import MARKET_DATA_VERSION, WATCHLIST_TICKERS
from us_stock_money.scoring import broad_flow_score, build_top_recommendations, classify_regime, flow_delta, theme_group_scores
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
def load_data(market_data_version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = download_prices()
    return build_theme_table(data), build_component_table(data), build_sector_table(data), benchmark_table(data)


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def format_component_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
        display[column] = display[column].map(fmt_pct)
    display["flow_score"] = display["flow_score"].map(lambda x: f"{x:.1f}")
    display["dollar_volume_m"] = display["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
    return display


def watchlist_theme_labels(theme_df: pd.DataFrame) -> pd.DataFrame:
    watchlist = set(WATCHLIST_TICKERS)
    display = theme_df.copy()
    display["selected_watchlist"] = display["components"].map(
        lambda value: ", ".join([ticker for ticker in str(value).split(", ") if ticker in watchlist])
    )
    return display


def main() -> None:
    st.title("US STOCK MONEY")
    st.caption("US thematic money-flow radar: AI compute chain, power, defense, space, rare earths, nuclear, medical, and other rotation themes.")

    if st.button("Refresh market data", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    try:
        theme_df, component_df, sector_df, bench_df = load_data(MARKET_DATA_VERSION)
    except Exception as exc:  # pragma: no cover - Streamlit runtime display
        st.error(f"Could not load market data: {exc}")
        return

    theme_scores = dict(zip(theme_df["theme"], theme_df["flow_score"], strict=False))
    groups = theme_group_scores(theme_scores)
    broad = broad_flow_score(theme_scores)
    risk_on = groups.get("AI Compute Chain", 0.0)
    defensive = groups.get("Healthcare / Automation", 0.0)
    regime = classify_regime(broad, risk_on, defensive)

    store = HistoryStore(Path("data/flow_history.sqlite3"))
    now = dt.datetime.now().replace(second=0, microsecond=0)
    record = {
        "time": now.strftime("%Y-%m-%d %H:%M"),
        "broad_flow_score": round(broad, 2),
        "risk_on_score": round(risk_on, 2),
        "defensive_score": round(defensive, 2),
        "regime": regime.name,
        "leaders": theme_df.head(3)[["theme", "flow_score", "top_component"]].to_dict("records"),
        "laggards": theme_df.tail(3)[["theme", "flow_score", "weak_component"]].to_dict("records"),
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
    recommendations = build_top_recommendations(component_df, theme_scores, limit=5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime", regime.name)
    col2.metric("Broad Flow", f"{broad:.1f}/100", None if delta_24h is None else f"{delta_24h:+.1f} 24H")
    col3.metric("AI Compute Chain", f"{risk_on:.1f}/100")
    col4.metric("Healthcare / Automation", f"{defensive:.1f}/100")

    st.divider()

    st.subheader("Top 5 Flow Candidates")
    st.caption("Ranked by component money-flow score plus related theme strength. Research signal only, not financial advice.")
    rec_cols = st.columns(5)
    for col, rec in zip(rec_cols, recommendations, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="flow-card">
                    <div class="small-label">#{recommendations.index(rec) + 1} Flow Candidate</div>
                    <h3 style="margin: 0.2rem 0 0.1rem 0;">{rec["ticker"]}</h3>
                    <div class="small-label">{rec["themes"]}</div>
                    <p style="font-size: 1.35rem; margin: 0.6rem 0 0.2rem 0;">{float(rec["composite_score"]):.1f}</p>
                    <div class="small-label">Composite score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    rec_display = pd.DataFrame(recommendations)
    if not rec_display.empty:
        for column in ["return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
            rec_display[column] = rec_display[column].map(fmt_pct)
        for column in ["flow_score", "theme_score", "composite_score", "volume_zscore"]:
            rec_display[column] = rec_display[column].map(lambda x: f"{x:.1f}")
        st.dataframe(
            rec_display[
                [
                    "ticker",
                    "themes",
                    "composite_score",
                    "flow_score",
                    "theme_score",
                    "return_5d",
                    "return_20d",
                    "relative_5d",
                    "dollar_volume_trend",
                    "volume_zscore",
                    "reason",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Selected Watchlist Components")
    st.caption("Your added stock universe is shown here directly, independent of flow ranking position.")
    watchlist_df = component_df[component_df["ticker"].isin(WATCHLIST_TICKERS)].copy()
    watchlist_found = set(watchlist_df["ticker"])
    watch_cols = st.columns(3)
    watch_cols[0].metric("Selected Tickers", len(WATCHLIST_TICKERS))
    watch_cols[1].metric("Loaded From Yahoo", len(watchlist_found))
    watch_cols[2].metric("Missing", len(set(WATCHLIST_TICKERS) - watchlist_found))
    missing_watchlist = sorted(set(WATCHLIST_TICKERS) - watchlist_found)
    if missing_watchlist:
        st.warning(f"Missing market data: {', '.join(missing_watchlist)}")
    if not watchlist_df.empty:
        watchlist_display = format_component_table(watchlist_df)
        st.dataframe(
            watchlist_display[
                [
                    "ticker",
                    "themes",
                    "last_price",
                    "flow_score",
                    "return_5d",
                    "return_20d",
                    "relative_5d",
                    "dollar_volume_m",
                    "dollar_volume_trend",
                    "volume_zscore",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns([1.4, 1])
    with left:
        fig = px.bar(
            theme_df,
            x="theme",
            y="flow_score",
            color="flow_score",
            hover_data=["description", "components", "return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_m"],
            color_continuous_scale=["#f85149", "#d29922", "#2ea043"],
            range_color=[0, 100],
            title="Thematic Money Flow Score",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", height=430)
        fig.update_xaxes(tickangle=-35)
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

    st.subheader("Theme Table")
    display_df = watchlist_theme_labels(theme_df)
    for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
        display_df[column] = display_df[column].map(fmt_pct)
    display_df["flow_score"] = display_df["flow_score"].map(lambda x: f"{x:.1f}")
    display_df["dollar_volume_m"] = display_df["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Components", "Sector ETFs", "Benchmarks"])
    with tab1:
        show_watchlist_only = st.checkbox("Show selected watchlist only", value=True)
        component_source = component_df[component_df["ticker"].isin(WATCHLIST_TICKERS)] if show_watchlist_only else component_df
        component_display = format_component_table(component_source)
        st.dataframe(component_display, use_container_width=True, hide_index=True)

    with tab2:
        sector_display = sector_df.copy()
        for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
            sector_display[column] = sector_display[column].map(fmt_pct)
        sector_display["flow_score"] = sector_display["flow_score"].map(lambda x: f"{x:.1f}")
        sector_display["dollar_volume_m"] = sector_display["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
        st.dataframe(sector_display, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Benchmark Pulse")
        if not bench_df.empty:
            bench_display = bench_df.copy()
            for column in ["return_1d", "return_5d", "return_20d"]:
                bench_display[column] = bench_display[column].map(fmt_pct)
            st.dataframe(bench_display, use_container_width=True, hide_index=True)

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

    st.caption("Research tool only. Theme flow scores are proxies derived from price and volume, not official fund-flow data.")


if __name__ == "__main__":
    main()

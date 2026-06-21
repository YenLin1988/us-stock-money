from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from us_stock_money import scoring as scoring_module
from us_stock_money.alerts import evaluate_alerts
from us_stock_money.congress_trades import DISPLAY_COLUMNS, download_congress_trades, filter_congress_trades
from us_stock_money.insider_trades import (
    DISPLAY_COLUMNS as INSIDER_DISPLAY_COLUMNS,
    download_insider_trades,
    filter_insider_trades,
    summarize_insider_trades,
)
from us_stock_money.market_data import (
    benchmark_table,
    build_component_table,
    build_intraday_component_table,
    build_intraday_market_table,
    build_sector_table,
    build_theme_table,
    build_weekly_theme_trends,
    download_intraday_component_prices,
    download_intraday_prices,
    download_prices,
)
from us_stock_money.model_config import MARKET_DATA_VERSION, WATCHLIST_TICKERS
from us_stock_money.scoring import (
    broad_flow_score,
    build_breakout_candidates,
    build_intraday_breakout_candidates,
    build_top_recommendations,
    classify_regime,
    flow_delta,
    intraday_market_signal,
    market_timing_signal,
    theme_group_scores,
)
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
    .price-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        margin-top: 0.65rem;
        font-size: 0.86rem;
    }
    .exit-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin-top: 0.75rem;
        padding-top: 0.65rem;
        border-top: 1px solid #30363d;
        font-size: 0.86rem;
    }
    .exit-signal {
        border-radius: 999px;
        padding: 2px 9px;
        font-weight: 700;
        white-space: nowrap;
    }
    .exit-hold { color: #3fb950; background: rgba(63, 185, 80, 0.12); }
    .exit-watch { color: #d29922; background: rgba(210, 153, 34, 0.14); }
    .exit-trim { color: #d29922; background: rgba(210, 153, 34, 0.14); }
    .exit-exit { color: #f85149; background: rgba(248, 81, 73, 0.12); }
    .positive-pct { color: #3fb950; font-weight: 700; }
    .negative-pct { color: #f85149; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900)
def load_data(
    market_data_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = download_prices()
    return (
        build_theme_table(data),
        build_component_table(data),
        build_sector_table(data),
        benchmark_table(data),
        build_weekly_theme_trends(data),
    )


@st.cache_data(ttl=300)
def load_intraday_data(market_data_version: str) -> pd.DataFrame:
    data = download_intraday_prices(period="5d", interval="5m")
    return build_intraday_market_table(data)


@st.cache_data(ttl=300)
def load_intraday_component_data(market_data_version: str) -> pd.DataFrame:
    data = download_intraday_component_prices(period="5d", interval="5m")
    return build_intraday_component_table(data)


@st.cache_data(ttl=3600)
def load_congress_trade_data() -> pd.DataFrame:
    return download_congress_trades()


@st.cache_data(ttl=3600)
def load_insider_trade_data() -> pd.DataFrame:
    return download_insider_trades()


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def format_component_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
        display[column] = display[column].map(fmt_pct)
    display["flow_score"] = display["flow_score"].map(lambda x: f"{x:.1f}")
    display["dollar_volume_m"] = display["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
    return display


def format_intraday_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ["day_return", "return_30m", "return_60m", "volume_trend"]:
        display[column] = display[column].map(fmt_pct)
    for column in ["last_price", "session_open", "vwap"]:
        display[column] = display[column].map(lambda x: f"{x:,.2f}")
    display["below_vwap"] = display["below_vwap"].map(lambda x: "Yes" if x else "No")
    return display


def watchlist_theme_labels(theme_df: pd.DataFrame) -> pd.DataFrame:
    watchlist = set(WATCHLIST_TICKERS)
    display = theme_df.copy()
    display["selected_watchlist"] = display["components"].map(
        lambda value: ", ".join([ticker for ticker in str(value).split(", ") if ticker in watchlist])
    )
    return display


def pct_color_class(value: float) -> str:
    return "positive-pct" if value >= 0 else "negative-pct"


def fmt_price(value: float) -> str:
    return f"${value:,.2f}"


def exit_signal_class(value: object) -> str:
    return {
        "Hold": "exit-hold",
        "Watch": "exit-watch",
        "Trim": "exit-trim",
        "Exit": "exit-exit",
    }.get(str(value), "exit-watch")


def rating_class(value: object) -> str:
    return {
        "High Conviction": "exit-hold",
        "Positive": "exit-hold",
        "Neutral": "exit-watch",
        "Caution": "exit-exit",
    }.get(str(value), "exit-watch")


def configure_auto_refresh() -> None:
    with st.sidebar:
        st.subheader("Auto Refresh")
        enabled = st.toggle("Enable auto refresh", value=True)
        interval_seconds = st.selectbox(
            "Refresh interval",
            options=[60, 180, 300, 600, 900],
            index=2,
            format_func=lambda seconds: f"{seconds // 60} min",
            disabled=not enabled,
        )
        st.caption(f"Last rendered: {dt.datetime.now().strftime('%H:%M:%S')}")
    if enabled:
        components.html(
            f"""
            <script>
            window.setTimeout(function() {{
                window.parent.location.reload();
            }}, {interval_seconds * 1000});
            </script>
            """,
            height=0,
        )


def fmt_dollar_compact(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def apply_intraday_prices(recommendations: list[dict[str, object]], intraday_prices: pd.DataFrame) -> list[dict[str, object]]:
    intraday_by_ticker = {} if intraday_prices.empty else intraday_prices.set_index("ticker").to_dict("index")
    enriched = []
    for rec in recommendations:
        item = dict(rec)
        intraday = intraday_by_ticker.get(str(item.get("ticker", "")))
        if intraday:
            item["open_price"] = float(intraday.get("open_price", intraday.get("session_open", 0.0)))
            item["last_price"] = float(intraday["last_price"])
            item["open_to_current_pct"] = float(
                intraday.get("open_to_current_pct", intraday.get("day_return", 0.0))
            )
            item["last_time"] = intraday["last_time"]
            item["price_source"] = "5m"
        else:
            item["price_source"] = "1d"
        enriched.append(item)
    return enriched


def build_integrated_recommendations(*args, **kwargs) -> list[dict[str, object]]:
    builder = getattr(scoring_module, "build_integrated_recommendations", None)
    if builder is not None:
        return builder(*args, **kwargs)

    component_rows, theme_scores = args[:2]
    limit = int(kwargs.get("limit", 5))
    fallback = build_top_recommendations(component_rows, theme_scores, limit=limit)
    return [
        {
            **item,
            "integrated_score": float(item["composite_score"]),
            "rating": "Neutral",
            "momentum_score": 50.0,
            "intraday_score": 50.0,
            "congress_score": 50.0,
            "insider_score": 50.0,
            "market_score": float(kwargs.get("market_score", 50.0)),
            "exit_signal": "Watch",
            "congress_buys": 0,
            "congress_sales": 0,
            "insider_buys": 0,
            "insider_sales": 0,
        }
        for item in fallback
    ]


def render_page_header(title: str, caption: str) -> None:
    configure_auto_refresh()
    st.title(title)
    st.caption(caption)
    if st.button("Refresh data", type="secondary"):
        st.cache_data.clear()
        st.rerun()


def load_market_page_context() -> dict[str, object] | None:
    try:
        theme_df, component_df, sector_df, bench_df, weekly_theme_df = load_data(MARKET_DATA_VERSION)
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
        return None
    try:
        intraday_df = load_intraday_data(MARKET_DATA_VERSION)
    except Exception:
        intraday_df = pd.DataFrame()
    try:
        intraday_component_df = load_intraday_component_data(MARKET_DATA_VERSION)
    except Exception:
        intraday_component_df = pd.DataFrame()

    theme_scores = dict(zip(theme_df["theme"], theme_df["flow_score"], strict=False))
    groups = theme_group_scores(theme_scores)
    broad = broad_flow_score(theme_scores)
    risk_on = groups.get("AI Compute Chain", 0.0)
    defensive = groups.get("Healthcare / Automation", 0.0)
    return {
        "theme_df": theme_df,
        "component_df": component_df,
        "sector_df": sector_df,
        "bench_df": bench_df,
        "weekly_theme_df": weekly_theme_df,
        "intraday_df": intraday_df,
        "intraday_component_df": intraday_component_df,
        "theme_scores": theme_scores,
        "groups": groups,
        "broad": broad,
        "risk_on": risk_on,
        "defensive": defensive,
        "regime": classify_regime(broad, risk_on, defensive),
        "timing_signal": market_timing_signal(bench_df, broad, risk_on),
        "intraday_signal": intraday_market_signal(intraday_df),
    }


def load_disclosure_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        congress_df = load_congress_trade_data()
    except Exception as exc:
        st.warning(f"Congress trade data is temporarily unavailable: {exc}")
        congress_df = pd.DataFrame()
    try:
        insider_df = load_insider_trade_data()
    except Exception as exc:
        st.warning(f"SEC insider trade data is temporarily unavailable: {exc}")
        insider_df = pd.DataFrame()
    return congress_df, insider_df


def recommendations_page() -> None:
    render_page_header(
        "Integrated Recommendations",
        "Combined ranking across flow, theme strength, momentum, 5m setup, disclosures, and market timing.",
    )
    context = load_market_page_context()
    if context is None:
        return
    congress_df, insider_df = load_disclosure_context()
    recent_congress_df = filter_congress_trades(congress_df, days=90) if not congress_df.empty else congress_df
    candidates = build_integrated_recommendations(
        context["component_df"],
        context["theme_scores"],
        context["intraday_component_df"],
        recent_congress_df,
        insider_df,
        market_score=context["timing_signal"].score,
        limit=10,
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Market Timing", f"{context['timing_signal'].score:.0f}/100")
    metric2.metric("Broad Flow", f"{context['broad']:.1f}/100")
    metric3.metric("Risk-On Flow", f"{context['risk_on']:.1f}/100")
    metric4.metric("High Conviction", sum(item["rating"] == "High Conviction" for item in candidates))

    top_candidates = candidates[:5]
    candidate_cols = st.columns(5)
    for index, candidate in enumerate(top_candidates, start=1):
        rating = str(candidate["rating"])
        with candidate_cols[index - 1]:
            st.markdown(
                f"""
                <div class="flow-card">
                    <div class="small-label">#{index} Integrated</div>
                    <h3 style="margin: 0.2rem 0 0.1rem 0;">{candidate["ticker"]}</h3>
                    <div class="small-label">{candidate["themes"]}</div>
                    <p style="font-size: 1.35rem; margin: 0.6rem 0 0.2rem 0;">{float(candidate["integrated_score"]):.1f}</p>
                    <div class="small-label">Integrated score</div>
                    <div class="price-row">
                        <span class="small-label">{fmt_price(float(candidate["open_price"]))} -> {fmt_price(float(candidate["last_price"]))}</span>
                        <span class="{pct_color_class(float(candidate["open_to_current_pct"]))}">{fmt_pct(float(candidate["open_to_current_pct"]))}</span>
                    </div>
                    <div class="exit-row">
                        <span class="small-label">{candidate["exit_signal"]}</span>
                        <span class="exit-signal {rating_class(rating)}">{rating}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    display = pd.DataFrame(candidates)
    score_columns = [
        "integrated_score",
        "flow_score",
        "theme_score",
        "momentum_score",
        "intraday_score",
        "congress_score",
        "insider_score",
        "market_score",
    ]
    columns = [
        "ticker",
        "themes",
        "integrated_score",
        "rating",
        *score_columns[1:],
        "exit_signal",
        "congress_buys",
        "congress_sales",
        "insider_buys",
        "insider_sales",
        "reason",
    ]
    styled = display[columns].style.format({column: "{:.1f}" for column in score_columns}).map(
        lambda value: {
            "High Conviction": "color: #3fb950; font-weight: 700",
            "Positive": "color: #3fb950; font-weight: 700",
            "Neutral": "color: #d29922; font-weight: 700",
            "Caution": "color: #f85149; font-weight: 700",
        }.get(value, ""),
        subset=["rating"],
    )
    st.dataframe(styled, width="stretch", hide_index=True)
    with st.expander("Scoring methodology"):
        st.write(
            "Flow 25%, theme 15%, daily momentum 15%, 5m setup 20%, Congress 10%, "
            "corporate insider activity 10%, market timing 5%. Trim and Exit signals apply risk deductions."
        )


def signals_page() -> None:
    render_page_header(
        "Market Signals",
        "Intraday market regime, 5-minute breakouts, exit signals, and daily flow candidates.",
    )
    context = load_market_page_context()
    if context is None:
        return

    timing_signal = context["timing_signal"]
    intraday_signal = context["intraday_signal"]
    timing_col, intraday_col = st.columns(2)
    with timing_col:
        st.subheader("Market Timing")
        st.metric("Timing Score", f"{timing_signal.score:.0f}/100")
        st.write(f"**{timing_signal.title}**")
        st.caption(timing_signal.message)
    with intraday_col:
        st.subheader("5m Market State")
        st.metric("Intraday Score", f"{intraday_signal.score:.0f}/100")
        st.write(f"**{intraday_signal.title}**")
        st.caption(intraday_signal.message)

    if not context["intraday_df"].empty:
        st.dataframe(format_intraday_table(context["intraday_df"]), width="stretch", hide_index=True)

    st.subheader("5m Breakout Candidates")
    breakouts = build_intraday_breakout_candidates(context["intraday_component_df"], limit=20)
    if breakouts:
        breakout_df = pd.DataFrame(breakouts)
        st.dataframe(
            breakout_df[
                [
                    "ticker",
                    "themes",
                    "breakout_score",
                    "exit_signal",
                    "day_return",
                    "return_30m",
                    "return_60m",
                    "vwap_gap_pct",
                    "volume_trend",
                    "exit_reason",
                    "reason",
                ]
            ].style.format(
                {
                    "breakout_score": "{:.1f}",
                    "day_return": fmt_pct,
                    "return_30m": fmt_pct,
                    "return_60m": fmt_pct,
                    "vwap_gap_pct": fmt_pct,
                    "volume_trend": fmt_pct,
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("5m component data is unavailable.")

    st.subheader("Daily Flow Candidates")
    daily_candidates = build_top_recommendations(context["component_df"], context["theme_scores"], limit=20)
    st.dataframe(pd.DataFrame(daily_candidates), width="stretch", hide_index=True)


def disclosures_page() -> None:
    render_page_header(
        "Disclosures",
        "Congressional STOCK Act filings and SEC Form 4/4-A open-market insider transactions.",
    )
    congress_df, insider_df = load_disclosure_context()

    st.subheader("Congress Stock Trades")
    if congress_df.empty:
        st.info("No congressional disclosure data is available.")
    else:
        filter1, filter2, filter3 = st.columns([0.8, 1.2, 1])
        days = filter1.selectbox("Lookback", [30, 90, 180, 365], index=1, format_func=lambda value: f"{value} days")
        sides = filter2.multiselect("Transaction", ["Purchase", "Sale", "Exchange", "Other"], default=["Purchase", "Sale"])
        ticker = filter3.text_input("Congress ticker", placeholder="NVDA")
        congress_display = filter_congress_trades(congress_df, days=days, sides=sides, ticker=ticker)
        metrics = st.columns(3)
        metrics[0].metric("Trades", len(congress_display))
        metrics[1].metric("Purchases", int((congress_display["trade_side"] == "Purchase").sum()))
        metrics[2].metric("Sales", int((congress_display["trade_side"] == "Sale").sum()))
        st.dataframe(
            congress_display[DISPLAY_COLUMNS + ["trade_side"]],
            width="stretch",
            hide_index=True,
            column_config={"doc_url": st.column_config.LinkColumn("Official Filing", display_text="Open filing")},
        )

    st.subheader("Corporate Insider Trades")
    if insider_df.empty:
        st.info("No SEC insider disclosure data is available.")
    else:
        filter1, filter2 = st.columns([1.2, 1])
        sides = filter1.multiselect("Insider transaction", ["Purchase", "Sale"], default=["Purchase", "Sale"])
        ticker = filter2.text_input("Insider ticker", placeholder="MSFT")
        insider_display = filter_insider_trades(insider_df, sides=sides, ticker=ticker)
        summary = summarize_insider_trades(insider_display)
        metrics = st.columns(5)
        metrics[0].metric("Buy Shares", f"{summary['purchase_shares']:,.0f}")
        metrics[1].metric("Sell Shares", f"{summary['sale_shares']:,.0f}")
        metrics[2].metric("Buy Value", fmt_dollar_compact(summary["purchase_value"]))
        metrics[3].metric("Sell Value", fmt_dollar_compact(summary["sale_value"]))
        metrics[4].metric("Net Value", fmt_dollar_compact(summary["net_value"]))
        st.dataframe(
            insider_display[INSIDER_DISPLAY_COLUMNS],
            width="stretch",
            hide_index=True,
            column_config={"filing_url": st.column_config.LinkColumn("SEC Filing", display_text="Open filing")},
        )


def research_page() -> None:
    render_page_header(
        "Market Research",
        "Theme rotation, selected watchlist, sector references, benchmarks, and component-level data.",
    )
    context = load_market_page_context()
    if context is None:
        return

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Regime", context["regime"].name)
    metric2.metric("Broad Flow", f"{context['broad']:.1f}/100")
    metric3.metric("AI Compute", f"{context['risk_on']:.1f}/100")
    metric4.metric("Healthcare / Automation", f"{context['defensive']:.1f}/100")

    left, right = st.columns([1.4, 1])
    with left:
        fig = px.bar(
            context["theme_df"],
            x="theme",
            y="flow_score",
            color="flow_score",
            color_continuous_scale=["#f85149", "#d29922", "#2ea043"],
            range_color=[0, 100],
            title="Thematic Money Flow Score",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", height=430)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, width="stretch")
    with right:
        radar = go.Figure(
            go.Scatterpolar(
                r=list(context["groups"].values()),
                theta=list(context["groups"].keys()),
                fill="toself",
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
        st.plotly_chart(radar, width="stretch")

    st.subheader("Weekly Money Flow and Sector Moves")
    st.caption(
        "Weekly money flow is a price-and-volume proxy built from equal-weight theme returns, "
        "relative strength versus SPY, and weekly dollar-volume trend."
    )
    weekly_df = context["weekly_theme_df"]
    if weekly_df.empty:
        st.info("Weekly trend data is unavailable.")
    else:
        week_window = st.segmented_control(
            "History window",
            options=[8, 13, 26, 52],
            default=13,
            format_func=lambda value: f"{value} weeks",
        )
        available_weeks = sorted(weekly_df["week"].dropna().unique())
        selected_weeks = available_weeks[-int(week_window or 13):]
        weekly_display = weekly_df[weekly_df["week"].isin(selected_weeks)].copy()
        weekly_display["week_label"] = weekly_display["week"].dt.strftime("%Y-%m-%d")

        latest_week = weekly_display["week"].max()
        latest = weekly_display[weekly_display["week"] == latest_week]
        top_inflow = latest.nlargest(3, "net_flow")
        top_outflow = latest.nsmallest(3, "net_flow")
        top_gainers = latest.nlargest(3, "weekly_return")
        top_losers = latest.nsmallest(3, "weekly_return")

        summary_cols = st.columns(4)
        summary_groups = [
            ("Strongest Inflow", top_inflow, "net_flow"),
            ("Strongest Outflow", top_outflow, "net_flow"),
            ("Biggest Weekly Gains", top_gainers, "weekly_return"),
            ("Biggest Weekly Losses", top_losers, "weekly_return"),
        ]
        for column, (title, frame, value_column) in zip(summary_cols, summary_groups, strict=False):
            with column:
                st.markdown(f"**{title}**")
                for row in frame.itertuples():
                    value = float(getattr(row, value_column))
                    st.markdown(
                        f"`{row.theme}` <span class=\"{pct_color_class(value)}\">{value:+.1f}{'%' if value_column == 'weekly_return' else ''}</span>",
                        unsafe_allow_html=True,
                    )

        flow_pivot = weekly_display.pivot(index="theme", columns="week_label", values="net_flow")
        return_pivot = weekly_display.pivot(index="theme", columns="week_label", values="weekly_return")
        flow_heatmap = px.imshow(
            flow_pivot,
            color_continuous_scale=["#f85149", "#111820", "#3fb950"],
            color_continuous_midpoint=0,
            aspect="auto",
            labels={"color": "Net flow"},
            title="Weekly Money Flow Direction (-50 Outflow to +50 Inflow)",
        )
        flow_heatmap.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(flow_heatmap, width="stretch")

        return_heatmap = px.imshow(
            return_pivot,
            color_continuous_scale=["#f85149", "#111820", "#3fb950"],
            color_continuous_midpoint=0,
            aspect="auto",
            labels={"color": "Weekly return %"},
            title="Weekly Theme Returns",
        )
        return_heatmap.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(return_heatmap, width="stretch")

        with st.expander("Weekly trend detail"):
            detail = weekly_display[
                [
                    "week",
                    "theme",
                    "flow_score",
                    "net_flow",
                    "weekly_return",
                    "relative_return",
                    "volume_trend",
                    "dollar_volume_m",
                    "component_count",
                ]
            ].sort_values(["week", "flow_score"], ascending=[False, False])
            st.dataframe(
                detail.style.format(
                    {
                        "flow_score": "{:.1f}",
                        "net_flow": "{:+.1f}",
                        "weekly_return": "{:+.2f}%",
                        "relative_return": "{:+.2f}%",
                        "volume_trend": "{:+.1f}%",
                        "dollar_volume_m": "${:,.0f}M",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    st.subheader("Selected Watchlist")
    watchlist_df = context["component_df"][context["component_df"]["ticker"].isin(WATCHLIST_TICKERS)]
    st.dataframe(format_component_table(watchlist_df), width="stretch", hide_index=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Themes", "Components", "Sectors", "Benchmarks"])
    with tab1:
        st.dataframe(watchlist_theme_labels(context["theme_df"]), width="stretch", hide_index=True)
    with tab2:
        st.dataframe(format_component_table(context["component_df"]), width="stretch", hide_index=True)
    with tab3:
        st.dataframe(context["sector_df"], width="stretch", hide_index=True)
    with tab4:
        st.dataframe(context["bench_df"], width="stretch", hide_index=True)


def main() -> None:
    configure_auto_refresh()

    st.title("US STOCK MONEY")
    st.caption("US thematic money-flow radar: AI compute chain, power, defense, space, rare earths, nuclear, medical, and other rotation themes.")

    if st.button("Refresh market data", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    try:
        theme_df, component_df, sector_df, bench_df, weekly_theme_df = load_data(MARKET_DATA_VERSION)
    except Exception as exc:  # pragma: no cover - Streamlit runtime display
        st.error(f"Could not load market data: {exc}")
        return
    try:
        intraday_df = load_intraday_data(MARKET_DATA_VERSION)
    except Exception:
        intraday_df = pd.DataFrame()
    try:
        intraday_component_df = load_intraday_component_data(MARKET_DATA_VERSION)
    except Exception:
        intraday_component_df = pd.DataFrame()
    congress_error = ""
    try:
        congress_df = load_congress_trade_data()
    except Exception as exc:
        congress_df = pd.DataFrame()
        congress_error = str(exc)
    insider_error = ""
    try:
        insider_df = load_insider_trade_data()
    except Exception as exc:
        insider_df = pd.DataFrame()
        insider_error = str(exc)

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
    timing_signal = market_timing_signal(bench_df, broad, risk_on)
    intraday_signal = intraday_market_signal(intraday_df)

    alerts = evaluate_alerts(
        {
            "broad_flow_score": broad,
            "risk_on_score": risk_on,
            "defensive_score": defensive,
            "delta_24h": delta_24h,
            "regime": regime.name,
            "market_timing_status": timing_signal.status,
            "market_timing_title": timing_signal.title,
            "market_timing_message": timing_signal.message,
            "intraday_status": intraday_signal.status,
            "intraday_title": intraday_signal.title,
            "intraday_message": intraday_signal.message,
        }
    )
    recommendations = build_top_recommendations(component_df, theme_scores, limit=5)
    intraday_breakout_candidates = build_intraday_breakout_candidates(intraday_component_df, limit=5)
    daily_breakout_candidates = build_breakout_candidates(component_df, limit=5)
    recommendations = apply_intraday_prices(recommendations, intraday_component_df)
    recent_congress_df = filter_congress_trades(congress_df, days=90) if not congress_df.empty else congress_df
    integrated_recommendations = build_integrated_recommendations(
        component_df,
        theme_scores,
        intraday_component_df,
        recent_congress_df,
        insider_df,
        market_score=timing_signal.score,
        limit=5,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime", regime.name)
    col2.metric("Broad Flow", f"{broad:.1f}/100", None if delta_24h is None else f"{delta_24h:+.1f} 24H")
    col3.metric("AI Compute Chain", f"{risk_on:.1f}/100")
    col4.metric("Healthcare / Automation", f"{defensive:.1f}/100")

    st.divider()

    st.subheader("Market Timing Signal")
    if timing_signal.status == "stand_aside":
        st.error(f"**{timing_signal.title}** - {timing_signal.message}")
    elif timing_signal.status == "recovery_confirmed":
        st.success(f"**{timing_signal.title}** - {timing_signal.message}")
    else:
        st.warning(f"**{timing_signal.title}** - {timing_signal.message}")
    with st.expander("Timing signal evidence", expanded=False):
        st.metric("Timing Score", f"{timing_signal.score:.0f}/100")
        for item in timing_signal.evidence:
            st.caption(item)

    st.subheader("5m Intraday Market Monitor")
    if intraday_signal.status == "intraday_stand_aside":
        st.error(f"**{intraday_signal.title}** - {intraday_signal.message}")
    elif intraday_signal.status == "intraday_recovery":
        st.success(f"**{intraday_signal.title}** - {intraday_signal.message}")
    else:
        st.warning(f"**{intraday_signal.title}** - {intraday_signal.message}")
    if not intraday_df.empty:
        st.dataframe(format_intraday_table(intraday_df), width="stretch", hide_index=True)
    with st.expander("5m intraday evidence", expanded=False):
        st.metric("Intraday Timing Score", f"{intraday_signal.score:.0f}/100")
        for item in intraday_signal.evidence:
            st.caption(item)

    st.subheader("Top 5 Integrated Recommendations")
    st.caption(
        "Combined score: flow 25%, theme 15%, daily momentum 15%, 5m setup 20%, "
        "Congress disclosures 10%, corporate insider trades 10%, and market timing 5%. "
        "Disclosure data is delayed and used only as a supporting signal."
    )
    integrated_cols = st.columns(5)
    for index, candidate in enumerate(integrated_recommendations, start=1):
        rating = str(candidate["rating"])
        with integrated_cols[index - 1]:
            st.markdown(
                f"""
                <div class="flow-card">
                    <div class="small-label">#{index} Integrated</div>
                    <h3 style="margin: 0.2rem 0 0.1rem 0;">{candidate["ticker"]}</h3>
                    <div class="small-label">{candidate["themes"]}</div>
                    <p style="font-size: 1.35rem; margin: 0.6rem 0 0.2rem 0;">{float(candidate["integrated_score"]):.1f}</p>
                    <div class="small-label">Integrated score</div>
                    <div class="price-row">
                        <span class="small-label">{fmt_price(float(candidate["open_price"]))} -> {fmt_price(float(candidate["last_price"]))}</span>
                        <span class="{pct_color_class(float(candidate["open_to_current_pct"]))}">{fmt_pct(float(candidate["open_to_current_pct"]))}</span>
                    </div>
                    <div class="exit-row">
                        <span class="small-label">{int(candidate["congress_buys"])}B/{int(candidate["congress_sales"])}S Congress · {int(candidate["insider_buys"])}B/{int(candidate["insider_sales"])}S insider</span>
                        <span class="exit-signal {rating_class(rating)}">{rating}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    integrated_display = pd.DataFrame(integrated_recommendations)
    if not integrated_display.empty:
        integrated_columns = [
            "ticker",
            "themes",
            "integrated_score",
            "rating",
            "flow_score",
            "theme_score",
            "momentum_score",
            "intraday_score",
            "congress_score",
            "insider_score",
            "market_score",
            "exit_signal",
            "congress_buys",
            "congress_sales",
            "insider_buys",
            "insider_sales",
            "reason",
        ]
        factor_columns = [
            "integrated_score",
            "flow_score",
            "theme_score",
            "momentum_score",
            "intraday_score",
            "congress_score",
            "insider_score",
            "market_score",
        ]
        integrated_styled = integrated_display[integrated_columns].style.format(
            {column: "{:.1f}" for column in factor_columns}
        ).map(
            lambda value: {
                "High Conviction": "color: #3fb950; font-weight: 700",
                "Positive": "color: #3fb950; font-weight: 700",
                "Neutral": "color: #d29922; font-weight: 700",
                "Caution": "color: #f85149; font-weight: 700",
            }.get(value, ""),
            subset=["rating"],
        )
        st.dataframe(integrated_styled, width="stretch", hide_index=True)

    st.subheader("Top 5 5m Breakout Candidates")
    if intraday_breakout_candidates:
        st.caption("Ranked from 5-minute candles: session move, 30/60 minute momentum, VWAP position, and live volume trend.")
        breakout_cols = st.columns(5)
        for index, candidate in enumerate(intraday_breakout_candidates, start=1):
            day_return = float(candidate["day_return"])
            signal = str(candidate["exit_signal"])
            signal_class = exit_signal_class(signal)
            with breakout_cols[index - 1]:
                st.markdown(
                    f"""
                    <div class="flow-card">
                        <div class="small-label">#{index} 5m Breakout</div>
                        <h3 style="margin: 0.2rem 0 0.1rem 0;">{candidate["ticker"]}</h3>
                        <div class="small-label">{candidate["themes"]}</div>
                        <p style="font-size: 1.35rem; margin: 0.6rem 0 0.2rem 0;">{float(candidate["breakout_score"]):.1f}</p>
                        <div class="small-label">5m breakout score</div>
                        <div class="price-row">
                            <span class="small-label">{fmt_price(float(candidate["session_open"]))} -> {fmt_price(float(candidate["last_price"]))}</span>
                            <span class="{pct_color_class(day_return)}">{fmt_pct(day_return)}</span>
                        </div>
                        <div class="small-label">Session open -> latest</div>
                        <div class="exit-row">
                            <span class="small-label">Exit signal</span>
                            <span class="exit-signal {signal_class}">{signal}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        breakout_display = pd.DataFrame(intraday_breakout_candidates)
        breakout_columns = [
            "ticker",
            "themes",
            "last_time",
            "session_open",
            "last_price",
            "day_return",
            "return_30m",
            "return_60m",
            "vwap_gap_pct",
            "volume_trend",
            "recent_dollar_volume_m",
            "breakout_score",
            "exit_signal",
            "exit_reason",
            "reason",
        ]
        pct_columns = ["day_return", "return_30m", "return_60m", "vwap_gap_pct", "volume_trend"]
        breakout_styled = breakout_display[breakout_columns].style.format(
            {
                "session_open": fmt_price,
                "last_price": fmt_price,
                "recent_dollar_volume_m": "${:,.0f}M",
                "breakout_score": "{:.1f}",
                **{column: fmt_pct for column in pct_columns},
            }
        )
        breakout_styled = breakout_styled.map(
            lambda value: "color: #3fb950; font-weight: 700" if value >= 0 else "color: #f85149; font-weight: 700",
            subset=["day_return", "return_30m", "return_60m", "vwap_gap_pct"],
        )
        breakout_styled = breakout_styled.map(
            lambda value: {
                "Hold": "color: #3fb950; font-weight: 700",
                "Watch": "color: #d29922; font-weight: 700",
                "Trim": "color: #d29922; font-weight: 700",
                "Exit": "color: #f85149; font-weight: 700",
            }.get(value, ""),
            subset=["exit_signal"],
        )
        st.dataframe(
            breakout_styled,
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("5m individual-stock data is unavailable right now; showing daily breakout fallback.")
        breakout_display = pd.DataFrame(daily_breakout_candidates)
        if not breakout_display.empty:
            st.dataframe(breakout_display, width="stretch", hide_index=True)

    st.subheader("Top 5 Flow Candidates")
    st.caption("Ranked by component money-flow score plus related theme strength. Research signal only, not financial advice.")
    rec_cols = st.columns(5)
    for col, rec in zip(rec_cols, recommendations, strict=False):
        open_to_current_pct = float(rec["open_to_current_pct"])
        with col:
            st.markdown(
                f"""
                <div class="flow-card">
                    <div class="small-label">#{recommendations.index(rec) + 1} Flow Candidate</div>
                    <h3 style="margin: 0.2rem 0 0.1rem 0;">{rec["ticker"]}</h3>
                    <div class="small-label">{rec["themes"]}</div>
                    <p style="font-size: 1.35rem; margin: 0.6rem 0 0.2rem 0;">{float(rec["composite_score"]):.1f}</p>
                    <div class="small-label">Composite score</div>
                    <div class="price-row">
                        <span class="small-label">{fmt_price(float(rec["open_price"]))} -> {fmt_price(float(rec["last_price"]))}</span>
                        <span class="{pct_color_class(open_to_current_pct)}">{fmt_pct(open_to_current_pct)}</span>
                    </div>
                    <div class="small-label">{rec.get("price_source", "1d")} Open -> Current</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    rec_display = pd.DataFrame(recommendations)
    if not rec_display.empty:
        rec_columns = [
            "ticker",
            "themes",
            "open_price",
            "last_price",
            "open_to_current_pct",
            "price_source",
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
        pct_columns = ["open_to_current_pct", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]
        score_columns = ["flow_score", "theme_score", "composite_score", "volume_zscore"]
        rec_styled = rec_display[rec_columns].style.format(
            {
                "open_price": fmt_price,
                "last_price": fmt_price,
                **{column: fmt_pct for column in pct_columns},
                **{column: "{:.1f}" for column in score_columns},
            }
        )
        rec_styled = rec_styled.map(
            lambda value: "color: #3fb950; font-weight: 700" if value >= 0 else "color: #f85149; font-weight: 700",
            subset=["open_to_current_pct"],
        )
        st.dataframe(
            rec_styled,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Congress Stock Trades")
    st.caption(
        "Recent STOCK Act disclosures from House and Senate filings. "
        "Transactions may be reported up to 45 days after the trade date."
    )
    if congress_error:
        st.warning(f"Congress trade data is temporarily unavailable: {congress_error}")

    if not congress_df.empty:
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([0.8, 1.2, 1.2, 1])
        with filter_col1:
            congress_days = st.selectbox("Lookback", [30, 90, 180, 365], index=1, format_func=lambda value: f"{value} days")
        with filter_col2:
            congress_chambers = st.multiselect(
                "Chamber",
                sorted(congress_df["chamber"].dropna().unique()),
                default=sorted(congress_df["chamber"].dropna().unique()),
            )
        with filter_col3:
            congress_sides = st.multiselect(
                "Transaction",
                ["Purchase", "Sale", "Exchange", "Other"],
                default=["Purchase", "Sale"],
            )
        with filter_col4:
            congress_ticker = st.text_input("Ticker", placeholder="NVDA")

        congress_display = filter_congress_trades(
            congress_df,
            days=congress_days,
            chambers=congress_chambers,
            sides=congress_sides,
            ticker=congress_ticker,
        )
        congress_metric1, congress_metric2, congress_metric3, congress_metric4 = st.columns(4)
        congress_metric1.metric("Disclosed Trades", len(congress_display))
        congress_metric2.metric("Purchases", int((congress_display["trade_side"] == "Purchase").sum()))
        congress_metric3.metric("Sales", int((congress_display["trade_side"] == "Sale").sum()))
        latest_filing = congress_display["filing_date"].max()
        congress_metric4.metric("Latest Filing", "N/A" if pd.isna(latest_filing) else latest_filing.strftime("%Y-%m-%d"))

        if congress_display.empty:
            st.info("No congressional trades match the selected filters.")
        else:
            congress_table = congress_display[DISPLAY_COLUMNS + ["trade_side"]].copy()
            congress_table["transaction_date"] = congress_table["transaction_date"].dt.strftime("%Y-%m-%d")
            congress_table["filing_date"] = congress_table["filing_date"].dt.strftime("%Y-%m-%d")
            congress_table["days_to_file"] = congress_table["days_to_file"].map(
                lambda value: "" if pd.isna(value) else f"{int(value)}"
            )
            congress_styled = congress_table.style.map(
                lambda value: (
                    "color: #3fb950; font-weight: 700"
                    if value == "Purchase"
                    else "color: #f85149; font-weight: 700"
                    if value == "Sale"
                    else ""
                ),
                subset=["trade_side"],
            )
            st.dataframe(
                congress_styled,
                width="stretch",
                hide_index=True,
                column_config={
                    "doc_url": st.column_config.LinkColumn("Official Filing", display_text="Open filing"),
                    "trade_side": "Side",
                },
            )
        st.caption(
            "Source: normalized public STOCK Act filings from the House Clerk and Senate eFD. "
            "Trade amounts are disclosed as ranges, not exact values."
        )

    st.subheader("Corporate Insider Trades")
    st.caption(
        "Latest SEC Form 4/4-A open-market transactions. Form 10-K filings are skipped. "
        "Only transaction codes P (purchase) and S (sale) are counted; awards, option exercises, gifts, and tax withholding are excluded."
    )
    if insider_error:
        st.warning(f"SEC insider trade data is temporarily unavailable: {insider_error}")

    if not insider_df.empty:
        insider_filter1, insider_filter2 = st.columns([1.2, 1])
        with insider_filter1:
            insider_sides = st.multiselect(
                "Insider transaction",
                ["Purchase", "Sale"],
                default=["Purchase", "Sale"],
            )
        with insider_filter2:
            insider_ticker = st.text_input("Insider ticker", placeholder="NVDA")

        insider_display = filter_insider_trades(
            insider_df,
            sides=insider_sides,
            ticker=insider_ticker,
        )
        insider_summary = summarize_insider_trades(insider_display)
        insider_metric1, insider_metric2, insider_metric3, insider_metric4, insider_metric5 = st.columns(5)
        insider_metric1.metric("Buy Shares", f"{insider_summary['purchase_shares']:,.0f}")
        insider_metric2.metric("Sell Shares", f"{insider_summary['sale_shares']:,.0f}")
        insider_metric3.metric("Buy Value", fmt_dollar_compact(insider_summary["purchase_value"]))
        insider_metric4.metric("Sell Value", fmt_dollar_compact(insider_summary["sale_value"]))
        insider_metric5.metric(
            "Net Insider Value",
            fmt_dollar_compact(insider_summary["net_value"]),
            delta="Net buying" if insider_summary["net_value"] >= 0 else "Net selling",
            delta_color="normal",
        )

        if insider_display.empty:
            st.info("No open-market insider trades match the selected filters.")
        else:
            insider_table = insider_display[INSIDER_DISPLAY_COLUMNS].copy()
            insider_table["transaction_date"] = insider_table["transaction_date"].dt.strftime("%Y-%m-%d")
            insider_table["filing_time"] = insider_table["filing_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
            insider_styled = insider_table.style.format(
                {
                    "shares": "{:,.0f}",
                    "price_per_share": "${:,.2f}",
                    "estimated_value": "${:,.0f}",
                    "shares_after": "{:,.0f}",
                }
            ).map(
                lambda value: (
                    "color: #3fb950; font-weight: 700"
                    if value == "Purchase"
                    else "color: #f85149; font-weight: 700"
                    if value == "Sale"
                    else ""
                ),
                subset=["trade_side"],
            )
            st.dataframe(
                insider_styled,
                width="stretch",
                hide_index=True,
                column_config={
                    "filing_url": st.column_config.LinkColumn("SEC Filing", display_text="Open filing"),
                    "trade_side": "Side",
                },
            )
        st.caption(
            "Source: SEC EDGAR Form 4 filings. Values are estimated as reported shares multiplied by reported price per share. "
            "The feed is cached for one hour and covers the latest SEC filings, not a fixed historical window."
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
            width="stretch",
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(radar, width="stretch")

    st.subheader("Theme Table")
    display_df = watchlist_theme_labels(theme_df)
    for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
        display_df[column] = display_df[column].map(fmt_pct)
    display_df["flow_score"] = display_df["flow_score"].map(lambda x: f"{x:.1f}")
    display_df["dollar_volume_m"] = display_df["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
    st.dataframe(display_df, width="stretch", hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Components", "Sector ETFs", "Benchmarks"])
    with tab1:
        show_watchlist_only = st.checkbox("Show selected watchlist only", value=True)
        component_source = component_df[component_df["ticker"].isin(WATCHLIST_TICKERS)] if show_watchlist_only else component_df
        component_display = format_component_table(component_source)
        st.dataframe(component_display, width="stretch", hide_index=True)

    with tab2:
        sector_display = sector_df.copy()
        for column in ["return_1d", "return_5d", "return_20d", "relative_5d", "dollar_volume_trend"]:
            sector_display[column] = sector_display[column].map(fmt_pct)
        sector_display["flow_score"] = sector_display["flow_score"].map(lambda x: f"{x:.1f}")
        sector_display["dollar_volume_m"] = sector_display["dollar_volume_m"].map(lambda x: f"${x:,.0f}M")
        st.dataframe(sector_display, width="stretch", hide_index=True)

    with tab3:
        st.subheader("Benchmark Pulse")
        if not bench_df.empty:
            bench_display = bench_df.copy()
            for column in ["return_1d", "return_5d", "return_20d"]:
                bench_display[column] = bench_display[column].map(fmt_pct)
            st.dataframe(bench_display, width="stretch", hide_index=True)

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
            st.plotly_chart(hist_fig, width="stretch")

    st.caption("Research tool only. Theme flow scores are proxies derived from price and volume, not official fund-flow data.")


if __name__ == "__main__":
    navigation = st.navigation(
        {
            "Dashboard": [
                st.Page(main, title="Overview", url_path="overview", default=True),
                st.Page(recommendations_page, title="Recommendations", url_path="recommendations"),
                st.Page(signals_page, title="Signals", url_path="signals"),
            ],
            "Research": [
                st.Page(disclosures_page, title="Disclosures", url_path="disclosures"),
                st.Page(research_page, title="Market Research", url_path="research"),
            ],
        },
        position="sidebar",
        expanded=True,
    )
    navigation.run()

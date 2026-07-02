from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from us_stock_money import scoring as scoring_module
from us_stock_money.analyst_data import (
    download_analyst_data,
    download_peer_valuations,
    pe_comparison_summary,
    themes_for_ticker,
)
from us_stock_money.alerts import evaluate_alerts
from us_stock_money.congress_trades import (
    DISPLAY_COLUMNS,
    aggregate_congress_by_ticker,
    download_congress_trades,
    filter_congress_trades,
    summarize_congress_trades,
)
from us_stock_money.dark_pool import add_market_context, build_ats_anomalies, download_ats_weekly_activity
from us_stock_money.insider_trades import (
    DISPLAY_COLUMNS as INSIDER_DISPLAY_COLUMNS,
    aggregate_insider_by_ticker,
    download_insider_trades,
    download_ticker_insider_trades,
    filter_insider_trades,
    normalize_insider_trades,
    summarize_insider_trades,
)
from us_stock_money.market_data import (
    benchmark_table,
    build_component_table,
    build_intraday_component_table,
    build_intraday_market_table,
    build_intraday_theme_table,
    build_sector_table,
    build_theme_table,
    build_weekly_theme_trends,
    download_intraday_component_prices,
    download_intraday_prices,
    download_prices,
)
from us_stock_money.pick_tracker import evaluate_intraday_picks, pick_hit_rate_summary
from us_stock_money.model_config import ALL_TICKERS, MARKET_DATA_VERSION, WATCHLIST_TICKERS
from us_stock_money.scoring import (
    broad_flow_score,
    build_breakout_candidates,
    build_intraday_breakout_candidates,
    build_top_recommendations,
    classify_regime,
    flow_delta,
    intraday_entry_gate,
    intraday_market_signal,
    market_timing_signal,
    theme_group_scores,
)
from us_stock_money.storage import HistoryStore
from us_stock_money.technical_analysis import build_ma60_alerts, build_stock_detail, stock_snapshot


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
    .decision-banner {
        border-left: 4px solid #58a6ff;
        padding: 16px 20px;
        margin: 0.5rem 0 1.25rem 0;
        background: #111820;
    }
    .decision-positive { border-left-color: #3fb950; }
    .decision-warning { border-left-color: #d29922; }
    .decision-danger { border-left-color: #f85149; }
    .decision-title { font-size: 1.05rem; font-weight: 700; color: #f0f6fc; }
    .decision-copy { color: #b1bac4; margin-top: 4px; }
    .recommend-card, .risk-card {
        min-height: 225px;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        background: #111820;
    }
    .recommend-card { border-top: 3px solid #3fb950; }
    .risk-card { border-top: 3px solid #f85149; }
    .card-score { font-size: 1.7rem; font-weight: 700; color: #f0f6fc; margin: 0.55rem 0 0; }
    .card-reason { color: #b1bac4; font-size: 0.86rem; margin-top: 0.75rem; line-height: 1.4; }
    .stock-analysis-link { color: #f0f6fc; text-decoration: none; }
    .stock-analysis-link:hover { color: #58a6ff; text-decoration: underline; }
    .analysis-action {
        display: inline-block;
        color: #58a6ff;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.75rem;
        text-decoration: none;
    }
    .analysis-action:hover { text-decoration: underline; }
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


@st.cache_data(ttl=120)
def load_intraday_data(market_data_version: str) -> pd.DataFrame:
    data = download_intraday_prices(period="5d", interval="5m")
    return build_intraday_market_table(data)


@st.cache_data(ttl=120)
def load_intraday_component_data(market_data_version: str) -> pd.DataFrame:
    data = download_intraday_component_prices(period="5d", interval="5m")
    return build_intraday_component_table(data)


@st.cache_data(ttl=900)
def load_technical_data(market_data_version: str) -> pd.DataFrame:
    return download_prices(period="1y", interval="1d")


@st.cache_data(ttl=3600)
def load_analyst_data(ticker: str) -> tuple[dict[str, object], pd.DataFrame]:
    return download_analyst_data(ticker)


@st.cache_data(ttl=3600)
def load_peer_valuation_data(ticker: str, theme: str) -> pd.DataFrame:
    return download_peer_valuations(ticker, theme)


@st.cache_data(ttl=21600)
def load_ats_activity_data(market_data_version: str) -> pd.DataFrame:
    return download_ats_weekly_activity(ALL_TICKERS)


@st.cache_data(ttl=3600)
def load_congress_trade_data() -> pd.DataFrame:
    return download_congress_trades()


@st.cache_data(ttl=3600)
def load_insider_trade_data() -> pd.DataFrame:
    return download_insider_trades()


@st.cache_data(ttl=3600)
def load_ticker_insider_trade_data(ticker: str) -> pd.DataFrame:
    return download_ticker_insider_trades(ticker)


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


def fmt_optional_multiple(value: object) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.1f}x"


def fmt_optional_price(value: object, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{float(value):,.2f}"


def stock_analysis_url(ticker: object) -> str:
    return f"/stock-analysis?ticker={str(ticker).strip().upper()}"


def add_stock_analysis_links(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["analysis_url"] = display["ticker"].map(stock_analysis_url)
    return display


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


def insider_value_display(value: float, currencies: set[str]) -> str:
    currencies = {currency for currency in currencies if currency}
    if currencies == {"USD"} or not currencies:
        return fmt_dollar_compact(value)
    if currencies == {"EUR"}:
        absolute = abs(value)
        if absolute >= 1_000_000:
            return f"EUR {value / 1_000_000:,.2f}M"
        if absolute >= 1_000:
            return f"EUR {value / 1_000:,.1f}K"
        return f"EUR {value:,.0f}"
    return "Mixed currencies"


def disclosure_style(value: object) -> str:
    if value in {"Purchase", "Net Buying"}:
        return "color: #3fb950; font-weight: 700"
    if value in {"Sale", "Net Selling"}:
        return "color: #f85149; font-weight: 700"
    if value == "Mixed":
        return "color: #d29922; font-weight: 700"
    return ""


def net_value_style(value: object) -> str:
    if pd.isna(value):
        return ""
    return "color: #3fb950; font-weight: 700" if float(value) >= 0 else "color: #f85149; font-weight: 700"


def render_disclosure_activity_chart(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
) -> None:
    if frame.empty:
        return
    chart_data = (
        frame.assign(_magnitude=frame[value_column].abs())
        .nlargest(12, "_magnitude")
        .drop(columns="_magnitude")
    )
    chart_data = chart_data.sort_values(value_column)
    chart = px.bar(
        chart_data,
        x=value_column,
        y="ticker",
        orientation="h",
        color=value_column,
        color_continuous_scale=["#f85149", "#111820", "#3fb950"],
        color_continuous_midpoint=0,
        title=title,
    )
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        height=max(330, len(chart_data) * 34),
        coloraxis_showscale=False,
        xaxis_title="Net selling ← Estimated value → Net buying",
        yaxis_title="",
        margin={"l": 20, "r": 20, "t": 50, "b": 35},
    )
    st.plotly_chart(chart, width="stretch")


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


def build_risk_watchlist(integrated_rows, limit: int = 5) -> list[dict[str, object]]:
    builder = getattr(scoring_module, "build_risk_watchlist", None)
    if builder is not None:
        return builder(integrated_rows, limit=limit)
    fallback = sorted(integrated_rows, key=lambda item: float(item.get("integrated_score", 50.0)))[:limit]
    return [
        {
            **item,
            "risk_score": 100.0 - float(item.get("integrated_score", 50.0)),
            "risk_level": "Watch",
            "risk_reason": "Integrated score is relatively weak.",
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
    intraday_theme_df = (
        build_intraday_theme_table(intraday_component_df) if not intraday_component_df.empty else pd.DataFrame()
    )
    return {
        "theme_df": theme_df,
        "component_df": component_df,
        "sector_df": sector_df,
        "bench_df": bench_df,
        "weekly_theme_df": weekly_theme_df,
        "intraday_df": intraday_df,
        "intraday_component_df": intraday_component_df,
        "intraday_theme_df": intraday_theme_df,
        "theme_scores": theme_scores,
        "groups": groups,
        "broad": broad,
        "risk_on": risk_on,
        "defensive": defensive,
        "regime": classify_regime(broad, risk_on, defensive),
        "timing_signal": market_timing_signal(bench_df, broad, risk_on),
        "intraday_signal": intraday_market_signal(intraday_df),
        "entry_gate": intraday_entry_gate(intraday_df, intraday_theme_df),
    }


def taipei_time(last_time: object) -> str:
    """Render an exchange timestamp alongside Taipei local time."""
    try:
        timestamp = pd.Timestamp(str(last_time))
    except (ValueError, TypeError):
        return str(last_time)
    if timestamp.tzinfo is None:
        return str(last_time)
    return timestamp.tz_convert("Asia/Taipei").strftime("%m-%d %H:%M 台北")


def is_live_session(last_time: object) -> bool:
    """True when the latest 5m bar is recent enough to be today's live session."""
    try:
        timestamp = pd.Timestamp(str(last_time))
    except (ValueError, TypeError):
        return False
    if timestamp.tzinfo is None:
        return False
    now = pd.Timestamp.now(tz=timestamp.tzinfo)
    return pd.Timedelta(0) <= (now - timestamp) <= pd.Timedelta(minutes=30)


def render_entry_gate(gate) -> None:
    """Prominent go / no-go banner for tonight's session."""
    if gate.status == "no_entry":
        st.error(f"🚫 **{gate.title}** — {gate.message}")
    elif gate.status == "caution":
        st.warning(f"⚠️ **{gate.title}** — {gate.message}")
    elif gate.status == "ok":
        st.success(f"✅ **{gate.title}** — {gate.message}")
    else:
        st.info(f"**{gate.title}** — {gate.message}")
    if gate.evidence:
        with st.expander("進場判斷依據", expanded=gate.status == "no_entry"):
            for item in gate.evidence:
                st.caption(item)


def render_intraday_theme_board(theme_df: pd.DataFrame) -> None:
    """Theme-level intraday money-flow board: where capital is moving right now."""
    st.subheader("盤中主題資金流向 (5m)")
    if theme_df is None or theme_df.empty:
        st.info("盤中主題資料暫時無法取得。")
        return
    chart_df = theme_df.sort_values("vs_spy_pct")
    fig = px.bar(
        chart_df,
        x="vs_spy_pct",
        y="theme",
        orientation="h",
        color="vs_spy_pct",
        color_continuous_scale=["#f85149", "#8b949e", "#3fb950"],
        color_continuous_midpoint=0.0,
        labels={"vs_spy_pct": "當日相對 SPY (%)", "theme": ""},
        title="主題當日表現 vs SPY（含跳空）",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        height=420,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")
    display = theme_df[
        [
            "theme",
            "intraday_flow_score",
            "day_change_pct",
            "vs_spy_pct",
            "gap_pct",
            "rvol",
            "pct_above_vwap",
            "component_count",
            "top_component",
            "weak_component",
        ]
    ].rename(
        columns={
            "theme": "主題",
            "intraday_flow_score": "盤中流向分數",
            "day_change_pct": "當日漲跌%",
            "vs_spy_pct": "vs SPY%",
            "gap_pct": "跳空%",
            "rvol": "同時段相對量",
            "pct_above_vwap": "站上VWAP比例%",
            "component_count": "成分數",
            "top_component": "最強",
            "weak_component": "最弱",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "盤中流向分數": "{:.1f}",
                "當日漲跌%": fmt_pct,
                "vs SPY%": fmt_pct,
                "跳空%": fmt_pct,
                "同時段相對量": "{:.2f}x",
                "站上VWAP比例%": "{:.0f}%",
            }
        ).map(
            lambda value: "color: #3fb950; font-weight: 700" if value >= 0 else "color: #f85149; font-weight: 700",
            subset=["當日漲跌%", "vs SPY%", "跳空%"],
        ),
        width="stretch",
        hide_index=True,
    )


def log_intraday_picks(candidates: list[dict[str, object]]) -> int:
    """Archive today's top live candidates once per session for hit-rate tracking."""
    live = [candidate for candidate in candidates if is_live_session(candidate.get("last_time"))]
    if not live:
        return 0
    picks = [
        {
            "pick_date": str(candidate.get("last_time", ""))[:10],
            "ticker": str(candidate.get("ticker", "")),
            "pick_price": float(candidate.get("last_price", 0.0)),
            "breakout_score": float(candidate.get("breakout_score", 0.0)),
            "themes": str(candidate.get("themes", "")),
        }
        for candidate in live[:5]
    ]
    store = HistoryStore(Path("data/flow_history.sqlite3"))
    return store.save_intraday_picks(picks, picked_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))


def render_pick_tracker() -> None:
    """Show archived intraday picks and their realized outcomes."""
    with st.expander("盤中推薦命中率追蹤", expanded=False):
        st.caption(
            "開盤時段載入儀表板時會自動記錄當日前 5 名盤中候選（每日每檔只記第一次），"
            "之後可對照當日與次日收盤驗證這套訊號在你的操作時段是否有正期望值。"
        )
        store = HistoryStore(Path("data/flow_history.sqlite3"))
        if st.button("更新驗證結果", key="evaluate_picks"):
            with st.spinner("下載日線收盤資料驗證中..."):
                picks_df = evaluate_intraday_picks(store)
        else:
            picks_df = pd.DataFrame(store.load_intraday_picks())
        if picks_df.empty:
            st.info("尚無紀錄。美股盤中載入儀表板後會自動記錄當日推薦。")
            return
        summary = pick_hit_rate_summary(picks_df)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("已驗證筆數", int(summary["evaluated"]))
        col2.metric("當日收盤勝率", f"{summary['win_rate']:.0f}%")
        col3.metric("當日平均報酬", f"{summary['avg_return']:+.2f}%")
        col4.metric("次日收盤勝率", f"{summary['next_day_win_rate']:.0f}%")
        col5.metric("次日平均報酬", f"{summary['next_day_avg_return']:+.2f}%")
        st.dataframe(
            picks_df.style.format(
                {
                    "pick_price": fmt_price,
                    "breakout_score": "{:.1f}",
                    "close_price": fmt_price,
                    "close_return_pct": fmt_pct,
                    "next_close_price": fmt_price,
                    "next_close_return_pct": fmt_pct,
                },
                na_rep="待驗證",
            ),
            width="stretch",
            hide_index=True,
        )


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


def decision_dashboard_page() -> None:
    render_page_header(
        "US STOCK MONEY",
        "A focused decision dashboard for recommendations, capital flow, and stocks that currently deserve caution.",
    )
    context = load_market_page_context()
    if context is None:
        return
    congress_df, insider_df = load_disclosure_context()
    recent_congress_df = filter_congress_trades(congress_df, days=90) if not congress_df.empty else congress_df
    all_candidates = build_integrated_recommendations(
        context["component_df"],
        context["theme_scores"],
        context["intraday_component_df"],
        recent_congress_df,
        insider_df,
        market_score=context["timing_signal"].score,
        limit=len(context["component_df"]),
    )
    recommended = [
        candidate
        for candidate in all_candidates
        if candidate["rating"] in {"High Conviction", "Positive"}
        and candidate["exit_signal"] not in {"Trim", "Exit"}
    ][:5]
    if len(recommended) < 3:
        recommended = [
            candidate for candidate in all_candidates if candidate["exit_signal"] not in {"Trim", "Exit"}
        ][:5]
    risks = build_risk_watchlist(all_candidates, limit=5)

    render_entry_gate(context["entry_gate"])

    timing_signal = context["timing_signal"]
    banner_class = (
        "decision-danger"
        if timing_signal.status == "stand_aside"
        else "decision-positive"
        if timing_signal.status == "recovery_confirmed"
        else "decision-warning"
    )
    st.markdown(
        f"""
        <div class="decision-banner {banner_class}">
            <div class="decision-title">{timing_signal.title}</div>
            <div class="decision-copy">{timing_signal.message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Market Timing", f"{timing_signal.score:.0f}/100")
    metric2.metric("Broad Money Flow", f"{context['broad']:.1f}/100")
    metric3.metric("Positive Candidates", len(recommended))
    metric4.metric("High-Risk Flags", sum(item["risk_level"] in {"Avoid", "High Risk"} for item in risks))

    render_intraday_theme_board(context["intraday_theme_df"])
    log_intraday_picks(build_intraday_breakout_candidates(context["intraday_component_df"], limit=5))

    st.subheader("Recommended Now")
    st.caption("Candidates with supportive flow and momentum, without an active 5m Trim or Exit signal.")
    if recommended:
        recommendation_cols = st.columns(min(3, len(recommended)))
        for index, candidate in enumerate(recommended[:3], start=1):
            with recommendation_cols[index - 1]:
                st.markdown(
                    f"""
                    <div class="recommend-card">
                        <div class="small-label">#{index} {candidate["rating"]}</div>
                        <h2 style="margin: 0.25rem 0 0;"><a class="stock-analysis-link" href="{stock_analysis_url(candidate["ticker"])}" target="_self">{candidate["ticker"]}</a></h2>
                        <div class="small-label">{candidate["themes"]}</div>
                        <div class="card-score">{float(candidate["integrated_score"]):.1f}</div>
                        <div class="price-row">
                            <span class="small-label">{fmt_price(float(candidate["last_price"]))}</span>
                            <span class="{pct_color_class(float(candidate["open_to_current_pct"]))}">{fmt_pct(float(candidate["open_to_current_pct"]))}</span>
                        </div>
                        <div class="card-reason">{candidate["reason"]}</div>
                        <a class="analysis-action" href="{stock_analysis_url(candidate["ticker"])}" target="_self">View technical analysis</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with st.expander("See recommendation factor details"):
            recommendation_df = add_stock_analysis_links(pd.DataFrame(recommended))
            st.dataframe(
                recommendation_df[
                    [
                        "ticker",
                        "analysis_url",
                        "integrated_score",
                        "rating",
                        "flow_score",
                        "theme_score",
                        "momentum_score",
                        "intraday_score",
                        "congress_score",
                        "insider_score",
                        "exit_signal",
                        "reason",
                    ]
                ].style.format(
                    {
                        "integrated_score": "{:.1f}",
                        "flow_score": "{:.1f}",
                        "theme_score": "{:.1f}",
                        "momentum_score": "{:.1f}",
                        "intraday_score": "{:.1f}",
                        "congress_score": "{:.1f}",
                        "insider_score": "{:.1f}",
                    }
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
                },
            )
    else:
        st.warning("No stock currently passes the recommendation and risk filters.")

    st.subheader("Avoid or Reduce Risk")
    st.caption("Stocks with weak combined factors, active Trim/Exit signals, or disclosure selling pressure.")
    if risks:
        risk_cols = st.columns(min(3, len(risks)))
        for index, candidate in enumerate(risks[:3]):
            with risk_cols[index]:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="small-label">{candidate["risk_level"]}</div>
                        <h2 style="margin: 0.25rem 0 0;"><a class="stock-analysis-link" href="{stock_analysis_url(candidate["ticker"])}" target="_self">{candidate["ticker"]}</a></h2>
                        <div class="small-label">{candidate.get("themes", "")}</div>
                        <div class="card-score">{float(candidate["risk_score"]):.1f}</div>
                        <div class="small-label">Risk score · 5m {candidate["exit_signal"]}</div>
                        <div class="card-reason">{candidate["risk_reason"]}</div>
                        <a class="analysis-action" href="{stock_analysis_url(candidate["ticker"])}" target="_self">View technical analysis</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with st.expander("See full risk watchlist"):
            risk_df = add_stock_analysis_links(pd.DataFrame(risks))
            st.dataframe(
                risk_df[
                    [
                        "ticker",
                        "analysis_url",
                        "risk_score",
                        "risk_level",
                        "integrated_score",
                        "flow_score",
                        "momentum_score",
                        "intraday_score",
                        "exit_signal",
                        "risk_reason",
                    ]
                ].style.format(
                    {
                        "risk_score": "{:.1f}",
                        "integrated_score": "{:.1f}",
                        "flow_score": "{:.1f}",
                        "momentum_score": "{:.1f}",
                        "intraday_score": "{:.1f}",
                    }
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
                },
            )
    else:
        st.success("No high-risk stocks were identified in the current universe.")

    st.subheader("Where Money Is Moving This Week")
    weekly_df = context["weekly_theme_df"]
    if weekly_df.empty:
        st.info("Weekly money-flow data is unavailable.")
    else:
        latest_week = weekly_df["week"].max()
        latest = weekly_df[weekly_df["week"] == latest_week].sort_values("net_flow")
        flow_chart = px.bar(
            latest,
            x="net_flow",
            y="theme",
            orientation="h",
            color="net_flow",
            color_continuous_scale=["#f85149", "#111820", "#3fb950"],
            color_continuous_midpoint=0,
            hover_data=["weekly_return", "relative_return", "volume_trend"],
            title=f"Weekly Theme Flow · Week Ending {latest_week:%Y-%m-%d}",
        )
        flow_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            height=520,
            coloraxis_showscale=False,
            xaxis_title="Outflow ← Net Flow → Inflow",
            yaxis_title="",
        )
        st.plotly_chart(flow_chart, width="stretch")
        inflow, outflow = st.columns(2)
        with inflow:
            st.markdown("**Strongest inflows**")
            st.dataframe(
                latest.nlargest(5, "net_flow")[["theme", "net_flow", "weekly_return"]].style.format(
                    {"net_flow": "{:+.1f}", "weekly_return": "{:+.2f}%"}
                ),
                width="stretch",
                hide_index=True,
            )
        with outflow:
            st.markdown("**Strongest outflows**")
            st.dataframe(
                latest.nsmallest(5, "net_flow")[["theme", "net_flow", "weekly_return"]].style.format(
                    {"net_flow": "{:+.1f}", "weekly_return": "{:+.2f}%"}
                ),
                width="stretch",
                hide_index=True,
            )

    render_pick_tracker()


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
                    <h3 style="margin: 0.2rem 0 0.1rem 0;"><a class="stock-analysis-link" href="{stock_analysis_url(candidate["ticker"])}" target="_self">{candidate["ticker"]}</a></h3>
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
                    <a class="analysis-action" href="{stock_analysis_url(candidate["ticker"])}" target="_self">View technical analysis</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    display = add_stock_analysis_links(pd.DataFrame(candidates))
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
        "analysis_url",
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
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
        },
    )
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

    render_entry_gate(context["entry_gate"])

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

    render_intraday_theme_board(context["intraday_theme_df"])

    st.subheader("5m Breakout Candidates")
    st.caption(
        "排名依據橫斷面百分位：vs SPY 30%、同時段相對量 20%、開盤後漲幅 15%、30 分鐘動能 10%、"
        "站上 VWAP 15%、突破開盤區間 10%。跳空後回落者扣分。已過濾近 30 分鐘美元成交額不足的標的。"
    )
    breakouts = build_intraday_breakout_candidates(context["intraday_component_df"], limit=20)
    log_intraday_picks(breakouts)
    if breakouts:
        breakout_df = pd.DataFrame(breakouts)
        latest_time = breakout_df["last_time"].iloc[0] if "last_time" in breakout_df else ""
        if latest_time:
            st.caption(f"最新 5m K 棒：{latest_time}（{taipei_time(latest_time)}）")
        st.dataframe(
            breakout_df[
                [
                    "ticker",
                    "themes",
                    "breakout_score",
                    "exit_signal",
                    "gap_pct",
                    "day_return",
                    "vs_spy_pct",
                    "rvol",
                    "return_30m",
                    "vwap_gap_pct",
                    "above_orb_high",
                    "entry_ref",
                    "stop_ref",
                    "exit_reason",
                    "reason",
                ]
            ].style.format(
                {
                    "breakout_score": "{:.1f}",
                    "gap_pct": fmt_pct,
                    "day_return": fmt_pct,
                    "vs_spy_pct": fmt_pct,
                    "rvol": "{:.2f}x",
                    "return_30m": fmt_pct,
                    "vwap_gap_pct": fmt_pct,
                    "entry_ref": fmt_price,
                    "stop_ref": fmt_price,
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("5m component data is unavailable.")

    render_pick_tracker()

    st.subheader("Daily Flow Candidates")
    daily_candidates = build_top_recommendations(context["component_df"], context["theme_scores"], limit=20)
    daily_display = add_stock_analysis_links(pd.DataFrame(daily_candidates))
    st.dataframe(
        daily_display,
        width="stretch",
        hide_index=True,
        column_config={
            "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
        },
    )


def disclosures_page() -> None:
    render_page_header(
        "Disclosures",
        "Readable stock-level summaries of congressional disclosures and SEC open-market insider transactions.",
    )
    congress_df, insider_df = load_disclosure_context()

    congress_tab, insider_tab = st.tabs(["Congress", "Corporate Insiders"])
    with congress_tab:
        st.caption(
            "STOCK Act transactions can be disclosed up to 45 days after trading. Estimated values below use "
            "the midpoint of each publicly reported amount range."
        )
        if congress_df.empty:
            st.info("No congressional disclosure data is available.")
        else:
            filter1, filter2, filter3, filter4 = st.columns([0.8, 1.1, 1.2, 1])
            days = filter1.selectbox(
                "Lookback",
                [30, 90, 180, 365],
                index=1,
                format_func=lambda value: f"{value} days",
                key="congress_days",
            )
            chambers = filter2.multiselect(
                "Chamber",
                sorted(congress_df["chamber"].dropna().unique()),
                default=sorted(congress_df["chamber"].dropna().unique()),
                key="congress_chambers",
            )
            sides = filter3.multiselect(
                "Transaction",
                ["Purchase", "Sale", "Exchange", "Other"],
                default=["Purchase", "Sale"],
                key="congress_sides",
            )
            ticker = filter4.text_input("Ticker", placeholder="NVDA", key="congress_ticker")
            congress_display = filter_congress_trades(
                congress_df,
                days=days,
                chambers=chambers,
                sides=sides,
                ticker=ticker,
            )
            summary = summarize_congress_trades(congress_display)
            metrics = st.columns(5)
            metrics[0].metric("Transactions", int(summary["trades"]))
            metrics[1].metric("Active Tickers", int(summary["tickers"]))
            metrics[2].metric("Estimated Buys", fmt_dollar_compact(summary["purchase_value"]))
            metrics[3].metric("Estimated Sales", fmt_dollar_compact(summary["sale_value"]))
            metrics[4].metric(
                "Estimated Net",
                fmt_dollar_compact(summary["net_value"]),
                f"{summary['net_value']:+,.0f}",
                delta_color="normal",
            )

            if congress_display.empty:
                if ticker:
                    coverage_start = congress_df["transaction_date"].min()
                    coverage_end = congress_df["transaction_date"].max()
                    st.info(
                        f"No congressional trades for {ticker.upper()} were found in the current source data "
                        f"from {coverage_start:%Y-%m-%d} through {coverage_end:%Y-%m-%d}."
                    )
                else:
                    st.info("No congressional trades match the selected filters.")
            else:
                view = st.segmented_control(
                    "Congress view",
                    ["By Stock", "Transactions"],
                    default="By Stock",
                    label_visibility="collapsed",
                )
                if view == "Transactions":
                    detail = congress_display.copy()
                    today = pd.Timestamp.now().normalize()
                    detail["days_ago"] = (today - detail["transaction_date"].dt.normalize()).dt.days
                    detail["filing_date"] = detail["filing_date"].dt.strftime("%Y-%m-%d")
                    detail["transaction_date"] = detail["transaction_date"].dt.strftime("%Y-%m-%d")
                    detail_columns = [
                        "transaction_date",
                        "days_ago",
                        "ticker",
                        "trade_side",
                        "amount_range_label",
                        "filer_name",
                        "chamber",
                        "party",
                        "state",
                        "filing_date",
                        "days_to_file",
                        "doc_url",
                    ]
                    detail_styled = detail[detail_columns].style.map(
                        disclosure_style,
                        subset=["trade_side"],
                    )
                    st.dataframe(
                        detail_styled,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "transaction_date": "Trade Date",
                            "days_ago": "Days Ago",
                            "ticker": "Ticker",
                            "trade_side": "Side",
                            "amount_range_label": "Reported Amount",
                            "filer_name": "Member",
                            "chamber": "Chamber",
                            "party": "Party",
                            "state": "State",
                            "filing_date": "Filed",
                            "days_to_file": st.column_config.NumberColumn("Filing Delay", format="%d days"),
                            "doc_url": st.column_config.LinkColumn("Source", display_text="Open filing"),
                        },
                    )
                else:
                    stock_summary = aggregate_congress_by_ticker(congress_display)
                    render_disclosure_activity_chart(
                        stock_summary,
                        value_column="estimated_net_value",
                        title="Congressional Net Activity by Stock",
                    )
                    stock_summary["analysis_url"] = stock_summary["ticker"].map(
                        lambda value: f"/stock-analysis?ticker={value}" if value in ALL_TICKERS else None
                    )
                    stock_summary["latest_trade"] = stock_summary["latest_trade"].dt.strftime("%Y-%m-%d")
                    stock_columns = [
                        "ticker",
                        "signal",
                        "trade_count",
                        "purchases",
                        "sales",
                        "estimated_buy_value",
                        "estimated_sale_value",
                        "estimated_net_value",
                        "filer_count",
                        "latest_trade",
                        "analysis_url",
                    ]
                    stock_styled = stock_summary[stock_columns].style.format(
                        {
                            "estimated_buy_value": "${:,.0f}",
                            "estimated_sale_value": "${:,.0f}",
                            "estimated_net_value": "${:+,.0f}",
                        }
                    ).map(disclosure_style, subset=["signal"]).map(
                        net_value_style,
                        subset=["estimated_net_value"],
                    )
                    st.dataframe(
                        stock_styled,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "ticker": "Ticker",
                            "signal": "Activity",
                            "trade_count": "Trades",
                            "purchases": "Buys",
                            "sales": "Sales",
                            "estimated_buy_value": "Estimated Buys",
                            "estimated_sale_value": "Estimated Sales",
                            "estimated_net_value": "Estimated Net",
                            "filer_count": "Members",
                            "latest_trade": "Latest Trade",
                            "analysis_url": st.column_config.LinkColumn("Chart", display_text="Open analysis"),
                        },
                    )

    with insider_tab:
        st.caption(
            "U.S. issuers use SEC Form 4/4-A open-market purchase and sale codes. Nokia manager transactions "
            "are supplemented from Nokia's official EU Market Abuse Regulation Article 19 releases. "
            "SEC awards, exercises, gifts, tax withholding, and Form 10-K filings are excluded."
        )
        if insider_df.empty:
            st.info("No SEC insider disclosure data is available.")
        else:
            filter1, filter2, filter3 = st.columns([1.2, 1, 1])
            sides = filter1.multiselect(
                "Transaction",
                ["Purchase", "Sale"],
                default=["Purchase", "Sale"],
                key="insider_sides",
            )
            ticker = filter2.text_input("Ticker", placeholder="MSFT", key="insider_ticker")
            minimum_value = filter3.selectbox(
                "Minimum Value",
                [0, 10_000, 50_000, 100_000, 500_000, 1_000_000],
                format_func=fmt_dollar_compact,
                key="insider_minimum_value",
            )
            insider_source = insider_df
            ticker_symbol = ticker.strip().upper()
            if ticker_symbol in ALL_TICKERS:
                try:
                    ticker_supplement = load_ticker_insider_trade_data(ticker_symbol)
                    if not ticker_supplement.empty:
                        insider_source = normalize_insider_trades(
                            pd.concat([insider_df, ticker_supplement], ignore_index=True).to_dict("records")
                        )
                except Exception as exc:
                    st.warning(f"Ticker-specific insider data is temporarily unavailable: {exc}")
            insider_display = filter_insider_trades(insider_source, sides=sides, ticker=ticker)
            insider_display = insider_display[insider_display["estimated_value"] >= minimum_value].reset_index(drop=True)
            summary = summarize_insider_trades(insider_display)
            currencies = set(insider_display["currency"].dropna().astype(str)) if "currency" in insider_display else set()
            mixed_currencies = len({currency for currency in currencies if currency}) > 1
            metrics = st.columns(5)
            metrics[0].metric("Transactions", int(summary["trades"]))
            metrics[1].metric("Active Tickers", int(summary["tickers"]))
            metrics[2].metric("Buy Value", insider_value_display(summary["purchase_value"], currencies))
            metrics[3].metric("Sale Value", insider_value_display(summary["sale_value"], currencies))
            metrics[4].metric(
                "Net Insider Value",
                insider_value_display(summary["net_value"], currencies),
                None if mixed_currencies else f"{summary['net_value']:+,.0f}",
                delta_color="normal",
            )

            if insider_display.empty:
                if ticker:
                    st.info(
                        f"No supported insider transactions for {ticker.upper()} match the selected filters. "
                        "The app checked the recent SEC feed and ticker-specific Yahoo Finance data. Foreign issuers "
                        "may report through a home-market disclosure system that is not yet machine-readable here, "
                        "so an empty result does not prove there were no transactions."
                    )
                else:
                    st.info("No supported insider trades match the selected filters.")
            else:
                if ticker_symbol in ALL_TICKERS:
                    source_names = ", ".join(sorted(insider_display["source"].dropna().astype(str).unique()))
                    st.caption(f"Sources used for {ticker_symbol}: {source_names or 'No source label available'}")
                view = st.segmented_control(
                    "Insider view",
                    ["By Stock", "Transactions"],
                    default="By Stock",
                    label_visibility="collapsed",
                )
                if view == "Transactions":
                    detail = insider_display.copy()
                    today = pd.Timestamp.now(tz="UTC").normalize()
                    detail["days_ago"] = (
                        today - detail["transaction_date"].dt.tz_localize("UTC").dt.normalize()
                    ).dt.days
                    detail["transaction_date"] = detail["transaction_date"].dt.strftime("%Y-%m-%d")
                    detail_columns = [
                        "transaction_date",
                        "days_ago",
                        "ticker",
                        "trade_side",
                        "estimated_value",
                        "owner_name",
                        "role",
                        "transaction_nature",
                        "shares",
                        "price_per_share",
                        "currency",
                        "shares_after",
                        "source",
                        "filing_url",
                    ]
                    detail_styled = detail[detail_columns].style.format(
                        {
                            "estimated_value": "{:,.0f}",
                            "shares": "{:,.0f}",
                            "price_per_share": "{:,.2f}",
                            "shares_after": "{:,.0f}",
                        }
                    ).map(disclosure_style, subset=["trade_side"])
                    st.dataframe(
                        detail_styled,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "transaction_date": "Trade Date",
                            "days_ago": "Days Ago",
                            "ticker": "Ticker",
                            "trade_side": "Side",
                            "estimated_value": "Estimated Value",
                            "owner_name": "Insider",
                            "role": "Role",
                            "transaction_nature": "Nature",
                            "shares": "Shares",
                            "price_per_share": "Price",
                            "currency": "Currency",
                            "shares_after": "Holdings After",
                            "source": "Data Source",
                            "filing_url": st.column_config.LinkColumn("Source", display_text="Official filing"),
                        },
                    )
                else:
                    stock_summary = aggregate_insider_by_ticker(insider_display)
                    render_disclosure_activity_chart(
                        stock_summary,
                        value_column="net_value",
                        title="Corporate Insider Net Activity by Stock",
                    )
                    stock_summary["analysis_url"] = stock_summary["ticker"].map(
                        lambda value: f"/stock-analysis?ticker={value}" if value in ALL_TICKERS else None
                    )
                    stock_summary["latest_trade"] = stock_summary["latest_trade"].dt.strftime("%Y-%m-%d")
                    stock_columns = [
                        "ticker",
                        "signal",
                        "trade_count",
                        "purchases",
                        "sales",
                        "buy_value",
                        "sale_value",
                        "net_value",
                        "insider_count",
                        "currencies",
                        "latest_trade",
                        "analysis_url",
                    ]
                    stock_styled = stock_summary[stock_columns].style.format(
                        {
                            "buy_value": "{:,.0f}",
                            "sale_value": "{:,.0f}",
                            "net_value": "{:+,.0f}",
                        }
                    ).map(disclosure_style, subset=["signal"]).map(
                        net_value_style,
                        subset=["net_value"],
                    )
                    st.dataframe(
                        stock_styled,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "ticker": "Ticker",
                            "signal": "Activity",
                            "trade_count": "Trades",
                            "purchases": "Buys",
                            "sales": "Sales",
                            "buy_value": "Buy Value",
                            "sale_value": "Sale Value",
                            "net_value": "Net Value",
                            "insider_count": "Insiders",
                            "currencies": "Currency",
                            "latest_trade": "Latest Trade",
                            "analysis_url": st.column_config.LinkColumn("Chart", display_text="Open analysis"),
                        },
                    )

    st.caption(
        "Disclosure data is delayed and should be treated as supporting context, not a real-time trading signal or financial advice. "
        "Search coverage combines the recent SEC feed, ticker-specific Yahoo Finance transactions, and supported official-company "
        "supplements. Nokia Article 19 transactions can be reported in USD or EUR; values are displayed in the reported currency."
    )


def dark_pool_page() -> None:
    render_page_header(
        "Dark Pool Activity",
        "Delayed FINRA ATS weekly volume anomalies across the tracked stock universe.",
    )
    st.warning(
        "FINRA ATS data is published with a two- to four-week delay and does not identify buyer- or seller-initiated direction. "
        "This page detects unusual activity, not dark-pool buy or sell signals."
    )
    try:
        ats_data = load_ats_activity_data(MARKET_DATA_VERSION)
        market_data = load_technical_data(MARKET_DATA_VERSION)
    except Exception as exc:
        st.error(f"Could not load FINRA ATS data: {exc}")
        return
    ats_context = add_market_context(ats_data, market_data)
    if ats_context.empty:
        st.info("No FINRA ATS data is available for the tracked universe.")
        return

    threshold1, threshold2, threshold3, search_col = st.columns([1, 1, 1, 1.2])
    volume_multiple = threshold1.number_input(
        "Volume Multiple",
        min_value=1.2,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )
    z_threshold = threshold2.number_input(
        "Z-Score",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
    )
    minimum_shares = threshold3.selectbox(
        "Minimum ATS Shares",
        [50_000, 100_000, 250_000, 500_000, 1_000_000],
        index=1,
        format_func=lambda value: f"{value / 1_000:,.0f}K",
    )
    ticker_search = search_col.text_input("Ticker", placeholder="NVDA").strip().upper()

    history, anomalies = build_ats_anomalies(
        ats_context,
        volume_multiple=volume_multiple,
        z_threshold=z_threshold,
        minimum_shares=minimum_shares,
    )
    if ticker_search:
        anomalies = anomalies[anomalies["ticker"].str.contains(ticker_search, regex=False)]

    latest_week = history["week"].max()
    latest_published = history["published_date"].max()
    strongest = anomalies.iloc[0] if not anomalies.empty else None
    metrics = st.columns(5)
    metrics[0].metric("Active Anomalies", len(anomalies))
    metrics[1].metric("Tracked ATS Symbols", history["ticker"].nunique())
    metrics[2].metric("Latest Trade Week", latest_week.strftime("%Y-%m-%d"))
    metrics[3].metric(
        "Published",
        "N/A" if pd.isna(latest_published) else latest_published.strftime("%Y-%m-%d"),
    )
    metrics[4].metric(
        "Strongest Signal",
        "None" if strongest is None else str(strongest["ticker"]),
        None if strongest is None else f"{float(strongest['volume_multiple']):.2f}x baseline",
    )

    st.subheader("Latest ATS Anomalies")
    if anomalies.empty:
        st.success("No tracked stock currently exceeds the selected ATS anomaly thresholds.")
    else:
        anomaly_display = anomalies.copy()
        anomaly_display["analysis_url"] = anomaly_display["ticker"].map(stock_analysis_url)
        anomaly_columns = [
            "ticker",
            "issue_name",
            "week",
            "published_date",
            "ats_shares",
            "ats_trades",
            "ats_notional",
            "baseline_median",
            "volume_multiple",
            "z_score",
            "week_change_pct",
            "ats_volume_pct",
            "weekly_return",
            "anomaly_score",
            "analysis_url",
        ]
        anomaly_styled = anomaly_display[anomaly_columns].style.format(
            {
                "week": lambda value: value.strftime("%Y-%m-%d"),
                "published_date": lambda value: value.strftime("%Y-%m-%d"),
                "ats_shares": "{:,.0f}",
                "ats_trades": "{:,.0f}",
                "ats_notional": "${:,.0f}",
                "baseline_median": "{:,.0f}",
                "volume_multiple": "{:.2f}x",
                "z_score": "{:.2f}",
                "week_change_pct": "{:+.1f}%",
                "ats_volume_pct": "{:.1f}%",
                "weekly_return": "{:+.1f}%",
                "anomaly_score": "{:.0f}",
            },
            na_rep="-",
        ).map(
            lambda value: (
                "color: #3fb950; font-weight: 700"
                if pd.notna(value) and float(value) >= 0
                else "color: #f85149; font-weight: 700"
                if pd.notna(value)
                else ""
            ),
            subset=["weekly_return"],
        )
        st.dataframe(
            anomaly_styled,
            width="stretch",
            hide_index=True,
            column_config={
                "ticker": "Ticker",
                "issue_name": "Company",
                "week": "Trade Week",
                "published_date": "FINRA Published",
                "ats_shares": "ATS Shares",
                "ats_trades": "ATS Trades",
                "ats_notional": "ATS Notional",
                "baseline_median": "8W Median",
                "volume_multiple": "vs Baseline",
                "z_score": "Z-Score",
                "week_change_pct": "Weekly Change",
                "ats_volume_pct": "ATS / Total Volume",
                "weekly_return": "Price 5D",
                "anomaly_score": "Anomaly Score",
                "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
            },
        )

    available_tickers = sorted(history["ticker"].dropna().unique())
    default_ticker = (
        str(strongest["ticker"])
        if strongest is not None
        else available_tickers[0]
    )
    selected_ticker = st.selectbox(
        "ATS history ticker",
        options=available_tickers,
        index=available_tickers.index(default_ticker),
    )
    ticker_history = history[history["ticker"] == selected_ticker].sort_values("week").copy()
    history_chart = go.Figure()
    history_chart.add_trace(
        go.Bar(
            x=ticker_history["week"],
            y=ticker_history["ats_shares"],
            name="ATS Shares",
            marker_color=ticker_history["is_anomaly"].map(
                lambda value: "#f85149" if value else "#58a6ff"
            ),
        )
    )
    history_chart.add_trace(
        go.Scatter(
            x=ticker_history["week"],
            y=ticker_history["baseline_median"],
            name="Prior 8W Median",
            line={"color": "#d29922", "width": 2},
        )
    )
    history_chart.update_layout(
        title=f"{selected_ticker} Weekly ATS Activity",
        template="plotly_dark",
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        height=440,
        xaxis_title="Trade Week",
        yaxis_title="ATS Shares",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 30},
    )
    st.plotly_chart(history_chart, width="stretch")
    st.caption(
        "Method: anomaly when ATS shares are at least the selected multiple of the prior eight-week median or "
        "the selected z-score above the prior eight-week mean, subject to the minimum-share filter. "
        "ATS / Total Volume uses Yahoo consolidated weekly volume as context."
    )
    st.caption(
        "Sources: FINRA OTC Transparency Weekly ATS Summary and Yahoo Finance market volume. "
        "FINRA data is delayed and can be revised."
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


def stock_analysis_page() -> None:
    render_page_header(
        "Stock Analysis",
        "Technical trends, valuation multiples, and dated institutional analyst price targets.",
    )
    try:
        technical_data = load_technical_data(MARKET_DATA_VERSION)
    except Exception as exc:
        st.error(f"Could not load technical data: {exc}")
        return

    alerts = build_ma60_alerts(technical_data, tickers=ALL_TICKERS, recent_sessions=5)
    if alerts.empty:
        st.info("Technical alerts are unavailable because there is not enough daily price history.")
        return

    st.subheader("MA60 Risk Alerts")
    st.caption(
        "Quarterly line means the 60-session moving average. A new breakdown means price crossed below MA60 "
        "during the latest five trading sessions and remains below it."
    )
    dangerous = alerts[alerts["below_ma60"]]
    new_breakdowns = alerts[alerts["recent_breakdown"] & alerts["below_ma60"]]
    weakest = alerts.iloc[alerts["distance_to_ma60_pct"].argmin()]
    summary = st.columns(4)
    summary[0].metric("New MA60 Breakdowns", len(new_breakdowns))
    summary[1].metric("Below MA60", len(dangerous))
    summary[2].metric("Universe", len(alerts))
    summary[3].metric(
        "Weakest vs MA60",
        str(weakest["ticker"]),
        fmt_pct(float(weakest["distance_to_ma60_pct"])),
        delta_color="normal",
    )

    filter_col, search_col = st.columns([1.6, 1])
    statuses = filter_col.multiselect(
        "Alert status",
        options=["New Breakdown", "Below MA60", "Above MA60"],
        default=["New Breakdown", "Below MA60"],
    )
    ticker_search = search_col.text_input("Ticker search", placeholder="NVDA").strip().upper()
    alert_display = alerts[alerts["status"].isin(statuses)].copy() if statuses else alerts.iloc[0:0].copy()
    if ticker_search:
        alert_display = alert_display[alert_display["ticker"].str.contains(ticker_search, regex=False)]
    alert_display["analysis_url"] = alert_display["ticker"].map(lambda ticker: f"/stock-analysis?ticker={ticker}")
    alert_display["cross_date"] = pd.to_datetime(alert_display["cross_date"]).dt.strftime("%Y-%m-%d").fillna("-")
    alert_columns = [
        "ticker",
        "status",
        "last_price",
        "ma20",
        "ma60",
        "distance_to_ma60_pct",
        "return_5d",
        "return_20d",
        "cross_date",
        "analysis_url",
    ]
    alert_styled = alert_display[alert_columns].style.format(
        {
            "last_price": "${:,.2f}",
            "ma20": "${:,.2f}",
            "ma60": "${:,.2f}",
            "distance_to_ma60_pct": "{:+.2f}%",
            "return_5d": "{:+.2f}%",
            "return_20d": "{:+.2f}%",
        },
        na_rep="-",
    ).map(
        lambda value: (
            "color: #f85149; font-weight: 700"
            if value in {"New Breakdown", "Below MA60"}
            else "color: #3fb950; font-weight: 700"
        ),
        subset=["status"],
    ).map(
        lambda value: (
            "color: #3fb950; font-weight: 700"
            if pd.notna(value) and float(value) >= 0
            else "color: #f85149; font-weight: 700"
            if pd.notna(value)
            else ""
        ),
        subset=["distance_to_ma60_pct", "return_5d", "return_20d"],
    )
    st.dataframe(
        alert_styled,
        width="stretch",
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "status": "MA60 Status",
            "last_price": "Last",
            "ma20": "MA20",
            "ma60": "MA60",
            "distance_to_ma60_pct": "vs MA60",
            "return_5d": "5D",
            "return_20d": "20D",
            "cross_date": "Last Breakdown",
            "analysis_url": st.column_config.LinkColumn("Detail", display_text="Open analysis"),
        },
    )

    st.divider()
    query_ticker = st.query_params.get("ticker", "")
    if isinstance(query_ticker, list):
        query_ticker = query_ticker[0] if query_ticker else ""
    requested_ticker = str(query_ticker).upper()
    default_ticker = requested_ticker if requested_ticker in ALL_TICKERS else "NVDA"
    ticker_col, range_col = st.columns([1, 1.5])
    selected_ticker = ticker_col.selectbox(
        "Stock",
        options=ALL_TICKERS,
        index=ALL_TICKERS.index(default_ticker),
    )
    history_window = range_col.segmented_control(
        "Chart range",
        options=["3M", "6M", "1Y"],
        default="6M",
    )
    if requested_ticker != selected_ticker:
        st.query_params["ticker"] = selected_ticker

    detail = build_stock_detail(technical_data, selected_ticker)
    snapshot = stock_snapshot(detail)
    if detail.empty or not snapshot:
        st.warning(f"No daily technical data is available for {selected_ticker}.")
        return

    selected_alert = alerts[alerts["ticker"] == selected_ticker]
    alert_status = str(selected_alert.iloc[0]["status"]) if not selected_alert.empty else "Unavailable"
    status_color = "#f85149" if alert_status in {"New Breakdown", "Below MA60"} else "#3fb950"
    st.markdown(
        f"### {selected_ticker} <span style=\"color:{status_color}; font-size:1rem;\">{alert_status}</span>",
        unsafe_allow_html=True,
    )
    metrics = st.columns(6)
    metrics[0].metric("Last Price", fmt_price(snapshot["last_price"]), fmt_pct(snapshot["daily_return"]))
    metrics[1].metric("vs MA20", fmt_pct(snapshot["ma20_gap"]))
    metrics[2].metric("vs MA60", fmt_pct(snapshot["ma60_gap"]))
    metrics[3].metric("RSI (14)", f"{snapshot['rsi14']:.1f}")
    metrics[4].metric("20D Volatility", f"{snapshot['volatility_20d']:.1f}%")
    metrics[5].metric("52W Drawdown", f"{snapshot['max_drawdown_52w']:.1f}%")
    st.caption(
        f"52-week range: {fmt_price(snapshot['low_52w'])} - {fmt_price(snapshot['high_52w'])}. "
        "MA5 is the short-term line, MA20 is the monthly line, and MA60 is the quarterly line."
    )

    st.subheader("Valuation and Analyst Targets")
    try:
        valuation, target_history = load_analyst_data(selected_ticker)
    except Exception as exc:
        valuation, target_history = {}, pd.DataFrame()
        st.warning(f"Valuation and analyst target data is temporarily unavailable: {exc}")

    if valuation:
        currency = str(valuation.get("currency") or "USD")
        valuation_metrics = st.columns(6)
        valuation_metrics[0].metric("Trailing P/E", fmt_optional_multiple(valuation.get("trailing_pe")))
        valuation_metrics[1].metric("Forward P/E", fmt_optional_multiple(valuation.get("forward_pe")))
        valuation_metrics[2].metric("Price / Sales", fmt_optional_multiple(valuation.get("price_to_sales")))
        valuation_metrics[3].metric("Price / Book", fmt_optional_multiple(valuation.get("price_to_book")))
        valuation_metrics[4].metric("EV / EBITDA", fmt_optional_multiple(valuation.get("enterprise_to_ebitda")))
        valuation_metrics[5].metric("PEG Ratio", fmt_optional_multiple(valuation.get("peg_ratio")))

        mean_target = valuation.get("target_mean")
        median_target = valuation.get("target_median")
        target_upside = (
            (float(mean_target) / snapshot["last_price"] - 1) * 100
            if mean_target is not None and snapshot["last_price"]
            else None
        )
        target_metrics = st.columns(6)
        target_metrics[0].metric("Consensus", valuation.get("recommendation") or "N/A")
        target_metrics[1].metric("Analysts", int(valuation.get("analyst_count") or 0))
        target_metrics[2].metric(
            "Mean Target",
            fmt_optional_price(mean_target, currency),
            None if target_upside is None else fmt_pct(target_upside),
        )
        target_metrics[3].metric("Median Target", fmt_optional_price(median_target, currency))
        target_metrics[4].metric("Low Target", fmt_optional_price(valuation.get("target_low"), currency))
        target_metrics[5].metric("High Target", fmt_optional_price(valuation.get("target_high"), currency))

    stock_themes = themes_for_ticker(selected_ticker)
    if stock_themes:
        st.markdown("#### P/E Compared with Theme Peers")
        peer_theme = st.selectbox(
            "Comparison theme",
            options=stock_themes,
            key=f"peer_theme_{selected_ticker}",
        )
        try:
            peer_valuations = load_peer_valuation_data(selected_ticker, peer_theme)
        except Exception as exc:
            peer_valuations = pd.DataFrame()
            st.warning(f"Peer valuation data is temporarily unavailable: {exc}")

        if peer_valuations.empty:
            st.info("No comparable P/E data is available for this theme.")
        else:
            peer_summary = pe_comparison_summary(peer_valuations, selected_ticker)
            peer_metrics = st.columns(5)
            peer_metrics[0].metric("Theme", peer_theme)
            peer_metrics[1].metric("Compared Stocks", int(peer_summary["peer_count"] or 0))
            peer_metrics[2].metric(
                "Trailing P/E",
                fmt_optional_multiple(peer_summary["trailing_pe"]),
                (
                    None
                    if peer_summary["trailing_premium_pct"] is None
                    else f"{peer_summary['trailing_premium_pct']:+.1f}% vs median"
                ),
                delta_color="inverse",
            )
            peer_metrics[3].metric(
                "Theme Median",
                fmt_optional_multiple(peer_summary["trailing_median"]),
            )
            peer_metrics[4].metric(
                "Forward P/E",
                fmt_optional_multiple(peer_summary["forward_pe"]),
                (
                    None
                    if peer_summary["forward_premium_pct"] is None
                    else f"{peer_summary['forward_premium_pct']:+.1f}% vs median"
                ),
                delta_color="inverse",
            )

            peer_chart_data = peer_valuations.melt(
                id_vars=["ticker", "selected"],
                value_vars=["trailing_pe", "forward_pe"],
                var_name="pe_type",
                value_name="pe",
            )
            peer_chart_data = peer_chart_data[pd.to_numeric(peer_chart_data["pe"], errors="coerce") > 0]
            peer_chart_data["pe_type"] = peer_chart_data["pe_type"].map(
                {"trailing_pe": "Trailing P/E", "forward_pe": "Forward P/E"}
            )
            if not peer_chart_data.empty:
                ticker_order = (
                    peer_valuations.assign(
                        _sort=pd.to_numeric(peer_valuations["forward_pe"], errors="coerce").fillna(float("inf"))
                    )
                    .sort_values("_sort")["ticker"]
                    .tolist()
                )
                peer_chart = px.bar(
                    peer_chart_data,
                    x="ticker",
                    y="pe",
                    color="pe_type",
                    barmode="group",
                    category_orders={"ticker": ticker_order},
                    color_discrete_map={
                        "Trailing P/E": "#58a6ff",
                        "Forward P/E": "#d29922",
                    },
                    title=f"{peer_theme} P/E Comparison",
                )
                peer_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0f14",
                    plot_bgcolor="#0b0f14",
                    height=430,
                    xaxis_title="",
                    yaxis_title="P/E Multiple",
                    legend_title="",
                    margin={"l": 20, "r": 20, "t": 55, "b": 30},
                )
                st.plotly_chart(peer_chart, width="stretch")

            peer_display = peer_valuations.copy()
            peer_display["position"] = peer_display["selected"].map(
                lambda value: "Selected Stock" if value else "Peer"
            )
            peer_display["analysis_url"] = peer_display["ticker"].map(stock_analysis_url)
            peer_columns = [
                "ticker",
                "position",
                "company",
                "trailing_pe",
                "forward_pe",
                "earnings_growth_pct",
                "revenue_growth_pct",
                "market_cap",
                "analysis_url",
            ]
            peer_styled = peer_display[peer_columns].style.format(
                {
                    "trailing_pe": "{:.1f}x",
                    "forward_pe": "{:.1f}x",
                    "earnings_growth_pct": "{:+.1f}%",
                    "revenue_growth_pct": "{:+.1f}%",
                    "market_cap": "${:,.0f}",
                },
                na_rep="N/A",
            ).map(
                lambda value: "color: #58a6ff; font-weight: 700" if value == "Selected Stock" else "",
                subset=["position"],
            )
            st.dataframe(
                peer_styled,
                width="stretch",
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "position": "Group",
                    "company": "Company",
                    "trailing_pe": "Trailing P/E",
                    "forward_pe": "Forward P/E",
                    "earnings_growth_pct": "Earnings Growth",
                    "revenue_growth_pct": "Revenue Growth",
                    "market_cap": "Market Cap",
                    "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
                },
            )
            st.caption(
                "Theme median uses only positive P/E values. Negative or missing P/E usually means earnings are "
                "negative or Yahoo Finance has no valid multiple, so those values are excluded from the median."
            )

    if target_history.empty:
        st.info("No dated institutional price-target records are available for this ticker.")
    else:
        display_count = st.segmented_control(
            "Institution records",
            options=[10, 25, 50],
            default=25,
            format_func=lambda value: f"Latest {value}",
        )
        analyst_display = target_history.head(int(display_count or 25)).copy()
        latest_by_firm = analyst_display.drop_duplicates("firm").head(15).sort_values("price_target")
        target_chart = px.scatter(
            latest_by_firm,
            x="price_target",
            y="firm",
            color="upside_pct",
            color_continuous_scale=["#f85149", "#d29922", "#3fb950"],
            color_continuous_midpoint=0,
            hover_data={
                "estimate_date": "|%Y-%m-%d",
                "rating": True,
                "target_action": True,
                "price_target": ":.2f",
                "upside_pct": ":.2f",
            },
            title="Latest Available Target by Institution",
        )
        target_chart.add_vline(
            x=snapshot["last_price"],
            line_dash="dash",
            line_color="#e6edf3",
            annotation_text="Current price",
        )
        target_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            height=max(380, len(latest_by_firm) * 30),
            coloraxis_colorbar_title="Upside %",
            xaxis_title=f"Price Target ({valuation.get('currency', 'USD') if valuation else 'USD'})",
            yaxis_title="",
            margin={"l": 20, "r": 20, "t": 55, "b": 30},
        )
        st.plotly_chart(target_chart, width="stretch")

        analyst_display["estimate_date"] = analyst_display["estimate_date"].dt.strftime("%Y-%m-%d")
        analyst_styled = analyst_display.style.format(
            {
                "price_target": "${:,.2f}",
                "prior_price_target": "${:,.2f}",
                "target_change_pct": "{:+.2f}%",
                "upside_pct": "{:+.2f}%",
            },
            na_rep="-",
        ).map(
            lambda value: (
                "color: #3fb950; font-weight: 700"
                if pd.notna(value) and float(value) >= 0
                else "color: #f85149; font-weight: 700"
                if pd.notna(value)
                else ""
            ),
            subset=["target_change_pct", "upside_pct"],
        )
        st.dataframe(
            analyst_styled,
            width="stretch",
            hide_index=True,
            column_config={
                "estimate_date": "Estimate Date",
                "firm": "Institution",
                "rating": "Rating",
                "action": "Rating Action",
                "target_action": "Target Action",
                "price_target": "Price Target",
                "prior_price_target": "Prior Target",
                "target_change_pct": "Target Change",
                "upside_pct": "vs Current",
            },
        )
        st.caption(
            "Source: Yahoo Finance analyst price-target and upgrade/downgrade history. Estimate Date is the "
            "recorded analyst action date, not the time this dashboard fetched the data. Coverage may be delayed or incomplete."
        )

    sessions = {"3M": 66, "6M": 132, "1Y": 252}
    chart_data = detail.tail(sessions.get(str(history_window), 132))
    price_chart = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )
    if {"open", "high", "low", "close"}.issubset(chart_data.columns):
        price_chart.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data["open"],
                high=chart_data["high"],
                low=chart_data["low"],
                close=chart_data["close"],
                name="Daily",
                increasing_line_color="#3fb950",
                decreasing_line_color="#f85149",
            ),
            row=1,
            col=1,
        )
    else:
        price_chart.add_trace(
            go.Scatter(x=chart_data.index, y=chart_data["close"], name="Close", line_color="#e6edf3"),
            row=1,
            col=1,
        )
    for column, label, color in [
        ("ma5", "MA5", "#58a6ff"),
        ("ma20", "MA20", "#d29922"),
        ("ma60", "MA60", "#bc8cff"),
    ]:
        price_chart.add_trace(
            go.Scatter(x=chart_data.index, y=chart_data[column], name=label, line={"color": color, "width": 1.6}),
            row=1,
            col=1,
        )
    volume_colors = ["#3fb950" if close >= open_price else "#f85149" for close, open_price in zip(
        chart_data["close"],
        chart_data.get("open", chart_data["close"]),
        strict=False,
    )]
    price_chart.add_trace(
        go.Bar(x=chart_data.index, y=chart_data.get("volume"), name="Volume", marker_color=volume_colors),
        row=2,
        col=1,
    )
    price_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        height=650,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    price_chart.update_yaxes(title_text="Price", row=1, col=1)
    price_chart.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(price_chart, width="stretch")

    rsi_col, macd_col = st.columns(2)
    with rsi_col:
        rsi_chart = go.Figure(go.Scatter(
            x=chart_data.index,
            y=chart_data["rsi14"],
            line_color="#58a6ff",
            name="RSI (14)",
        ))
        rsi_chart.add_hline(y=70, line_dash="dash", line_color="#f85149")
        rsi_chart.add_hline(y=30, line_dash="dash", line_color="#3fb950")
        rsi_chart.update_layout(
            title="RSI (14)",
            template="plotly_dark",
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            height=300,
            yaxis_range=[0, 100],
            margin={"l": 20, "r": 20, "t": 45, "b": 20},
        )
        st.plotly_chart(rsi_chart, width="stretch")
    with macd_col:
        histogram_colors = chart_data["macd_histogram"].map(lambda value: "#3fb950" if value >= 0 else "#f85149")
        macd_chart = go.Figure()
        macd_chart.add_trace(go.Bar(
            x=chart_data.index,
            y=chart_data["macd_histogram"],
            marker_color=histogram_colors,
            name="Histogram",
        ))
        macd_chart.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data["macd"],
            line_color="#58a6ff",
            name="MACD",
        ))
        macd_chart.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data["macd_signal"],
            line_color="#d29922",
            name="Signal",
        ))
        macd_chart.update_layout(
            title="MACD (12, 26, 9)",
            template="plotly_dark",
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            height=300,
            margin={"l": 20, "r": 20, "t": 45, "b": 20},
        )
        st.plotly_chart(macd_chart, width="stretch")

    st.caption(
        "Technical indicators use adjusted Yahoo Finance daily prices and may be delayed or incomplete. "
        "This is a research alert, not financial advice."
    )


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
    intraday_theme_df = (
        build_intraday_theme_table(intraday_component_df) if not intraday_component_df.empty else pd.DataFrame()
    )
    entry_gate = intraday_entry_gate(intraday_df, intraday_theme_df)

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

    st.subheader("今晚可以進場嗎？")
    render_entry_gate(entry_gate)

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

    render_intraday_theme_board(intraday_theme_df)

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
                    <h3 style="margin: 0.2rem 0 0.1rem 0;"><a class="stock-analysis-link" href="{stock_analysis_url(candidate["ticker"])}" target="_self">{candidate["ticker"]}</a></h3>
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
                    <a class="analysis-action" href="{stock_analysis_url(candidate["ticker"])}" target="_self">View technical analysis</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    integrated_display = add_stock_analysis_links(pd.DataFrame(integrated_recommendations))
    if not integrated_display.empty:
        integrated_columns = [
            "ticker",
            "analysis_url",
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
        st.dataframe(
            integrated_styled,
            width="stretch",
            hide_index=True,
            column_config={
                "analysis_url": st.column_config.LinkColumn("Analysis", display_text="Open chart"),
            },
        )

    st.subheader("Top 5 5m Breakout Candidates")
    log_intraday_picks(intraday_breakout_candidates)
    if intraday_breakout_candidates:
        st.caption(
            "橫斷面排名：vs SPY、同時段相對量 (RVOL)、開盤後漲幅、30 分鐘動能、VWAP 與開盤區間位置。"
            "entry_ref = max(開盤區間高點, VWAP)，stop_ref = 開盤區間低點，僅供參考。"
        )
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
                        <h3 style="margin: 0.2rem 0 0.1rem 0;"><a class="stock-analysis-link" href="{stock_analysis_url(candidate["ticker"])}" target="_self">{candidate["ticker"]}</a></h3>
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
            "gap_pct",
            "day_return",
            "vs_spy_pct",
            "rvol",
            "return_30m",
            "vwap_gap_pct",
            "above_orb_high",
            "entry_ref",
            "stop_ref",
            "recent_dollar_volume_m",
            "breakout_score",
            "exit_signal",
            "exit_reason",
            "reason",
        ]
        pct_columns = ["gap_pct", "day_return", "vs_spy_pct", "return_30m", "vwap_gap_pct"]
        breakout_styled = breakout_display[breakout_columns].style.format(
            {
                "session_open": fmt_price,
                "last_price": fmt_price,
                "rvol": "{:.2f}x",
                "entry_ref": fmt_price,
                "stop_ref": fmt_price,
                "recent_dollar_volume_m": "${:,.0f}M",
                "breakout_score": "{:.1f}",
                **{column: fmt_pct for column in pct_columns},
            }
        )
        breakout_styled = breakout_styled.map(
            lambda value: "color: #3fb950; font-weight: 700" if value >= 0 else "color: #f85149; font-weight: 700",
            subset=["gap_pct", "day_return", "vs_spy_pct", "return_30m", "vwap_gap_pct"],
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

    render_pick_tracker()

    st.caption("Research tool only. Theme flow scores are proxies derived from price and volume, not official fund-flow data.")


if __name__ == "__main__":
    navigation = st.navigation(
        {
            "Dashboard": [
                st.Page(decision_dashboard_page, title="Decision Dashboard", url_path="overview", default=True),
                st.Page(recommendations_page, title="Recommendations", url_path="recommendations"),
                st.Page(signals_page, title="Signals", url_path="signals"),
                st.Page(dark_pool_page, title="Dark Pool Activity", url_path="dark-pool"),
            ],
            "Research": [
                st.Page(stock_analysis_page, title="Stock Analysis", url_path="stock-analysis"),
                st.Page(disclosures_page, title="Disclosures", url_path="disclosures"),
                st.Page(research_page, title="Market Research", url_path="research"),
                st.Page(main, title="Full Dashboard", url_path="full-dashboard"),
            ],
        },
        position="sidebar",
        expanded=True,
    )
    navigation.run()

"""Yahoo Finance data collection and money-flow feature generation."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .model_config import ALL_TICKERS, BENCHMARKS, INTRADAY_BENCHMARKS, SECTOR_ETFS, THEME_BASKETS
from .scoring import normalize, score_sector_flow


def download_prices(period: str = "6mo", interval: str = "1d", tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = sorted(set(tickers or ALL_TICKERS))
    data = _download_in_chunks(tickers, period=period, interval=interval)
    missing = sorted(set(tickers) - _available_tickers(data))
    if missing:
        repairs = [_download_yahoo([ticker], period=period, interval=interval) for ticker in missing]
        data = _merge_frames([data, *repairs])
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    return data


def download_intraday_prices(
    period: str = "5d",
    interval: str = "5m",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    return download_prices(period=period, interval=interval, tickers=tickers or list(INTRADAY_BENCHMARKS))


def download_intraday_component_prices(period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    # SPY is included so intraday component strength can be measured against the market.
    tickers = sorted({ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]} | {"SPY"})
    return download_prices(period=period, interval=interval, tickers=tickers)


def _download_in_chunks(tickers: list[str], period: str, interval: str, chunk_size: int = 35) -> pd.DataFrame:
    frames = []
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start + chunk_size]
        frames.append(_download_yahoo(chunk, period=period, interval=interval))
    return _merge_frames(frames)


def _download_yahoo(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=len(tickers) > 1,
    )
    return _ensure_multiindex(data, tickers[0] if len(tickers) == 1 else None)


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    data = pd.concat(usable, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data.dropna(how="all")


def _ensure_multiindex(data: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
    if data.empty or ticker is None or isinstance(data.columns, pd.MultiIndex):
        return data
    data = data.copy()
    data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
    return data


def _available_tickers(data: pd.DataFrame) -> set[str]:
    if data.empty:
        return set()
    try:
        close = _field(data, "Close")
        volume = _field(data, "Volume")
    except (KeyError, TypeError):
        return set()
    return {
        ticker for ticker in close.columns
        if ticker in volume.columns and len(close[ticker].dropna()) >= 25 and len(volume[ticker].dropna()) >= 25
    }


def _field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        return data[field].dropna(how="all")
    return data[[field]].dropna(how="all")


def build_theme_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    component_df = build_component_table(data)
    rows = []
    for theme, config in THEME_BASKETS.items():
        tickers = [ticker for ticker in config["tickers"] if ticker in set(component_df["ticker"])]
        if not tickers:
            continue
        members = component_df[component_df["ticker"].isin(tickers)].copy()
        proxy = _best_proxy(close, tickers)
        rows.append(
            {
                "theme": theme,
                "description": config["description"],
                "proxy": proxy,
                "components": ", ".join(tickers),
                "component_count": len(tickers),
                "flow_score": float(members["flow_score"].mean()),
                "return_1d": float(members["return_1d"].mean()),
                "return_5d": float(members["return_5d"].mean()),
                "return_20d": float(members["return_20d"].mean()),
                "relative_5d": float(members["relative_5d"].mean()),
                "dollar_volume_m": float(members["dollar_volume_m"].sum()),
                "dollar_volume_trend": float(members["dollar_volume_trend"].mean()),
                "volume_zscore": float(members["volume_zscore"].mean()),
                "top_component": str(members.sort_values("flow_score", ascending=False).iloc[0]["ticker"]),
                "weak_component": str(members.sort_values("flow_score", ascending=True).iloc[0]["ticker"]),
            }
        )
    if not rows:
        raise RuntimeError("No theme rows could be computed from market data")
    return pd.DataFrame(rows).sort_values("flow_score", ascending=False)


def build_weekly_theme_trends(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily theme prices and volume into weekly flow proxies."""
    close = _field(data, "Close")
    volume = _field(data, "Volume")
    weekly_close = close.resample("W-FRI").last()
    weekly_returns = weekly_close.pct_change(fill_method=None) * 100
    weekly_dollar_volume = (close * volume).resample("W-FRI").sum()
    trailing_volume = weekly_dollar_volume.rolling(8, min_periods=3).mean().shift(1)
    weekly_volume_trend = ((weekly_dollar_volume / trailing_volume) - 1) * 100
    spy_returns = weekly_returns["SPY"] if "SPY" in weekly_returns else pd.Series(0.0, index=weekly_returns.index)

    rows = []
    for week in weekly_returns.index:
        for theme, config in THEME_BASKETS.items():
            tickers = [ticker for ticker in config["tickers"] if ticker in weekly_returns.columns]
            if not tickers:
                continue
            member_returns = weekly_returns.loc[week, tickers].dropna()
            if member_returns.empty:
                continue
            member_volume_trend = weekly_volume_trend.loc[week, tickers].dropna()
            theme_return = float(member_returns.mean())
            relative_return = theme_return - float(spy_returns.get(week, 0.0) or 0.0)
            volume_trend = float(member_volume_trend.mean()) if not member_volume_trend.empty else 0.0
            flow_score = (
                normalize(theme_return, -8.0, 8.0) * 0.45
                + normalize(relative_return, -6.0, 6.0) * 0.30
                + normalize(volume_trend, -40.0, 80.0) * 0.25
            )
            dollar_volume = weekly_dollar_volume.loc[week, tickers].dropna()
            rows.append(
                {
                    "week": week,
                    "theme": theme,
                    "weekly_return": theme_return,
                    "relative_return": relative_return,
                    "volume_trend": volume_trend,
                    "flow_score": flow_score,
                    "net_flow": flow_score - 50.0,
                    "dollar_volume_m": float(dollar_volume.sum() / 1_000_000) if not dollar_volume.empty else 0.0,
                    "component_count": int(member_returns.count()),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "week",
                "theme",
                "weekly_return",
                "relative_return",
                "volume_trend",
                "flow_score",
                "net_flow",
                "dollar_volume_m",
                "component_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["week", "flow_score"], ascending=[True, False]).reset_index(drop=True)


def build_component_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    open_prices = _field(data, "Open")
    volume = _field(data, "Volume")
    dollar_volume = close * volume

    rows = []
    theme_tickers = sorted({ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]})
    for ticker in theme_tickers:
        if ticker not in close or ticker not in open_prices or ticker not in volume:
            continue
        prices = close[ticker].dropna()
        opens = open_prices[ticker].dropna()
        vols = volume[ticker].dropna()
        if len(prices) < 25 or len(opens) < 25 or len(vols) < 25:
            continue

        current_price = float(prices.iloc[-1])
        open_price = float(opens.iloc[-1])
        open_to_current_pct = ((current_price / open_price) - 1) * 100 if open_price else 0.0
        returns = _returns(prices)
        ret_1d = _pct_change(prices, 1)
        ret_5d = _pct_change(prices, 5)
        ret_20d = _pct_change(prices, 20)
        spy_5d = _pct_change(close["SPY"].dropna(), 5) if "SPY" in close else 0.0
        relative_5d = ret_5d - spy_5d

        dv = dollar_volume[ticker].dropna()
        recent_dv = float(dv.tail(5).mean())
        base_dv = float(dv.tail(60).mean()) if len(dv) >= 60 else float(dv.mean())
        dollar_volume_trend = ((recent_dv / base_dv) - 1) * 100 if base_dv else 0.0

        recent_vol = float(vols.iloc[-1])
        vol_mean = float(vols.tail(60).mean()) if len(vols) >= 60 else float(vols.mean())
        vol_std = float(vols.tail(60).std()) if len(vols) >= 60 else float(vols.std())
        volume_zscore = (recent_vol - vol_mean) / vol_std if vol_std else 0.0

        metrics = {
            "return_1d": ret_1d,
            "return_5d": ret_5d,
            "return_20d": ret_20d,
            "relative_5d": relative_5d,
            "dollar_volume_trend": dollar_volume_trend,
            "volume_zscore": volume_zscore,
        }
        rows.append(
            {
                "ticker": ticker,
                "themes": ", ".join(_themes_for_ticker(ticker)),
                "open_price": open_price,
                "last_price": current_price,
                "open_to_current_pct": open_to_current_pct,
                "return_1d": ret_1d,
                "return_5d": ret_5d,
                "return_20d": ret_20d,
                "relative_5d": relative_5d,
                "dollar_volume_m": recent_dv / 1_000_000,
                "dollar_volume_trend": dollar_volume_trend,
                "volume_zscore": volume_zscore,
                "flow_score": score_sector_flow(metrics),
                "daily_volatility": float(returns.tail(20).std() * 100),
            }
        )

    if not rows:
        raise RuntimeError("No component rows could be computed from market data")
    return pd.DataFrame(rows).sort_values("flow_score", ascending=False)


def build_sector_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    volume = _field(data, "Volume")
    dollar_volume = close * volume

    rows = []
    for ticker, sector in SECTOR_ETFS.items():
        if ticker not in close or ticker not in volume:
            continue
        prices = close[ticker].dropna()
        vols = volume[ticker].dropna()
        if len(prices) < 25 or len(vols) < 25:
            continue

        returns = _returns(prices)
        ret_1d = _pct_change(prices, 1)
        ret_5d = _pct_change(prices, 5)
        ret_20d = _pct_change(prices, 20)
        spy_5d = _pct_change(close["SPY"].dropna(), 5) if "SPY" in close else 0.0
        relative_5d = ret_5d - spy_5d

        dv = dollar_volume[ticker].dropna()
        recent_dv = float(dv.tail(5).mean())
        base_dv = float(dv.tail(60).mean()) if len(dv) >= 60 else float(dv.mean())
        dollar_volume_trend = ((recent_dv / base_dv) - 1) * 100 if base_dv else 0.0

        recent_vol = float(vols.iloc[-1])
        vol_mean = float(vols.tail(60).mean()) if len(vols) >= 60 else float(vols.mean())
        vol_std = float(vols.tail(60).std()) if len(vols) >= 60 else float(vols.std())
        volume_zscore = (recent_vol - vol_mean) / vol_std if vol_std else 0.0

        metrics = {
            "return_1d": ret_1d,
            "return_5d": ret_5d,
            "return_20d": ret_20d,
            "relative_5d": relative_5d,
            "dollar_volume_trend": dollar_volume_trend,
            "volume_zscore": volume_zscore,
        }
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "last_price": float(prices.iloc[-1]),
                "return_1d": ret_1d,
                "return_5d": ret_5d,
                "return_20d": ret_20d,
                "relative_5d": relative_5d,
                "dollar_volume_m": recent_dv / 1_000_000,
                "dollar_volume_trend": dollar_volume_trend,
                "volume_zscore": volume_zscore,
                "flow_score": score_sector_flow(metrics),
                "daily_volatility": float(returns.tail(20).std() * 100),
            }
        )

    if not rows:
        raise RuntimeError("No sector rows could be computed from market data")
    return pd.DataFrame(rows).sort_values("flow_score", ascending=False)


def benchmark_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    rows = []
    for ticker, label in BENCHMARKS.items():
        if ticker not in close:
            continue
        prices = close[ticker].dropna()
        if len(prices) < 25:
            continue
        rows.append(
            {
                "ticker": ticker,
                "name": label,
                "last_price": float(prices.iloc[-1]),
                "return_1d": _pct_change(prices, 1),
                "return_5d": _pct_change(prices, 5),
                "return_20d": _pct_change(prices, 20),
            }
        )
    return pd.DataFrame(rows)


def build_intraday_market_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    volume = _field(data, "Volume")
    highs, lows = _optional_fields(data)
    rows = []
    for ticker, label in INTRADAY_BENCHMARKS.items():
        if ticker not in close or ticker not in volume:
            continue
        metrics = _intraday_ticker_metrics(
            close[ticker].dropna(),
            volume[ticker],
            highs[ticker].dropna() if highs is not None and ticker in highs else None,
            lows[ticker].dropna() if lows is not None and ticker in lows else None,
        )
        if metrics is None:
            continue
        rows.append({"ticker": ticker, "name": label, **metrics})
    return pd.DataFrame(rows)


def build_intraday_component_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    volume = _field(data, "Volume")
    highs, lows = _optional_fields(data)

    spy_day_change = 0.0
    if "SPY" in close and "SPY" in volume:
        spy_metrics = _intraday_ticker_metrics(close["SPY"].dropna(), volume["SPY"])
        if spy_metrics is not None:
            spy_day_change = float(spy_metrics["day_change_pct"])

    rows = []
    theme_tickers = sorted({ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]})
    for ticker in theme_tickers:
        if ticker not in close or ticker not in volume:
            continue
        metrics = _intraday_ticker_metrics(
            close[ticker].dropna(),
            volume[ticker],
            highs[ticker].dropna() if highs is not None and ticker in highs else None,
            lows[ticker].dropna() if lows is not None and ticker in lows else None,
        )
        if metrics is None:
            continue
        rows.append(
            {
                "ticker": ticker,
                "themes": ", ".join(_themes_for_ticker(ticker)),
                **metrics,
                "vs_spy_pct": float(metrics["day_change_pct"]) - spy_day_change,
            }
        )
    return pd.DataFrame(rows)


def build_intraday_theme_table(component_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday component metrics into a theme-level money-flow board.

    Answers "where is money flowing right now" at the theme level: median
    session change including the overnight gap, strength versus SPY,
    time-of-day relative volume, and the share of components holding VWAP.
    """
    columns = [
        "theme",
        "component_count",
        "day_change_pct",
        "vs_spy_pct",
        "gap_pct",
        "rvol",
        "pct_above_vwap",
        "intraday_flow_score",
        "top_component",
        "weak_component",
    ]
    if component_df is None or component_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for theme, config in THEME_BASKETS.items():
        members = component_df[component_df["ticker"].isin(config["tickers"])]
        if members.empty:
            continue
        day_change = float(members["day_change_pct"].median())
        vs_spy = float(members["vs_spy_pct"].median())
        gap = float(members["gap_pct"].median())
        rvol = float(members["rvol"].median())
        pct_above_vwap = float((~members["below_vwap"].astype(bool)).mean() * 100)
        flow_score = (
            normalize(vs_spy, -2.0, 2.0) * 0.35
            + normalize(day_change, -2.5, 2.5) * 0.25
            + pct_above_vwap * 0.25
            + normalize(rvol, 0.4, 2.0) * 0.15
        )
        ranked = members.sort_values("day_change_pct", ascending=False)
        rows.append(
            {
                "theme": theme,
                "component_count": int(len(members)),
                "day_change_pct": day_change,
                "vs_spy_pct": vs_spy,
                "gap_pct": gap,
                "rvol": rvol,
                "pct_above_vwap": pct_above_vwap,
                "intraday_flow_score": flow_score,
                "top_component": str(ranked.iloc[0]["ticker"]),
                "weak_component": str(ranked.iloc[-1]["ticker"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("intraday_flow_score", ascending=False).reset_index(drop=True)


def _optional_fields(data: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        return _field(data, "High"), _field(data, "Low")
    except KeyError:
        return None, None


def _intraday_ticker_metrics(
    prices: pd.Series,
    volume: pd.Series,
    highs: pd.Series | None = None,
    lows: pd.Series | None = None,
) -> dict[str, object] | None:
    vols = volume.reindex(prices.index).fillna(0)
    if len(prices) < 12:
        return None

    latest_ts = prices.index[-1]
    latest_session = _session_slice(prices, latest_ts)
    latest_session_vol = vols.reindex(latest_session.index).fillna(0)
    if len(latest_session) < 3:
        return None

    last_price = float(latest_session.iloc[-1])
    session_open = float(latest_session.iloc[0])
    day_return = (last_price / session_open - 1) * 100 if session_open else 0.0

    prior_bars = prices[prices.index < latest_session.index[0]]
    prev_close = float(prior_bars.iloc[-1]) if not prior_bars.empty else 0.0
    gap_pct = (session_open / prev_close - 1) * 100 if prev_close else 0.0
    day_change_pct = (last_price / prev_close - 1) * 100 if prev_close else day_return

    return_30m = _pct_change(latest_session, 6)
    return_60m = _pct_change(latest_session, 12)
    vwap = _vwap(latest_session, latest_session_vol)
    vwap_gap_pct = (last_price / vwap - 1) * 100 if vwap else 0.0
    below_vwap = last_price < vwap if vwap else False

    recent_volume = float(latest_session_vol.tail(6).mean())
    base_volume = float(vols.tail(120).mean()) if len(vols) >= 120 else float(vols.mean())
    volume_trend = ((recent_volume / base_volume) - 1) * 100 if base_volume else 0.0
    rvol = _time_of_day_rvol(vols, latest_session_vol)
    recent_dollar_volume = float((latest_session.tail(6) * latest_session_vol.tail(6)).sum())

    # Opening range: first 30 minutes (6 five-minute bars) of the latest session.
    orb_bars = latest_session.head(6)
    orb_high = float(orb_bars.max())
    orb_low = float(orb_bars.min())
    if highs is not None and not highs.empty:
        session_highs = _session_slice(highs, latest_ts).head(6)
        if not session_highs.empty:
            orb_high = float(session_highs.max())
    if lows is not None and not lows.empty:
        session_lows = _session_slice(lows, latest_ts).head(6)
        if not session_lows.empty:
            orb_low = float(session_lows.min())

    return {
        "last_time": str(latest_ts),
        "last_price": last_price,
        "session_open": session_open,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "day_return": day_return,
        "day_change_pct": day_change_pct,
        "return_30m": return_30m,
        "return_60m": return_60m,
        "vwap": vwap,
        "vwap_gap_pct": vwap_gap_pct,
        "below_vwap": below_vwap,
        "volume_trend": volume_trend,
        "rvol": rvol,
        "recent_dollar_volume_m": recent_dollar_volume / 1_000_000,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "above_orb_high": last_price > orb_high,
    }


def _time_of_day_rvol(vols: pd.Series, latest_session_vol: pd.Series) -> float:
    """Cumulative session volume relative to the same elapsed time on prior sessions.

    Intraday volume follows a U-shape, so comparing the first hour against a
    flat all-day average always reads "elevated". Comparing against the same
    time-of-day on prior sessions keeps the measure honest near the open.
    """
    bars_elapsed = len(latest_session_vol)
    current_cum = float(latest_session_vol.sum())
    if not bars_elapsed:
        return 1.0
    try:
        latest_date = latest_session_vol.index[0].date()
        grouped = vols.groupby([idx.date() for idx in vols.index])
    except (AttributeError, TypeError):
        return 1.0
    prior_cums = [
        float(session.head(bars_elapsed).sum())
        for session_date, session in grouped
        if session_date < latest_date and len(session) >= min(bars_elapsed, 3)
    ]
    if not prior_cums:
        return 1.0
    baseline = sum(prior_cums) / len(prior_cums)
    if not baseline:
        return 1.0
    return current_cum / baseline


def build_intraday_price_table(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Build latest 5m session open/current price rows for selected tickers."""
    close = _field(data, "Close")
    try:
        open_prices = _field(data, "Open")
    except KeyError:
        open_prices = close

    rows = []
    for ticker in tickers:
        if ticker not in close:
            continue
        prices = close[ticker].dropna()
        if len(prices) < 2:
            continue

        latest_ts = prices.index[-1]
        latest_session = _session_slice(prices, latest_ts)
        open_series = open_prices[ticker].dropna() if ticker in open_prices else prices
        latest_open_session = _session_slice(open_series, latest_ts)
        if latest_session.empty or latest_open_session.empty:
            continue

        open_price = float(latest_open_session.iloc[0])
        last_price = float(latest_session.iloc[-1])
        rows.append(
            {
                "ticker": ticker,
                "last_time": str(latest_ts),
                "open_price": open_price,
                "last_price": last_price,
                "open_to_current_pct": ((last_price / open_price) - 1) * 100 if open_price else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    return (float(series.iloc[-1]) / float(series.iloc[-periods - 1]) - 1) * 100


def _returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def _session_slice(series: pd.Series, latest_ts) -> pd.Series:
    try:
        latest_date = latest_ts.date()
        return series[[idx.date() == latest_date for idx in series.index]]
    except AttributeError:
        return series.tail(78)


def _vwap(prices: pd.Series, volumes: pd.Series) -> float:
    volume_sum = float(volumes.sum())
    if not volume_sum:
        return float(prices.iloc[-1])
    return float((prices * volumes).sum() / volume_sum)


def _themes_for_ticker(ticker: str) -> list[str]:
    return [theme for theme, config in THEME_BASKETS.items() if ticker in config["tickers"]]


def _best_proxy(close: pd.DataFrame, tickers: list[str]) -> str:
    available = [ticker for ticker in tickers if ticker in close]
    if not available:
        return ""
    lengths = {ticker: len(close[ticker].dropna()) for ticker in available}
    return max(lengths, key=lengths.get)

"""Yahoo Finance data collection and money-flow feature generation."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .model_config import ALL_TICKERS, BENCHMARKS, INTRADAY_BENCHMARKS, SECTOR_ETFS, THEME_BASKETS
from .scoring import score_sector_flow


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
    tickers = sorted({ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]})
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
    rows = []
    for ticker, label in INTRADAY_BENCHMARKS.items():
        if ticker not in close or ticker not in volume:
            continue
        prices = close[ticker].dropna()
        vols = volume[ticker].reindex(prices.index).fillna(0)
        if len(prices) < 12:
            continue

        latest_ts = prices.index[-1]
        latest_session = _session_slice(prices, latest_ts)
        latest_session_vol = vols.reindex(latest_session.index).fillna(0)
        if len(latest_session) < 3:
            continue

        last_price = float(latest_session.iloc[-1])
        open_price = float(latest_session.iloc[0])
        day_return = (last_price / open_price - 1) * 100 if open_price else 0.0
        return_30m = _pct_change(latest_session, 6)
        return_60m = _pct_change(latest_session, 12)
        vwap = _vwap(latest_session, latest_session_vol)
        below_vwap = last_price < vwap if vwap else False
        recent_volume = float(latest_session_vol.tail(6).mean())
        base_volume = float(vols.tail(120).mean()) if len(vols) >= 120 else float(vols.mean())
        volume_trend = ((recent_volume / base_volume) - 1) * 100 if base_volume else 0.0

        rows.append(
            {
                "ticker": ticker,
                "name": label,
                "last_time": str(latest_ts),
                "last_price": last_price,
                "session_open": open_price,
                "day_return": day_return,
                "return_30m": return_30m,
                "return_60m": return_60m,
                "vwap": vwap,
                "below_vwap": below_vwap,
                "volume_trend": volume_trend,
            }
        )
    return pd.DataFrame(rows)


def build_intraday_component_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    volume = _field(data, "Volume")
    rows = []
    theme_tickers = sorted({ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]})
    for ticker in theme_tickers:
        if ticker not in close or ticker not in volume:
            continue
        prices = close[ticker].dropna()
        vols = volume[ticker].reindex(prices.index).fillna(0)
        if len(prices) < 12:
            continue

        latest_ts = prices.index[-1]
        latest_session = _session_slice(prices, latest_ts)
        latest_session_vol = vols.reindex(latest_session.index).fillna(0)
        if len(latest_session) < 3:
            continue

        last_price = float(latest_session.iloc[-1])
        session_open = float(latest_session.iloc[0])
        day_return = (last_price / session_open - 1) * 100 if session_open else 0.0
        return_30m = _pct_change(latest_session, 6)
        return_60m = _pct_change(latest_session, 12)
        vwap = _vwap(latest_session, latest_session_vol)
        vwap_gap_pct = (last_price / vwap - 1) * 100 if vwap else 0.0
        below_vwap = last_price < vwap if vwap else False
        recent_volume = float(latest_session_vol.tail(6).mean())
        base_volume = float(vols.tail(120).mean()) if len(vols) >= 120 else float(vols.mean())
        volume_trend = ((recent_volume / base_volume) - 1) * 100 if base_volume else 0.0
        recent_dollar_volume = float((latest_session.tail(6) * latest_session_vol.tail(6)).sum())

        rows.append(
            {
                "ticker": ticker,
                "themes": ", ".join(_themes_for_ticker(ticker)),
                "last_time": str(latest_ts),
                "last_price": last_price,
                "session_open": session_open,
                "day_return": day_return,
                "return_30m": return_30m,
                "return_60m": return_60m,
                "vwap": vwap,
                "vwap_gap_pct": vwap_gap_pct,
                "below_vwap": below_vwap,
                "volume_trend": volume_trend,
                "recent_dollar_volume_m": recent_dollar_volume / 1_000_000,
            }
        )
    return pd.DataFrame(rows)


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

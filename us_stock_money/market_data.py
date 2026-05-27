"""Yahoo Finance data collection and sector-flow feature generation."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .model_config import BENCHMARKS, SECTOR_ETFS
from .scoring import score_sector_flow


def download_prices(period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    tickers = list(SECTOR_ETFS) + list(BENCHMARKS)
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    return data


def _field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        return data[field].dropna(how="all")
    return data[[field]].dropna(how="all")


def build_sector_table(data: pd.DataFrame) -> pd.DataFrame:
    close = _field(data, "Close")
    volume = _field(data, "Volume")
    dollar_volume = close * volume
    spy_returns = _returns(close["SPY"]) if "SPY" in close else pd.Series(dtype=float)

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


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    return (float(series.iloc[-1]) / float(series.iloc[-periods - 1]) - 1) * 100


def _returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()

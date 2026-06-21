"""Technical indicators and moving-average risk alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ticker_field(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        if (field, ticker) not in data.columns:
            return pd.Series(dtype=float)
        series = data[(field, ticker)]
    elif field in data.columns:
        series = data[field]
    else:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").dropna()


def available_tickers(data: pd.DataFrame) -> list[str]:
    if data.empty:
        return []
    if isinstance(data.columns, pd.MultiIndex):
        return sorted(set(data.columns.get_level_values(1)))
    return []


def build_stock_detail(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    fields = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        series = _ticker_field(data, field, ticker)
        if not series.empty:
            fields[field.lower()] = series
    if "close" not in fields:
        return pd.DataFrame()

    detail = pd.concat(fields, axis=1).sort_index()
    detail["ma5"] = detail["close"].rolling(5).mean()
    detail["ma20"] = detail["close"].rolling(20).mean()
    detail["ma60"] = detail["close"].rolling(60).mean()

    delta = detail["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    detail["rsi14"] = 100 - (100 / (1 + relative_strength))
    detail.loc[(average_loss == 0) & (average_gain > 0), "rsi14"] = 100
    detail.loc[(average_loss == 0) & (average_gain == 0), "rsi14"] = 50

    ema12 = detail["close"].ewm(span=12, adjust=False).mean()
    ema26 = detail["close"].ewm(span=26, adjust=False).mean()
    detail["macd"] = ema12 - ema26
    detail["macd_signal"] = detail["macd"].ewm(span=9, adjust=False).mean()
    detail["macd_histogram"] = detail["macd"] - detail["macd_signal"]
    detail["drawdown"] = (detail["close"] / detail["close"].cummax() - 1) * 100
    return detail


def build_ma60_alerts(
    data: pd.DataFrame,
    tickers: list[str] | None = None,
    recent_sessions: int = 5,
) -> pd.DataFrame:
    rows = []
    candidates = tickers or available_tickers(data)
    for ticker in candidates:
        detail = build_stock_detail(data, ticker)
        valid = detail.dropna(subset=["close", "ma60"])
        if len(valid) < 2:
            continue

        below = valid["close"] < valid["ma60"]
        crossed_below = below & (valid["close"].shift(1) >= valid["ma60"].shift(1))
        cross_dates = valid.index[crossed_below.fillna(False)]
        latest_cross = cross_dates[-1] if len(cross_dates) else pd.NaT
        recent_index = valid.index[-max(1, recent_sessions):]
        recent_breakdown = bool(latest_cross in recent_index) if not pd.isna(latest_cross) else False
        latest = valid.iloc[-1]
        is_below = bool(below.iloc[-1])

        if recent_breakdown and is_below:
            status = "New Breakdown"
            priority = 0
        elif is_below:
            status = "Below MA60"
            priority = 1
        else:
            status = "Above MA60"
            priority = 2

        rows.append(
            {
                "ticker": ticker,
                "status": status,
                "last_price": float(latest["close"]),
                "ma20": float(latest["ma20"]) if pd.notna(latest["ma20"]) else np.nan,
                "ma60": float(latest["ma60"]),
                "distance_to_ma60_pct": float((latest["close"] / latest["ma60"] - 1) * 100),
                "return_5d": _period_return(valid["close"], 5),
                "return_20d": _period_return(valid["close"], 20),
                "cross_date": latest_cross,
                "recent_breakdown": recent_breakdown,
                "below_ma60": is_below,
                "_priority": priority,
            }
        )

    columns = [
        "ticker",
        "status",
        "last_price",
        "ma20",
        "ma60",
        "distance_to_ma60_pct",
        "return_5d",
        "return_20d",
        "cross_date",
        "recent_breakdown",
        "below_ma60",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows).sort_values(
        ["_priority", "distance_to_ma60_pct"],
        ascending=[True, True],
    )
    return result.drop(columns="_priority").reset_index(drop=True)


def stock_snapshot(detail: pd.DataFrame) -> dict[str, float]:
    valid = detail.dropna(subset=["close"])
    if valid.empty:
        return {}
    latest = valid.iloc[-1]
    returns = valid["close"].pct_change(fill_method=None)
    trailing_year = valid.tail(252)
    return {
        "last_price": float(latest["close"]),
        "daily_return": _period_return(valid["close"], 1),
        "ma20_gap": _gap(latest["close"], latest.get("ma20")),
        "ma60_gap": _gap(latest["close"], latest.get("ma60")),
        "rsi14": float(latest.get("rsi14", np.nan)),
        "macd": float(latest.get("macd", np.nan)),
        "high_52w": float(trailing_year["close"].max()),
        "low_52w": float(trailing_year["close"].min()),
        "volatility_20d": float(returns.tail(20).std() * np.sqrt(252) * 100),
        "max_drawdown_52w": float(trailing_year["drawdown"].min()),
    }


def _period_return(prices: pd.Series, periods: int) -> float:
    if len(prices) <= periods:
        return np.nan
    return float((prices.iloc[-1] / prices.iloc[-periods - 1] - 1) * 100)


def _gap(price: float, moving_average: object) -> float:
    if moving_average is None or pd.isna(moving_average) or float(moving_average) == 0:
        return np.nan
    return float((price / float(moving_average) - 1) * 100)

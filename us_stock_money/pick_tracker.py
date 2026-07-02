"""Evaluate archived intraday picks against later daily closes.

The dashboard logs the top intraday breakout candidates once per session.
This module fills in what actually happened afterwards (same-day close and
next-day close), so the hit rate of the 5m recommendation engine can be
measured instead of assumed.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .market_data import _field, download_prices
from .storage import HistoryStore


def evaluate_intraday_picks(store: HistoryStore, today: dt.date | None = None) -> pd.DataFrame:
    """Fill missing outcomes for archived picks and return the full pick table.

    Picks made today stay pending until their session has closed; only prior
    sessions are evaluated.
    """
    picks = store.load_intraday_picks()
    if not picks:
        return pd.DataFrame()
    today = today or dt.date.today()

    pending = [
        pick
        for pick in picks
        if (pick.get("close_return_pct") is None or pick.get("next_close_return_pct") is None)
        and (parsed := _parse_date(str(pick.get("pick_date", "")))) is not None
        and parsed < today
    ]
    if pending:
        outcomes = _compute_outcomes(pending)
        if outcomes:
            store.update_pick_outcomes(outcomes)
        picks = store.load_intraday_picks()
    return pd.DataFrame(picks)


def _compute_outcomes(pending: list[dict[str, object]]) -> list[dict[str, object]]:
    tickers = sorted({str(pick["ticker"]) for pick in pending})
    try:
        data = download_prices(period="2mo", interval="1d", tickers=tickers)
        close = _field(data, "Close")
    except Exception:
        return []

    outcomes: list[dict[str, object]] = []
    for pick in pending:
        ticker = str(pick["ticker"])
        pick_date = _parse_date(str(pick["pick_date"]))
        pick_price = float(pick.get("pick_price") or 0.0)
        if ticker not in close or pick_date is None or not pick_price:
            continue
        prices = close[ticker].dropna()
        session_dates = [timestamp.date() for timestamp in prices.index]
        matches = [index for index, session in enumerate(session_dates) if session == pick_date]
        if not matches:
            continue
        position = matches[0]
        close_price = float(prices.iloc[position])
        outcome: dict[str, object] = {
            "pick_date": pick["pick_date"],
            "ticker": ticker,
            "close_price": close_price,
            "close_return_pct": (close_price / pick_price - 1) * 100,
            "next_close_price": None,
            "next_close_return_pct": None,
        }
        if position + 1 < len(prices):
            next_close = float(prices.iloc[position + 1])
            outcome["next_close_price"] = next_close
            outcome["next_close_return_pct"] = (next_close / pick_price - 1) * 100
        outcomes.append(outcome)
    return outcomes


def pick_hit_rate_summary(picks_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate win rate and average return over evaluated picks."""
    empty = {
        "evaluated": 0.0,
        "win_rate": 0.0,
        "avg_return": 0.0,
        "next_day_evaluated": 0.0,
        "next_day_win_rate": 0.0,
        "next_day_avg_return": 0.0,
    }
    if picks_df is None or picks_df.empty or "close_return_pct" not in picks_df.columns:
        return empty

    same_day = picks_df["close_return_pct"].dropna()
    summary = dict(empty)
    if not same_day.empty:
        summary["evaluated"] = float(len(same_day))
        summary["win_rate"] = float((same_day > 0).mean() * 100)
        summary["avg_return"] = float(same_day.mean())
    if "next_close_return_pct" in picks_df.columns:
        next_day = picks_df["next_close_return_pct"].dropna()
        if not next_day.empty:
            summary["next_day_evaluated"] = float(len(next_day))
            summary["next_day_win_rate"] = float((next_day > 0).mean() * 100)
            summary["next_day_avg_return"] = float(next_day.mean())
    return summary


def _parse_date(value: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

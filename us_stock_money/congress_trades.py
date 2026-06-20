"""Congressional STOCK Act transaction data loading and normalization."""

from __future__ import annotations

import json

import pandas as pd
import requests


CONGRESS_TRADES_URL = (
    "https://raw.githubusercontent.com/kadoa-org/"
    "congress-trading-monitor/main/public/data/trades.json"
)

DISPLAY_COLUMNS = [
    "transaction_date",
    "filing_date",
    "filer_name",
    "chamber",
    "party",
    "state",
    "ticker",
    "asset_name",
    "transaction_type",
    "amount_range_label",
    "days_to_file",
    "doc_url",
]


def download_congress_trades(url: str = CONGRESS_TRADES_URL) -> pd.DataFrame:
    response = requests.get(url, headers={"User-Agent": "us-stock-money/1.0"}, timeout=30)
    response.raise_for_status()
    payload = json.loads(response.content)
    return normalize_congress_trades(payload)


def normalize_congress_trades(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    for column in DISPLAY_COLUMNS:
        if column not in frame:
            frame[column] = None

    if "branch" in frame:
        frame = frame[frame["branch"].fillna("").str.lower() == "congress"]
    if "filing_type" in frame:
        frame = frame[frame["filing_type"].fillna("").str.upper() == "PTR"]

    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.upper()
    frame["chamber"] = frame["chamber"].fillna("unknown").astype(str).str.title()
    frame["transaction_type"] = frame["transaction_type"].fillna("Unknown").astype(str)
    frame["trade_side"] = frame["transaction_type"].map(classify_trade_side)
    frame["days_to_file"] = pd.to_numeric(frame["days_to_file"], errors="coerce")

    return (
        frame.dropna(subset=["transaction_date"])
        .sort_values(["filing_date", "transaction_date"], ascending=False)
        .reset_index(drop=True)
    )


def filter_congress_trades(
    frame: pd.DataFrame,
    *,
    days: int = 90,
    chambers: list[str] | None = None,
    sides: list[str] | None = None,
    ticker: str = "",
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    current_day = (today or pd.Timestamp.now()).normalize()
    cutoff = current_day - pd.Timedelta(days=days)
    filtered = frame[frame["transaction_date"] >= cutoff].copy()

    if chambers:
        filtered = filtered[filtered["chamber"].isin(chambers)]
    if sides:
        filtered = filtered[filtered["trade_side"].isin(sides)]
    if ticker.strip():
        filtered = filtered[filtered["ticker"].str.contains(ticker.strip().upper(), regex=False)]

    return filtered.reset_index(drop=True)


def classify_trade_side(transaction_type: object) -> str:
    value = str(transaction_type).lower()
    if "purchase" in value or "buy" in value:
        return "Purchase"
    if "sale" in value or "sell" in value:
        return "Sale"
    if "exchange" in value:
        return "Exchange"
    return "Other"

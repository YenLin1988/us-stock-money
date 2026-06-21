"""Congressional STOCK Act transaction data loading and normalization."""

from __future__ import annotations

import json
import re

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
    parsed_ranges = frame["amount_range_label"].map(_parse_amount_range)
    for column, position in [("amount_range_low", 0), ("amount_range_high", 1)]:
        existing = (
            pd.to_numeric(frame[column], errors="coerce")
            if column in frame
            else pd.Series(float("nan"), index=frame.index)
        )
        frame[column] = existing.fillna(parsed_ranges.map(lambda value: value[position]))

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


def summarize_congress_trades(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0.0,
            "tickers": 0.0,
            "purchase_value": 0.0,
            "sale_value": 0.0,
            "net_value": 0.0,
        }
    values = _estimated_values(frame)
    purchases = frame["trade_side"] == "Purchase"
    sales = frame["trade_side"] == "Sale"
    purchase_value = float(values[purchases].sum())
    sale_value = float(values[sales].sum())
    return {
        "trades": float(len(frame)),
        "tickers": float(frame.loc[frame["ticker"] != "", "ticker"].nunique()),
        "purchase_value": purchase_value,
        "sale_value": sale_value,
        "net_value": purchase_value - sale_value,
    }


def aggregate_congress_by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "asset_name",
        "signal",
        "trade_count",
        "purchases",
        "sales",
        "estimated_buy_value",
        "estimated_sale_value",
        "estimated_net_value",
        "filer_count",
        "latest_trade",
    ]
    usable = frame[frame["ticker"].fillna("") != ""].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    usable["estimated_value"] = _estimated_values(usable)
    rows = []
    for ticker, group in usable.groupby("ticker", sort=False):
        purchases = group[group["trade_side"] == "Purchase"]
        sales = group[group["trade_side"] == "Sale"]
        buy_value = float(purchases["estimated_value"].sum())
        sale_value = float(sales["estimated_value"].sum())
        rows.append(
            {
                "ticker": ticker,
                "asset_name": _latest_text(group, "asset_name"),
                "signal": _activity_signal(buy_value, sale_value),
                "trade_count": len(group),
                "purchases": len(purchases),
                "sales": len(sales),
                "estimated_buy_value": buy_value,
                "estimated_sale_value": sale_value,
                "estimated_net_value": buy_value - sale_value,
                "filer_count": group["filer_name"].nunique(),
                "latest_trade": group["transaction_date"].max(),
            }
        )
    result = pd.DataFrame(rows)
    result["_activity"] = result["estimated_buy_value"] + result["estimated_sale_value"]
    return result.sort_values("_activity", ascending=False).drop(columns="_activity").reset_index(drop=True)


def classify_trade_side(transaction_type: object) -> str:
    value = str(transaction_type).lower()
    if "purchase" in value or "buy" in value:
        return "Purchase"
    if "sale" in value or "sell" in value:
        return "Sale"
    if "exchange" in value:
        return "Exchange"
    return "Other"


def _estimated_values(frame: pd.DataFrame) -> pd.Series:
    low_source = frame["amount_range_low"] if "amount_range_low" in frame else pd.Series(0.0, index=frame.index)
    high_source = frame["amount_range_high"] if "amount_range_high" in frame else low_source
    low = pd.to_numeric(low_source, errors="coerce").fillna(0.0)
    high = pd.to_numeric(high_source, errors="coerce").fillna(low)
    return (low + high) / 2


def _latest_text(frame: pd.DataFrame, column: str) -> str:
    values = frame.sort_values("transaction_date", ascending=False)[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""


def _activity_signal(buy_value: float, sale_value: float) -> str:
    total = buy_value + sale_value
    if total == 0:
        return "No Value"
    buy_share = buy_value / total
    if buy_share >= 0.65:
        return "Net Buying"
    if buy_share <= 0.35:
        return "Net Selling"
    return "Mixed"


def _parse_amount_range(value: object) -> tuple[float, float]:
    numbers = [float(item.replace(",", "")) for item in re.findall(r"\d[\d,]*", str(value))]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return 0.0, 0.0

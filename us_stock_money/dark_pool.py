"""FINRA ATS weekly activity and anomaly detection."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import requests


FINRA_WEEKLY_SUMMARY_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"


def download_ats_weekly_activity(
    tickers: list[str],
    *,
    weeks: int = 14,
    today: dt.date | None = None,
) -> pd.DataFrame:
    end_date = today or dt.date.today()
    start_date = end_date - dt.timedelta(weeks=weeks + 6)
    payload = {
        "fields": [
            "issueSymbolIdentifier",
            "issueName",
            "tierIdentifier",
            "summaryStartDate",
            "totalWeeklyTradeCount",
            "totalWeeklyShareQuantity",
            "totalNotionalSum",
            "initialPublishedDate",
        ],
        "domainFilters": [
            {
                "fieldName": "issueSymbolIdentifier",
                "values": sorted(set(tickers)),
            }
        ],
        "compareFilters": [
            {
                "compareType": "EQUAL",
                "fieldName": "summaryTypeCode",
                "fieldValue": "ATS_W_SMBL",
            }
        ],
        "dateRangeFilters": [
            {
                "fieldName": "weekStartDate",
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            }
        ],
        "limit": 5000,
    }
    response = requests.post(
        FINRA_WEEKLY_SUMMARY_URL,
        json=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "us-stock-money/1.0",
        },
        timeout=45,
    )
    response.raise_for_status()
    return normalize_ats_activity(response.json()).groupby("ticker", group_keys=False).tail(weeks)


def normalize_ats_activity(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "ticker",
        "issue_name",
        "tier",
        "week",
        "ats_trades",
        "ats_shares",
        "ats_notional",
        "published_date",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records).rename(
        columns={
            "issueSymbolIdentifier": "ticker",
            "issueName": "issue_name",
            "tierIdentifier": "tier",
            "summaryStartDate": "week",
            "totalWeeklyTradeCount": "ats_trades",
            "totalWeeklyShareQuantity": "ats_shares",
            "totalNotionalSum": "ats_notional",
            "initialPublishedDate": "published_date",
        }
    )
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.upper()
    frame["week"] = pd.to_datetime(frame["week"], errors="coerce")
    frame["published_date"] = pd.to_datetime(frame["published_date"], errors="coerce")
    for column in ["ats_trades", "ats_shares", "ats_notional"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return (
        frame[columns]
        .dropna(subset=["week"])
        .drop_duplicates(["ticker", "week"])
        .sort_values(["ticker", "week"])
        .reset_index(drop=True)
    )


def add_market_context(ats_frame: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    if ats_frame.empty:
        return ats_frame.assign(market_volume=pd.Series(dtype=float), ats_volume_pct=pd.Series(dtype=float))
    volume = _field(market_data, "Volume")
    close = _field(market_data, "Close")
    rows = []
    for row in ats_frame.itertuples(index=False):
        market_volume = np.nan
        weekly_return = np.nan
        if row.ticker in volume.columns:
            week_end = row.week + pd.Timedelta(days=4)
            weekly_volume = volume.loc[(volume.index >= row.week) & (volume.index <= week_end), row.ticker].dropna()
            market_volume = float(weekly_volume.sum()) if not weekly_volume.empty else np.nan
        if row.ticker in close.columns:
            prices = close.loc[close.index <= row.week + pd.Timedelta(days=4), row.ticker].dropna()
            if len(prices) >= 6:
                weekly_return = float((prices.iloc[-1] / prices.iloc[-6] - 1) * 100)
        item = row._asdict()
        item["market_volume"] = market_volume
        item["ats_volume_pct"] = (
            float(row.ats_shares / market_volume * 100)
            if pd.notna(market_volume) and market_volume > 0
            else np.nan
        )
        item["weekly_return"] = weekly_return
        rows.append(item)
    return pd.DataFrame(rows)


def build_ats_anomalies(
    frame: pd.DataFrame,
    *,
    baseline_weeks: int = 8,
    volume_multiple: float = 2.0,
    z_threshold: float = 2.5,
    minimum_shares: float = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    enriched_groups = []
    for _, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("week").copy()
        prior = group["ats_shares"].shift(1)
        group["baseline_median"] = prior.rolling(baseline_weeks, min_periods=4).median()
        group["baseline_mean"] = prior.rolling(baseline_weeks, min_periods=4).mean()
        group["baseline_std"] = prior.rolling(baseline_weeks, min_periods=4).std()
        group["volume_multiple"] = group["ats_shares"] / group["baseline_median"].replace(0, np.nan)
        group["z_score"] = (
            (group["ats_shares"] - group["baseline_mean"])
            / group["baseline_std"].replace(0, np.nan)
        )
        group["week_change_pct"] = group["ats_shares"].pct_change(fill_method=None) * 100
        group["is_anomaly"] = (
            (group["ats_shares"] >= minimum_shares)
            & (
                (group["volume_multiple"] >= volume_multiple)
                | (group["z_score"] >= z_threshold)
            )
        )
        group["anomaly_score"] = (
            group["volume_multiple"].fillna(0).clip(upper=5) * 35
            + group["z_score"].fillna(0).clip(lower=0, upper=5) * 15
            + group.get("ats_volume_pct", pd.Series(0, index=group.index)).fillna(0).clip(upper=50)
        ).clip(upper=100)
        enriched_groups.append(group)
    history = pd.concat(enriched_groups, ignore_index=True)
    latest = history.sort_values("week").groupby("ticker", as_index=False).tail(1)
    anomalies = latest[latest["is_anomaly"]].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    return history, anomalies


def _field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data[field].dropna(how="all")
    return data[[field]].dropna(how="all")

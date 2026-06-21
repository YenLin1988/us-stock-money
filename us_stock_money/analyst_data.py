"""Valuation and analyst price-target data from Yahoo Finance."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from .model_config import THEME_BASKETS


VALUATION_FIELDS = [
    "trailing_pe",
    "forward_pe",
    "price_to_sales",
    "price_to_book",
    "enterprise_to_ebitda",
    "peg_ratio",
    "target_high",
    "target_low",
    "target_mean",
    "target_median",
    "analyst_count",
    "recommendation",
    "recommendation_mean",
    "currency",
]


def download_analyst_data(ticker: str) -> tuple[dict[str, object], pd.DataFrame]:
    stock = yf.Ticker(ticker)
    info = stock.get_info()
    price_targets = stock.get_analyst_price_targets() or {}
    upgrades = stock.get_upgrades_downgrades()
    valuation = normalize_valuation(info, price_targets)
    target_history = normalize_target_history(
        upgrades,
        current_price=price_targets.get("current", info.get("currentPrice")),
    )
    return valuation, target_history


def themes_for_ticker(ticker: str) -> list[str]:
    symbol = ticker.upper()
    return [
        theme
        for theme, config in THEME_BASKETS.items()
        if symbol in config["tickers"]
    ]


def peer_tickers_for_theme(ticker: str, theme: str, limit: int = 15) -> list[str]:
    symbol = ticker.upper()
    configured = list(THEME_BASKETS.get(theme, {}).get("tickers", []))
    peers = [item for item in configured if item != symbol]
    return [symbol, *peers[: max(0, limit - 1)]]


def download_peer_valuations(
    ticker: str,
    theme: str,
    *,
    limit: int = 15,
) -> pd.DataFrame:
    tickers = peer_tickers_for_theme(ticker, theme, limit=limit)
    infos: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(tickers))) as executor:
        futures = {
            executor.submit(yf.Ticker(symbol).get_info): symbol
            for symbol in tickers
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                infos[symbol] = future.result() or {}
            except Exception:
                infos[symbol] = {}
    return normalize_peer_valuations(ticker, theme, infos, ticker_order=tickers)


def normalize_peer_valuations(
    ticker: str,
    theme: str,
    infos: dict[str, dict[str, object]],
    *,
    ticker_order: list[str] | None = None,
) -> pd.DataFrame:
    selected = ticker.upper()
    order = ticker_order or list(infos)
    rows = []
    for symbol in order:
        info = infos.get(symbol, {})
        rows.append(
            {
                "ticker": symbol,
                "company": str(info.get("shortName") or info.get("longName") or symbol),
                "theme": theme,
                "selected": symbol == selected,
                "trailing_pe": _number(info.get("trailingPE")),
                "forward_pe": _number(info.get("forwardPE")),
                "market_cap": _number(info.get("marketCap")),
                "earnings_growth_pct": _percentage(info.get("earningsGrowth")),
                "revenue_growth_pct": _percentage(info.get("revenueGrowth")),
            }
        )
    return pd.DataFrame(rows)


def pe_comparison_summary(frame: pd.DataFrame, ticker: str) -> dict[str, float | None]:
    if frame.empty:
        return {
            "trailing_pe": None,
            "trailing_median": None,
            "trailing_premium_pct": None,
            "forward_pe": None,
            "forward_median": None,
            "forward_premium_pct": None,
            "peer_count": 0.0,
        }
    selected_rows = frame[frame["ticker"] == ticker.upper()]
    selected_row = selected_rows.iloc[0] if not selected_rows.empty else pd.Series(dtype=object)
    trailing_median = _positive_median(frame["trailing_pe"])
    forward_median = _positive_median(frame["forward_pe"])
    trailing_pe = _number(selected_row.get("trailing_pe"))
    forward_pe = _number(selected_row.get("forward_pe"))
    return {
        "trailing_pe": trailing_pe,
        "trailing_median": trailing_median,
        "trailing_premium_pct": _premium(trailing_pe, trailing_median),
        "forward_pe": forward_pe,
        "forward_median": forward_median,
        "forward_premium_pct": _premium(forward_pe, forward_median),
        "peer_count": float(len(frame)),
    }


def normalize_valuation(
    info: dict[str, object] | None,
    price_targets: dict[str, object] | None = None,
) -> dict[str, object]:
    source = info or {}
    targets = price_targets or {}
    return {
        "trailing_pe": _number(source.get("trailingPE")),
        "forward_pe": _number(source.get("forwardPE")),
        "price_to_sales": _number(source.get("priceToSalesTrailing12Months")),
        "price_to_book": _number(source.get("priceToBook")),
        "enterprise_to_ebitda": _number(source.get("enterpriseToEbitda")),
        "peg_ratio": _number(source.get("trailingPegRatio")),
        "target_high": _number(targets.get("high", source.get("targetHighPrice"))),
        "target_low": _number(targets.get("low", source.get("targetLowPrice"))),
        "target_mean": _number(targets.get("mean", source.get("targetMeanPrice"))),
        "target_median": _number(targets.get("median", source.get("targetMedianPrice"))),
        "analyst_count": int(_number(source.get("numberOfAnalystOpinions")) or 0),
        "recommendation": str(source.get("recommendationKey") or "").replace("_", " ").title(),
        "recommendation_mean": _number(source.get("recommendationMean")),
        "currency": str(source.get("currency") or "USD"),
    }


def normalize_target_history(
    frame: pd.DataFrame | None,
    *,
    current_price: object = None,
) -> pd.DataFrame:
    columns = [
        "estimate_date",
        "firm",
        "rating",
        "action",
        "target_action",
        "price_target",
        "prior_price_target",
        "target_change_pct",
        "upside_pct",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    normalized = frame.reset_index().copy()
    index_column = normalized.columns[0]
    if "GradeDate" not in normalized and index_column not in frame.columns:
        normalized = normalized.rename(columns={index_column: "GradeDate"})
    rename_map = {
        "GradeDate": "estimate_date",
        "Firm": "firm",
        "ToGrade": "rating",
        "Action": "action",
        "priceTargetAction": "target_action",
        "currentPriceTarget": "price_target",
        "priorPriceTarget": "prior_price_target",
    }
    normalized = normalized.rename(columns=rename_map)
    for column in rename_map.values():
        if column not in normalized:
            normalized[column] = None

    normalized["estimate_date"] = pd.to_datetime(normalized["estimate_date"], errors="coerce")
    normalized["price_target"] = pd.to_numeric(normalized["price_target"], errors="coerce")
    normalized["prior_price_target"] = pd.to_numeric(normalized["prior_price_target"], errors="coerce")
    normalized = normalized[normalized["price_target"].fillna(0) > 0].copy()
    normalized["target_change_pct"] = (
        (normalized["price_target"] / normalized["prior_price_target"] - 1) * 100
    ).where(normalized["prior_price_target"] > 0)

    price = _number(current_price)
    normalized["upside_pct"] = (
        (normalized["price_target"] / price - 1) * 100
        if price is not None and price > 0
        else float("nan")
    )
    normalized["firm"] = normalized["firm"].fillna("").astype(str)
    normalized["rating"] = normalized["rating"].fillna("").astype(str)
    normalized["action"] = normalized["action"].fillna("").astype(str).map(_action_label)
    normalized["target_action"] = normalized["target_action"].fillna("").astype(str).map(_target_action_label)
    return (
        normalized[columns]
        .sort_values("estimate_date", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _percentage(value: object) -> float | None:
    number = _number(value)
    return None if number is None else number * 100


def _positive_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[values > 0]
    return None if values.empty else float(values.median())


def _premium(value: float | None, median: float | None) -> float | None:
    if value is None or median is None or median <= 0 or value <= 0:
        return None
    return (value / median - 1) * 100


def _action_label(value: str) -> str:
    return {
        "init": "Initiated",
        "main": "Maintained",
        "reit": "Reiterated",
        "up": "Upgraded",
        "down": "Downgraded",
    }.get(value.lower(), value.title())


def _target_action_label(value: str) -> str:
    return {
        "announces": "Announced",
        "maintains": "Maintained",
        "raises": "Raised",
        "lowers": "Lowered",
        "sets": "Set",
    }.get(value.lower(), value.title())

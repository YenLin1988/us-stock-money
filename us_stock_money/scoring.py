"""Pure scoring helpers for sector money-flow classification."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Mapping

from .model_config import FLOW_WEIGHTS, GROUPS, THEME_GROUPS


@dataclass(frozen=True)
class FlowRegime:
    name: str
    color: str
    css: str
    icon: str


@dataclass(frozen=True)
class MarketTimingSignal:
    status: str
    severity: str
    title: str
    message: str
    score: float
    evidence: list[str]


@dataclass(frozen=True)
class IntradayMarketSignal:
    status: str
    severity: str
    title: str
    message: str
    score: float
    evidence: list[str]


@dataclass(frozen=True)
class IntradayEntryGate:
    """Session-level go / no-go verdict for opening new positions today."""

    status: str  # "no_entry" | "caution" | "ok" | "unavailable"
    severity: str
    title: str
    message: str
    evidence: list[str]
    negative_theme_share: float
    semis_day_change: float
    semis_weak: bool


def normalize(value: float, low: float, high: float) -> float:
    """Linearly map a value onto a clamped 0-100 score."""
    if high == low:
        raise ValueError("low and high must differ")
    score = ((value - low) / (high - low)) * 100
    return min(100, max(0, score))


def score_sector_flow(metrics: Mapping[str, float], weights: Mapping[str, float] = FLOW_WEIGHTS) -> float:
    """Convert sector metrics into one 0-100 money-flow score."""
    factor_scores = {
        "return_1d": normalize(metrics.get("return_1d", 0), -2.5, 2.5),
        "return_5d": normalize(metrics.get("return_5d", 0), -5.0, 5.0),
        "return_20d": normalize(metrics.get("return_20d", 0), -10.0, 10.0),
        "relative_5d": normalize(metrics.get("relative_5d", 0), -4.0, 4.0),
        "dollar_volume_trend": normalize(metrics.get("dollar_volume_trend", 0), -30.0, 30.0),
        "volume_zscore": normalize(metrics.get("volume_zscore", 0), -2.0, 2.0),
    }
    missing = set(factor_scores) - set(weights)
    if missing:
        raise KeyError(f"Missing flow weights for factors: {sorted(missing)}")
    return sum(factor_scores[name] * weights[name] for name in factor_scores)


def group_scores(sector_scores: Mapping[str, float], groups: Mapping[str, set[str]] = GROUPS) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, tickers in groups.items():
        available = [sector_scores[ticker] for ticker in tickers if ticker in sector_scores]
        result[group] = sum(available) / len(available) if available else 0.0
    return result


def theme_group_scores(theme_scores: Mapping[str, float], groups: Mapping[str, set[str]] = THEME_GROUPS) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, themes in groups.items():
        available = [theme_scores[theme] for theme in themes if theme in theme_scores]
        result[group] = sum(available) / len(available) if available else 0.0
    return result


def build_top_recommendations(component_rows, theme_scores: Mapping[str, float], limit: int = 5) -> list[dict[str, object]]:
    """Rank component tickers by their own flow plus the strength of related themes."""
    recommendations = []
    for row in _iter_records(component_rows):
        themes = [theme.strip() for theme in str(row.get("themes", "")).split(",") if theme.strip()]
        related_theme_scores = [float(theme_scores[theme]) for theme in themes if theme in theme_scores]
        theme_boost = sum(related_theme_scores) / len(related_theme_scores) if related_theme_scores else 0.0
        flow_score = float(row.get("flow_score", 0.0))
        composite_score = (flow_score * 0.7) + (theme_boost * 0.3)
        recommendations.append(
            {
                "ticker": row.get("ticker", ""),
                "themes": ", ".join(themes),
                "open_price": float(row.get("open_price", 0.0)),
                "last_price": float(row.get("last_price", 0.0)),
                "open_to_current_pct": float(row.get("open_to_current_pct", 0.0)),
                "flow_score": flow_score,
                "theme_score": theme_boost,
                "composite_score": composite_score,
                "return_5d": float(row.get("return_5d", 0.0)),
                "return_20d": float(row.get("return_20d", 0.0)),
                "relative_5d": float(row.get("relative_5d", 0.0)),
                "dollar_volume_trend": float(row.get("dollar_volume_trend", 0.0)),
                "volume_zscore": float(row.get("volume_zscore", 0.0)),
                "reason": recommendation_reason(row, theme_boost),
            }
        )
    return sorted(recommendations, key=lambda item: float(item["composite_score"]), reverse=True)[:limit]


def build_integrated_recommendations(
    component_rows,
    theme_scores: Mapping[str, float],
    intraday_rows,
    congress_rows,
    insider_rows,
    market_score: float,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Combine market, flow, disclosure, and risk signals into one transparent score."""
    intraday_by_ticker = {
        str(row.get("ticker", "")): row
        for row in build_intraday_breakout_candidates(intraday_rows, limit=10_000)
    }
    congress_by_ticker = _disclosure_scores(congress_rows, value_fields=("amount_range_low", "amount_range_high"))
    insider_by_ticker = _disclosure_scores(insider_rows, value_fields=("estimated_value",))

    recommendations = []
    for row in _iter_records(component_rows):
        ticker = str(row.get("ticker", ""))
        themes = [theme.strip() for theme in str(row.get("themes", "")).split(",") if theme.strip()]
        related_theme_scores = [float(theme_scores[theme]) for theme in themes if theme in theme_scores]
        theme_score = sum(related_theme_scores) / len(related_theme_scores) if related_theme_scores else 50.0
        flow_score = float(row.get("flow_score", 0.0))
        momentum_score = (
            normalize(float(row.get("return_5d", 0.0)), -5.0, 10.0)
            + normalize(float(row.get("return_20d", 0.0)), -10.0, 20.0)
            + normalize(float(row.get("relative_5d", 0.0)), -4.0, 8.0)
        ) / 3

        intraday = intraday_by_ticker.get(ticker, {})
        intraday_score = float(intraday.get("breakout_score", 50.0))
        exit_signal = str(intraday.get("exit_signal", "Watch"))
        risk_penalty = {"Exit": 20.0, "Trim": 10.0, "Watch": 3.0, "Hold": 0.0}.get(exit_signal, 3.0)

        congress = congress_by_ticker.get(ticker, _neutral_disclosure_score())
        insider = insider_by_ticker.get(ticker, _neutral_disclosure_score())
        integrated_score = (
            flow_score * 0.25
            + theme_score * 0.15
            + momentum_score * 0.15
            + intraday_score * 0.20
            + float(congress["score"]) * 0.10
            + float(insider["score"]) * 0.10
            + min(100.0, max(0.0, float(market_score))) * 0.05
            - risk_penalty
        )
        integrated_score = min(100.0, max(0.0, integrated_score))

        recommendations.append(
            {
                "ticker": ticker,
                "themes": ", ".join(themes),
                "integrated_score": integrated_score,
                "rating": _integrated_rating(integrated_score, exit_signal),
                "flow_score": flow_score,
                "theme_score": theme_score,
                "momentum_score": momentum_score,
                "intraday_score": intraday_score,
                "congress_score": float(congress["score"]),
                "insider_score": float(insider["score"]),
                "market_score": float(market_score),
                "exit_signal": exit_signal,
                "open_price": float(intraday.get("session_open", row.get("open_price", 0.0))),
                "last_price": float(intraday.get("last_price", row.get("last_price", 0.0))),
                "open_to_current_pct": float(intraday.get("day_return", row.get("open_to_current_pct", 0.0))),
                "congress_buys": int(congress["buys"]),
                "congress_sales": int(congress["sales"]),
                "insider_buys": int(insider["buys"]),
                "insider_sales": int(insider["sales"]),
                "reason": integrated_recommendation_reason(
                    flow_score=flow_score,
                    theme_score=theme_score,
                    momentum_score=momentum_score,
                    intraday_score=intraday_score,
                    congress=congress,
                    insider=insider,
                    exit_signal=exit_signal,
                ),
            }
        )
    return sorted(recommendations, key=lambda item: float(item["integrated_score"]), reverse=True)[:limit]


def integrated_recommendation_reason(
    *,
    flow_score: float,
    theme_score: float,
    momentum_score: float,
    intraday_score: float,
    congress: Mapping[str, object],
    insider: Mapping[str, object],
    exit_signal: str,
) -> str:
    reasons = [
        f"flow {flow_score:.0f}",
        f"theme {theme_score:.0f}",
        f"momentum {momentum_score:.0f}",
        f"5m {intraday_score:.0f}",
    ]
    if int(congress["buys"]) or int(congress["sales"]):
        reasons.append(f"Congress {int(congress['buys'])}B/{int(congress['sales'])}S")
    if int(insider["buys"]) or int(insider["sales"]):
        reasons.append(f"insider {int(insider['buys'])}B/{int(insider['sales'])}S")
    if exit_signal in {"Trim", "Exit"}:
        reasons.append(f"risk signal: {exit_signal}")
    return "; ".join(reasons) + "."


def build_risk_watchlist(integrated_rows, limit: int = 5) -> list[dict[str, object]]:
    """Rank stocks that combine weak factors with active intraday risk signals."""
    risks = []
    for row in _iter_records(integrated_rows):
        exit_signal = str(row.get("exit_signal", "Watch"))
        integrated_score = float(row.get("integrated_score", 50.0))
        flow_score = float(row.get("flow_score", 50.0))
        momentum_score = float(row.get("momentum_score", 50.0))
        intraday_score = float(row.get("intraday_score", 50.0))
        insider_score = float(row.get("insider_score", 50.0))
        congress_score = float(row.get("congress_score", 50.0))
        signal_penalty = {"Exit": 25.0, "Trim": 12.0, "Watch": 3.0, "Hold": 0.0}.get(exit_signal, 3.0)
        risk_score = (
            (100.0 - integrated_score) * 0.55
            + (100.0 - flow_score) * 0.15
            + (100.0 - momentum_score) * 0.10
            + (100.0 - intraday_score) * 0.10
            + (100.0 - insider_score) * 0.05
            + (100.0 - congress_score) * 0.05
            + signal_penalty
        )
        risk_score = min(100.0, max(0.0, risk_score))
        risk_level = (
            "Avoid"
            if risk_score >= 70
            else "High Risk"
            if risk_score >= 55
            else "Watch"
            if risk_score >= 40
            else "Normal"
        )
        reasons = []
        if exit_signal in {"Exit", "Trim"}:
            reasons.append(f"5m signal is {exit_signal}")
        if flow_score < 45:
            reasons.append(f"weak flow {flow_score:.0f}")
        if momentum_score < 40:
            reasons.append(f"weak momentum {momentum_score:.0f}")
        if insider_score < 40:
            reasons.append("insider selling pressure")
        if congress_score < 40:
            reasons.append("Congress selling pressure")
        if not reasons:
            reasons.append(f"overall score only {integrated_score:.0f}")
        risks.append(
            {
                **row,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_reason": "; ".join(reasons[:3]) + ".",
            }
        )
    return sorted(risks, key=lambda item: float(item["risk_score"]), reverse=True)[:limit]


def _disclosure_scores(rows, value_fields: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for row in _iter_records(rows):
        ticker = str(row.get("ticker", "")).upper()
        side = str(row.get("trade_side", ""))
        if not ticker or side not in {"Purchase", "Sale"}:
            continue
        value = _disclosure_value(row, value_fields)
        aggregate = aggregates.setdefault(ticker, {"buy_value": 0.0, "sale_value": 0.0, "buys": 0, "sales": 0})
        if side == "Purchase":
            aggregate["buy_value"] = float(aggregate["buy_value"]) + value
            aggregate["buys"] = int(aggregate["buys"]) + 1
        else:
            aggregate["sale_value"] = float(aggregate["sale_value"]) + value
            aggregate["sales"] = int(aggregate["sales"]) + 1

    results = {}
    for ticker, aggregate in aggregates.items():
        buy_value = float(aggregate["buy_value"])
        sale_value = float(aggregate["sale_value"])
        total_value = buy_value + sale_value
        if total_value:
            score = 50.0 + 50.0 * ((buy_value - sale_value) / total_value)
        else:
            total_count = int(aggregate["buys"]) + int(aggregate["sales"])
            score = 50.0 if not total_count else 50.0 + 50.0 * (
                (int(aggregate["buys"]) - int(aggregate["sales"])) / total_count
            )
        results[ticker] = {**aggregate, "score": min(100.0, max(0.0, score))}
    return results


def _disclosure_value(row: Mapping[str, object], value_fields: tuple[str, ...]) -> float:
    values = [float(row.get(field, 0.0) or 0.0) for field in value_fields]
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return sum(positive) / len(positive) if len(value_fields) > 1 else positive[0]


def _neutral_disclosure_score() -> dict[str, float | int]:
    return {"buy_value": 0.0, "sale_value": 0.0, "buys": 0, "sales": 0, "score": 50.0}


def _integrated_rating(score: float, exit_signal: str) -> str:
    if exit_signal == "Exit":
        return "Caution"
    if score >= 75:
        return "High Conviction"
    if score >= 65:
        return "Positive"
    if score >= 50:
        return "Neutral"
    return "Caution"


def build_breakout_candidates(component_rows, limit: int = 5) -> list[dict[str, object]]:
    """Rank single-stock breakout candidates independent of their theme basket score."""
    candidates = []
    for row in _iter_records(component_rows):
        flow_score = float(row.get("flow_score", 0.0))
        open_to_current_pct = float(row.get("open_to_current_pct", 0.0))
        return_1d = float(row.get("return_1d", 0.0))
        dollar_volume_trend = float(row.get("dollar_volume_trend", 0.0))
        volume_zscore = float(row.get("volume_zscore", 0.0))
        breakout_score = (
            normalize(open_to_current_pct, 0.0, 8.0) * 0.25
            + normalize(return_1d, 0.0, 8.0) * 0.20
            + normalize(volume_zscore, 0.0, 2.5) * 0.20
            + normalize(dollar_volume_trend, 0.0, 100.0) * 0.20
            + normalize(flow_score, 70.0, 100.0) * 0.15
        )
        themes = [theme.strip() for theme in str(row.get("themes", "")).split(",") if theme.strip()]
        candidates.append(
            {
                "ticker": row.get("ticker", ""),
                "themes": ", ".join(themes),
                "open_price": float(row.get("open_price", 0.0)),
                "last_price": float(row.get("last_price", 0.0)),
                "open_to_current_pct": open_to_current_pct,
                "flow_score": flow_score,
                "breakout_score": breakout_score,
                "return_1d": return_1d,
                "return_5d": float(row.get("return_5d", 0.0)),
                "return_20d": float(row.get("return_20d", 0.0)),
                "relative_5d": float(row.get("relative_5d", 0.0)),
                "dollar_volume_trend": dollar_volume_trend,
                "volume_zscore": volume_zscore,
                "reason": breakout_reason(row, breakout_score),
            }
        )
    return sorted(candidates, key=lambda item: float(item["breakout_score"]), reverse=True)[:limit]


def build_intraday_breakout_candidates(
    intraday_rows,
    limit: int = 5,
    min_dollar_volume_m: float = 3.0,
) -> list[dict[str, object]]:
    """Rank 5-minute intraday breakout candidates for same-session monitoring.

    Ranking is cross-sectional (percentile within today's universe) rather than
    against fixed thresholds, so quiet days and hot days both produce a usable
    ordering. Illiquid rows (last-30m dollar volume below the floor) are
    excluded so top picks stay tradeable; if nothing passes the floor the
    filter is relaxed instead of returning nothing.
    """
    rows = list(_iter_records(intraday_rows))
    liquid = [row for row in rows if float(row.get("recent_dollar_volume_m", 0.0)) >= min_dollar_volume_m]
    universe = liquid or rows

    vs_spy_values = [float(row.get("vs_spy_pct", 0.0)) for row in universe]
    rvol_values = [float(row.get("rvol", 1.0)) for row in universe]
    day_values = [float(row.get("day_return", 0.0)) for row in universe]
    mom_values = [float(row.get("return_30m", 0.0)) for row in universe]

    candidates = []
    for row in universe:
        day_return = float(row.get("day_return", 0.0))
        gap_pct = float(row.get("gap_pct", 0.0))
        vwap = float(row.get("vwap", 0.0))
        orb_high = float(row.get("orb_high", 0.0))
        orb_low = float(row.get("orb_low", 0.0))
        below_vwap = bool(row.get("below_vwap", False))
        above_orb_high = bool(row.get("above_orb_high", False))
        gap_fade = gap_pct >= 1.5 and day_return <= -0.5

        breakout_score = (
            _pct_rank(vs_spy_values, float(row.get("vs_spy_pct", 0.0))) * 0.30
            + _pct_rank(rvol_values, float(row.get("rvol", 1.0))) * 0.20
            + _pct_rank(day_values, day_return) * 0.15
            + _pct_rank(mom_values, float(row.get("return_30m", 0.0))) * 0.10
            + (0.0 if below_vwap else 100.0) * 0.15
            + (100.0 if above_orb_high else 0.0) * 0.10
        )
        if gap_fade:
            breakout_score = max(0.0, breakout_score - 15.0)

        candidates.append(
            {
                "ticker": row.get("ticker", ""),
                "themes": row.get("themes", ""),
                "last_time": row.get("last_time", ""),
                "session_open": float(row.get("session_open", 0.0)),
                "last_price": float(row.get("last_price", 0.0)),
                "gap_pct": gap_pct,
                "day_return": day_return,
                "day_change_pct": float(row.get("day_change_pct", day_return)),
                "vs_spy_pct": float(row.get("vs_spy_pct", 0.0)),
                "return_30m": float(row.get("return_30m", 0.0)),
                "return_60m": float(row.get("return_60m", 0.0)),
                "vwap": vwap,
                "vwap_gap_pct": float(row.get("vwap_gap_pct", 0.0)),
                "below_vwap": below_vwap,
                "rvol": float(row.get("rvol", 1.0)),
                "volume_trend": float(row.get("volume_trend", 0.0)),
                "recent_dollar_volume_m": float(row.get("recent_dollar_volume_m", 0.0)),
                "orb_high": orb_high,
                "orb_low": orb_low,
                "above_orb_high": above_orb_high,
                "gap_fade": gap_fade,
                "entry_ref": max(orb_high, vwap),
                "stop_ref": orb_low if orb_low else vwap,
                "breakout_score": breakout_score,
                "reason": intraday_breakout_reason(row, breakout_score),
                **intraday_exit_signal(row),
            }
        )
    return sorted(candidates, key=lambda item: float(item["breakout_score"]), reverse=True)[:limit]


def _pct_rank(values: list[float], value: float) -> float:
    """Midrank percentile of value within values, on a 0-100 scale."""
    if not values:
        return 50.0
    below = sum(1 for item in values if item < value)
    below_or_equal = sum(1 for item in values if item <= value)
    return (below + below_or_equal) / (2 * len(values)) * 100


def intraday_exit_signal(row: Mapping[str, object]) -> dict[str, str]:
    """Classify whether a 5m breakout candidate still deserves holding."""
    day_return = float(row.get("day_return", 0.0))
    gap_pct = float(row.get("gap_pct", 0.0))
    return_30m = float(row.get("return_30m", 0.0))
    return_60m = float(row.get("return_60m", 0.0))
    vwap_gap_pct = float(row.get("vwap_gap_pct", 0.0))
    volume_trend = float(row.get("volume_trend", 0.0))
    below_vwap = bool(row.get("below_vwap", False))

    if gap_pct >= 2.0 and day_return <= -1.0:
        return {
            "exit_signal": "Exit",
            "exit_reason": "跳空高開後開盤即回落逾 1%，常見出貨型態。",
        }
    if below_vwap and return_30m < 0:
        return {
            "exit_signal": "Exit",
            "exit_reason": "跌破 VWAP 且近 30 分鐘動能轉負。",
        }
    if return_30m <= -0.75 and return_60m <= -0.50:
        return {
            "exit_signal": "Exit",
            "exit_reason": "30 分鐘與 60 分鐘動能同步走弱。",
        }
    if day_return >= 5 and return_30m < 0:
        return {
            "exit_signal": "Trim",
            "exit_reason": "當日漲幅已大但短線動能降溫，先減碼。",
        }
    if below_vwap:
        return {
            "exit_signal": "Trim",
            "exit_reason": "價格位於 VWAP 之下，未快速收復前先降低風險。",
        }
    if return_30m < 0 and volume_trend < 0:
        return {
            "exit_signal": "Trim",
            "exit_reason": "動能與 5 分鐘量能同步降溫。",
        }
    if vwap_gap_pct >= 0 and return_30m >= 0:
        return {
            "exit_signal": "Hold",
            "exit_reason": "站上 VWAP 且 30 分鐘動能未轉負。",
        }
    return {
        "exit_signal": "Watch",
        "exit_reason": "盤中訊號混合，等 VWAP 或 30 分鐘動能確認。",
    }


def intraday_breakout_reason(row: Mapping[str, object], breakout_score: float) -> str:
    reasons = []
    day_return = float(row.get("day_return", 0.0))
    gap_pct = float(row.get("gap_pct", 0.0))
    vs_spy_pct = float(row.get("vs_spy_pct", 0.0))
    rvol = float(row.get("rvol", 1.0))
    return_30m = float(row.get("return_30m", 0.0))
    vwap_gap_pct = float(row.get("vwap_gap_pct", 0.0))
    below_vwap = bool(row.get("below_vwap", False))
    above_orb_high = bool(row.get("above_orb_high", False))

    if gap_pct >= 1.5 and day_return <= -0.5:
        reasons.append(f"跳空 {gap_pct:+.1f}% 後回落 {day_return:+.1f}%，慎防假突破")
    elif gap_pct >= 1.0 and day_return >= 0:
        reasons.append(f"跳空 {gap_pct:+.1f}% 後守住開盤價")
    if vs_spy_pct >= 0.5:
        reasons.append(f"當日跑贏 SPY {vs_spy_pct:+.1f}%")
    if rvol >= 1.5:
        reasons.append(f"同時段相對量 {rvol:.1f}x")
    if above_orb_high:
        reasons.append("突破開盤 30 分鐘高點")
    if vwap_gap_pct > 0 and not below_vwap:
        reasons.append(f"高於 VWAP {vwap_gap_pct:+.1f}%")
    if return_30m >= 1:
        reasons.append(f"近 30 分鐘動能 {return_30m:+.1f}%")

    if not reasons:
        reasons.append(f"盤中綜合分數 {breakout_score:.1f}/100")

    return "; ".join(reasons[:4]) + "。"


def breakout_reason(row: Mapping[str, object], breakout_score: float) -> str:
    reasons = []
    flow_score = float(row.get("flow_score", 0.0))
    open_to_current_pct = float(row.get("open_to_current_pct", 0.0))
    return_1d = float(row.get("return_1d", 0.0))
    dollar_volume_trend = float(row.get("dollar_volume_trend", 0.0))
    volume_zscore = float(row.get("volume_zscore", 0.0))

    if open_to_current_pct >= 5:
        reasons.append(f"open-to-current move is {open_to_current_pct:+.1f}%")
    elif open_to_current_pct > 0:
        reasons.append(f"open-to-current move is positive at {open_to_current_pct:+.1f}%")

    if return_1d >= 5:
        reasons.append(f"1D move is {return_1d:+.1f}%")
    elif return_1d > 0:
        reasons.append(f"1D move is positive at {return_1d:+.1f}%")

    if volume_zscore >= 1:
        reasons.append(f"volume shock is {volume_zscore:+.1f} z-score")
    if dollar_volume_trend >= 25:
        reasons.append(f"dollar volume trend is {dollar_volume_trend:+.1f}%")
    if flow_score >= 85:
        reasons.append(f"flow score is strong at {flow_score:.1f}/100")

    if not reasons:
        reasons.append(f"breakout setup score is {breakout_score:.1f}/100")

    return "; ".join(reasons[:4]) + "."


def recommendation_reason(row: Mapping[str, object], theme_score: float) -> str:
    reasons = []
    flow_score = float(row.get("flow_score", 0.0))
    return_5d = float(row.get("return_5d", 0.0))
    return_20d = float(row.get("return_20d", 0.0))
    relative_5d = float(row.get("relative_5d", 0.0))
    dollar_volume_trend = float(row.get("dollar_volume_trend", 0.0))
    volume_zscore = float(row.get("volume_zscore", 0.0))

    if flow_score >= 80:
        reasons.append(f"flow score is very strong at {flow_score:.1f}/100")
    elif flow_score >= 65:
        reasons.append(f"flow score is positive at {flow_score:.1f}/100")
    else:
        reasons.append(f"flow score is improving but not extended at {flow_score:.1f}/100")

    if relative_5d > 0:
        reasons.append(f"outperforming SPY by {relative_5d:+.1f}% over 5D")
    if return_20d > 0:
        reasons.append(f"20D momentum is {return_20d:+.1f}%")
    elif return_5d > 0:
        reasons.append(f"short-term 5D momentum is {return_5d:+.1f}%")
    if dollar_volume_trend > 0:
        reasons.append(f"dollar volume trend is {dollar_volume_trend:+.1f}%")
    if volume_zscore > 0.75:
        reasons.append(f"volume is elevated at {volume_zscore:+.1f} z-score")
    if theme_score >= 70:
        reasons.append(f"related theme basket is strong at {theme_score:.1f}/100")

    return "; ".join(reasons[:4]) + "."


def _iter_records(rows):
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return rows


def broad_flow_score(sector_scores: Mapping[str, float]) -> float:
    if not sector_scores:
        return 0.0
    return sum(sector_scores.values()) / len(sector_scores)


def classify_regime(broad_score: float, risk_on_score: float, defensive_score: float) -> FlowRegime:
    leadership_gap = risk_on_score - defensive_score
    if broad_score >= 55 and leadership_gap >= 5:
        return FlowRegime("Risk-On Accumulation", "#2EA043", "risk-on", "+")
    if defensive_score >= 55 and leadership_gap <= -5:
        return FlowRegime("Defensive Rotation", "#D29922", "defensive", "!")
    if broad_score < 45 and defensive_score < 55:
        return FlowRegime("Broad Distribution", "#F85149", "distribution", "-")
    return FlowRegime("Mixed Rotation", "#58A6FF", "mixed", "=")


def market_timing_signal(benchmark_rows, broad_score: float, risk_on_score: float) -> MarketTimingSignal:
    """Evaluate whether broad-market weakness argues for waiting or re-entry."""
    benchmarks = {str(row.get("ticker")): row for row in _iter_records(benchmark_rows)}
    core = [ticker for ticker in ["SPY", "QQQ", "IWM"] if ticker in benchmarks]

    weak_count = 0
    recovery_count = 0
    evidence: list[str] = []

    for ticker in core:
        row = benchmarks[ticker]
        ret_1d = float(row.get("return_1d", 0.0))
        ret_5d = float(row.get("return_5d", 0.0))
        ret_20d = float(row.get("return_20d", 0.0))
        if ret_1d < -0.5 and ret_5d < -2.0 and ret_20d < -3.0:
            weak_count += 1
            evidence.append(f"{ticker} is still falling: 1D {ret_1d:+.1f}%, 5D {ret_5d:+.1f}%, 20D {ret_20d:+.1f}%")
        if ret_1d > 0.0 and ret_5d > 1.0 and ret_20d > -3.0:
            recovery_count += 1
            evidence.append(f"{ticker} is stabilizing: 1D {ret_1d:+.1f}%, 5D {ret_5d:+.1f}%, 20D {ret_20d:+.1f}%")

    broad_weak = broad_score < 45 or risk_on_score < 45
    broad_recovering = broad_score >= 50 and risk_on_score >= 50
    if broad_weak:
        evidence.append(f"Broad/risk-on flow is weak: broad {broad_score:.1f}, AI compute {risk_on_score:.1f}")
    if broad_recovering:
        evidence.append(f"Flow backdrop is recovering: broad {broad_score:.1f}, AI compute {risk_on_score:.1f}")

    if weak_count >= 2 and broad_weak:
        return MarketTimingSignal(
            status="stand_aside",
            severity="critical",
            title="大盤短線連續走弱，暫時不要進場",
            message="SPY/QQQ/IWM 多數仍在短線下跌，且資金流尚未修復。優先等待回穩訊號。",
            score=25.0,
            evidence=evidence[:5],
        )

    if recovery_count >= 2 and broad_recovering:
        return MarketTimingSignal(
            status="recovery_confirmed",
            severity="info",
            title="市場已出現回穩訊號，可以開始評估進場",
            message="主要指數短線轉強，且 broad/risk-on flow 回到可接受區間。可分批觀察強勢主題。",
            score=75.0,
            evidence=evidence[:5],
        )

    return MarketTimingSignal(
        status="wait_for_confirmation",
        severity="warning",
        title="大盤尚未給出明確進場訊號",
        message="短線下跌或回穩條件尚未同時確認。等待 SPY/QQQ/IWM 與資金流同步改善。",
        score=50.0,
        evidence=evidence[:5] or ["Benchmark data is mixed; no clean timing signal."],
    )


def intraday_market_signal(intraday_rows) -> IntradayMarketSignal:
    """Evaluate 5-minute broad-market pressure for same-day entry timing."""
    rows = list(_iter_records(intraday_rows))
    if not rows:
        return IntradayMarketSignal(
            status="intraday_unavailable",
            severity="warning",
            title="盤中 5m 資料暫時無法取得",
            message="Yahoo Finance 盤中資料可能延遲或暫時漏抓，先以日線大盤訊號為主。",
            score=0.0,
            evidence=["No intraday rows were available."],
        )

    weak_count = 0
    recovery_count = 0
    evidence: list[str] = []

    for row in rows:
        ticker = str(row.get("ticker", ""))
        # Prefer the gap-aware change from the prior close when available.
        day_return = float(row.get("day_change_pct", row.get("day_return", 0.0)))
        return_30m = float(row.get("return_30m", 0.0))
        return_60m = float(row.get("return_60m", 0.0))
        below_vwap = bool(row.get("below_vwap", False))

        if day_return <= -0.8 and return_30m <= -0.2 and below_vwap:
            weak_count += 1
            evidence.append(f"{ticker} intraday weak: day {day_return:+.2f}%, 30m {return_30m:+.2f}%, below VWAP")
        if day_return >= -0.25 and return_30m > 0 and return_60m > -0.3 and not below_vwap:
            recovery_count += 1
            evidence.append(f"{ticker} stabilizing: day {day_return:+.2f}%, 30m {return_30m:+.2f}%, above VWAP")

    if weak_count >= 2:
        return IntradayMarketSignal(
            status="intraday_stand_aside",
            severity="critical",
            title="盤中 5m 急跌風險，暫時不要進場",
            message="SPY/QQQ/IWM 多數低於 VWAP 且短線仍走弱，等待盤中止跌或回到 VWAP 上方。",
            score=20.0,
            evidence=evidence[:5],
        )

    if recovery_count >= 2:
        return IntradayMarketSignal(
            status="intraday_recovery",
            severity="info",
            title="盤中 5m 回穩訊號出現，可以開始觀察進場",
            message="主要指數盤中站回 VWAP 附近或上方，且短線動能轉正。可搭配日線訊號分批觀察。",
            score=80.0,
            evidence=evidence[:5],
        )

    return IntradayMarketSignal(
        status="intraday_wait",
        severity="warning",
        title="盤中 5m 訊號尚未明確",
        message="盤中賣壓與回穩條件都未充分確認，等待 SPY/QQQ/IWM 方向更清楚。",
        score=50.0,
        evidence=evidence[:5] or ["Intraday benchmark conditions are mixed."],
    )


SEMIS_THEMES = THEME_GROUPS["AI Compute Chain"]


def intraday_entry_gate(intraday_market_rows, intraday_theme_rows) -> IntradayEntryGate:
    """Decide whether today's session is worth entering at all.

    Combines benchmark pressure with theme-level breadth so that a day where
    most themes are bleeding produces an explicit "do not enter" warning, with
    a dedicated call-out when the semiconductor chain is the weak spot.
    """
    market = list(_iter_records(intraday_market_rows))
    themes = list(_iter_records(intraday_theme_rows))
    if not themes:
        return IntradayEntryGate(
            status="unavailable",
            severity="warning",
            title="盤中主題資料暫時無法取得",
            message="無法計算今日主題資金流向，先以大盤 5m 訊號與日線訊號為主。",
            evidence=[],
            negative_theme_share=0.0,
            semis_day_change=0.0,
            semis_weak=False,
        )

    weak_benchmarks = [
        str(row.get("ticker", ""))
        for row in market
        if float(row.get("day_change_pct", row.get("day_return", 0.0))) <= -0.5 and bool(row.get("below_vwap", False))
    ]
    negative_themes = [row for row in themes if float(row.get("day_change_pct", 0.0)) < 0]
    negative_share = len(negative_themes) / len(themes)
    above_vwap_values = sorted(float(row.get("pct_above_vwap", 0.0)) for row in themes)
    median_above_vwap = above_vwap_values[len(above_vwap_values) // 2]

    semis = [row for row in themes if str(row.get("theme", "")) in SEMIS_THEMES]
    semis_changes = [float(row.get("day_change_pct", 0.0)) for row in semis]
    semis_day_change = sum(semis_changes) / len(semis_changes) if semis_changes else 0.0
    semis_weak = bool(semis_changes) and (semis_day_change <= -1.0 or all(value < 0 for value in semis_changes))

    evidence = [
        f"{len(weak_benchmarks)}/3 檔大盤指數走弱（含跳空跌逾 0.5% 且低於 VWAP）" + (f"：{', '.join(weak_benchmarks)}" if weak_benchmarks else ""),
        f"{len(negative_themes)}/{len(themes)} 個主題今日下跌",
        f"主題中位數僅 {median_above_vwap:.0f}% 成分股站上 VWAP",
    ]
    if semis_changes:
        evidence.append(
            f"半導體鏈（記憶體/光通訊/CPU 封裝/AI 基建）今日平均 {semis_day_change:+.2f}%"
            + ("，全數下跌" if all(value < 0 for value in semis_changes) else "")
        )

    if len(weak_benchmarks) >= 2 or (negative_share >= 0.7 and median_above_vwap < 40):
        message = "大盤與多數主題同步走弱，今晚進場勝率偏低。建議觀望，等待止跌或資金明確回流再操作。"
        if semis_weak:
            message += "半導體相關類股流出特別明顯，記憶體、光通訊、CPU/封裝與 AI 基建先避開。"
        return IntradayEntryGate(
            status="no_entry",
            severity="critical",
            title="今日資金流出居多，不建議進場",
            message=message,
            evidence=evidence,
            negative_theme_share=negative_share,
            semis_day_change=semis_day_change,
            semis_weak=semis_weak,
        )

    if negative_share >= 0.5 or len(weak_benchmarks) == 1 or semis_weak:
        message = "部分主題仍有資金流入，但整體風險偏高。只考慮跑贏 SPY、量能放大且守住 VWAP 的標的，並縮小部位。"
        if semis_weak:
            message += "注意：半導體鏈今日以流出為主，該族群暫勿進場。"
        return IntradayEntryGate(
            status="caution",
            severity="warning",
            title="資金流向分歧，僅適合小倉位選股",
            message=message,
            evidence=evidence,
            negative_theme_share=negative_share,
            semis_day_change=semis_day_change,
            semis_weak=semis_weak,
        )

    return IntradayEntryGate(
        status="ok",
        severity="info",
        title="今日資金以流入為主，可依候選清單選股",
        message="多數主題上漲且守住 VWAP。優先挑選盤中主題看板前段、同時段相對量 1.2x 以上的標的。",
        evidence=evidence,
        negative_theme_share=negative_share,
        semis_day_change=semis_day_change,
        semis_weak=semis_weak,
    )


def flow_delta(history: list[dict[str, object]], current_score: float, hours_back: int, now: dt.datetime) -> float | None:
    if len(history) < 2:
        return None
    cutoff = now - dt.timedelta(hours=hours_back)
    past_records = []
    for record in history[:-1]:
        try:
            record_time = dt.datetime.strptime(str(record["time"]), "%Y-%m-%d %H:%M")
        except (KeyError, TypeError, ValueError):
            continue
        if record_time <= cutoff:
            past_records.append(record)
    ref = past_records[-1] if past_records else history[0]
    try:
        return current_score - float(ref["broad_flow_score"])
    except (KeyError, TypeError, ValueError):
        return None

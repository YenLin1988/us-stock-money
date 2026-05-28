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

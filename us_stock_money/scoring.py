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

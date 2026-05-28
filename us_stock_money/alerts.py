"""Alert rule evaluation for sector money-flow regimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .model_config import ALERT_THRESHOLDS


@dataclass(frozen=True)
class Alert:
    key: str
    severity: str
    title: str
    message: str
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_alerts(context: dict[str, Any]) -> list[Alert]:
    alerts: list[Alert] = []
    broad = _num(context.get("broad_flow_score"))
    risk_on = _num(context.get("risk_on_score"))
    defensive = _num(context.get("defensive_score"))
    delta_24h = _num(context.get("delta_24h"))
    regime = str(context.get("regime", ""))
    market_timing_status = str(context.get("market_timing_status", ""))
    market_timing_title = str(context.get("market_timing_title", ""))
    market_timing_message = str(context.get("market_timing_message", ""))
    intraday_status = str(context.get("intraday_status", ""))
    intraday_title = str(context.get("intraday_title", ""))
    intraday_message = str(context.get("intraday_message", ""))

    if broad is not None and broad <= ALERT_THRESHOLDS["broad_distribution"]:
        alerts.append(Alert("broad-distribution", "critical", "Broad equity distribution", f"Broad flow score is {broad:.1f}/100.", broad))
    elif broad is not None and broad >= ALERT_THRESHOLDS["strong_inflow"]:
        alerts.append(Alert("strong-inflow", "info", "Strong broad equity inflow", f"Broad flow score is {broad:.1f}/100.", broad))

    if defensive is not None and defensive >= ALERT_THRESHOLDS["defensive_rotation"] and "Defensive" in regime:
        alerts.append(Alert("defensive-rotation", "warning", "Defensive rotation detected", f"Defensive flow score is {defensive:.1f}/100.", defensive))

    if risk_on is not None and risk_on >= ALERT_THRESHOLDS["risk_on_leadership"] and "Risk-On" in regime:
        alerts.append(Alert("risk-on-leadership", "info", "Risk-on leadership confirmed", f"Risk-on flow score is {risk_on:.1f}/100.", risk_on))

    if delta_24h is not None and delta_24h <= -8:
        alerts.append(Alert("flow-velocity-down", "warning", "Money flow deteriorated over 24H", f"24H flow delta is {delta_24h:+.1f}.", delta_24h))

    if market_timing_status == "stand_aside":
        alerts.append(Alert("market-stand-aside", "critical", market_timing_title, market_timing_message))
    elif market_timing_status == "recovery_confirmed":
        alerts.append(Alert("market-recovery-confirmed", "info", market_timing_title, market_timing_message))

    if intraday_status == "intraday_stand_aside":
        alerts.append(Alert("intraday-stand-aside", "critical", intraday_title, intraday_message))
    elif intraday_status == "intraday_recovery":
        alerts.append(Alert("intraday-recovery", "info", intraday_title, intraday_message))

    return alerts


def utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

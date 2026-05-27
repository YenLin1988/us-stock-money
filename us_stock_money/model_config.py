"""Model configuration for US STOCK MONEY."""

from __future__ import annotations

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "IWF": "Growth",
    "IWD": "Value",
}

RISK_ON = {"XLK", "XLC", "XLY", "XLF", "XLI"}
DEFENSIVE = {"XLP", "XLV", "XLU"}
CYCLICAL = {"XLY", "XLE", "XLF", "XLI", "XLB"}
RATE_SENSITIVE = {"XLRE", "XLU", "XLF"}

GROUPS = {
    "Risk-On": RISK_ON,
    "Defensive": DEFENSIVE,
    "Cyclicals": CYCLICAL,
    "Rate Sensitive": RATE_SENSITIVE,
}

FLOW_WEIGHTS = {
    "return_1d": 0.15,
    "return_5d": 0.20,
    "return_20d": 0.20,
    "relative_5d": 0.20,
    "dollar_volume_trend": 0.15,
    "volume_zscore": 0.10,
}

ALERT_THRESHOLDS = {
    "broad_distribution": 35,
    "strong_inflow": 70,
    "defensive_rotation": 60,
    "risk_on_leadership": 60,
}

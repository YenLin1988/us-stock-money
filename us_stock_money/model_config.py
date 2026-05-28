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

THEME_BASKETS = {
    "Memory / HBM": {
        "description": "DRAM, NAND, storage, and high-bandwidth memory proxies",
        "tickers": ["MU", "WDC", "STX", "RMBS", "SIMO", "SNDK"],
    },
    "Optical Communication": {
        "description": "AI data-center optical networking and transport equipment",
        "tickers": ["COHR", "LITE", "CIEN", "FN", "AAOI", "NOK", "MRVL", "GLW", "KOPN", "AXTI"],
    },
    "Electric Power / Grid": {
        "description": "Power generation, grid equipment, electrification, and AI data-center power demand",
        "tickers": ["CEG", "VST", "NEE", "ETN", "PWR", "GEV", "GNRC", "BE", "EOSE", "TSLA", "VRT"],
    },
    "CPU / Advanced Packaging": {
        "description": "CPU designers, foundry, packaging, and semiconductor equipment",
        "tickers": ["AMD", "INTC", "ARM", "TSM", "AMKR", "ASML", "AMAT", "KLAC", "ONTO", "ASX", "ALAB", "ALMU", "PENG"],
    },
    "Space": {
        "description": "Launch, satellites, space connectivity, and geospatial intelligence",
        "tickers": ["RKLB", "ASTS", "LUNR", "IRDM", "PL", "VSAT", "RDW", "SIDU"],
    },
    "Drones / Autonomy": {
        "description": "Unmanned systems, defense autonomy, and tactical aerospace",
        "tickers": ["AVAV", "KTOS", "RCAT", "ONDS", "LMT", "NOC"],
    },
    "Rare Earth / Strategic Metals": {
        "description": "Rare earths, uranium/critical minerals, lithium, and strategic materials",
        "tickers": ["REMX", "MP", "UUUU", "LAC", "ALB", "SCCO"],
    },
    "Nuclear Energy": {
        "description": "Uranium, nuclear utilities, reactors, fuel cycle, and nuclear services",
        "tickers": ["NLR", "URA", "CCJ", "CEG", "BWXT", "OKLO", "SMR", "LEU", "VST"],
    },
    "Medical / Devices": {
        "description": "Healthcare, medical devices, diagnostics, and surgical robotics",
        "tickers": ["XLV", "IHI", "ISRG", "ABT", "SYK", "BSX", "TMO", "MDT"],
    },
    "AI Infrastructure": {
        "description": "AI accelerators, networking, servers, and data-center electrical infrastructure",
        "tickers": ["NVDA", "AVGO", "ANET", "VRT", "DELL", "SMCI", "ETN", "MRVL", "PENG", "ALAB", "CRWV", "NBIS", "IREN", "AMD", "INTC", "TSM", "AMAT", "OSS"],
    },
    "AI Software / Data": {
        "description": "Enterprise AI software, data platforms, cloud software, and analytics",
        "tickers": ["MSFT", "PLTR", "SNOW", "MDB", "NOW", "ORCL", "APP", "FIG"],
    },
    "Cybersecurity": {
        "description": "Cloud, endpoint, identity, and network security",
        "tickers": ["PANW", "CRWD", "ZS", "NET", "FTNT", "S"],
    },
    "Robotics / Automation": {
        "description": "Industrial automation, surgical robotics, warehouse automation, and applied robotics",
        "tickers": ["BOTZ", "ISRG", "TER", "ROK", "SYM", "HON", "KULR"],
    },
    "Defense / Aerospace": {
        "description": "Prime defense contractors, aerospace suppliers, and defense ETF proxy",
        "tickers": ["ITA", "LMT", "RTX", "NOC", "GD", "HWM"],
    },
}

THEME_GROUPS = {
    "AI Compute Chain": {"Memory / HBM", "Optical Communication", "CPU / Advanced Packaging", "AI Infrastructure"},
    "Energy / Materials": {"Electric Power / Grid", "Rare Earth / Strategic Metals", "Nuclear Energy"},
    "Defense / Space": {"Space", "Drones / Autonomy", "Defense / Aerospace"},
    "Software / Security": {"AI Software / Data", "Cybersecurity"},
    "Healthcare / Automation": {"Medical / Devices", "Robotics / Automation"},
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

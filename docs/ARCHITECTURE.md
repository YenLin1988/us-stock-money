# US STOCK MONEY Architecture

US STOCK MONEY is a sector-rotation monitor built around free public market data.

## Runtime Layers

| Layer | Path | Responsibility |
|---|---|---|
| Streamlit UI | `US_STOCK_MONEY.py` | Dashboard layout, charts, tables, alerts |
| Model config | `us_stock_money/model_config.py` | Sector ETFs, groups, weights, thresholds |
| Market data | `us_stock_money/market_data.py` | Yahoo Finance downloads and feature generation |
| Scoring | `us_stock_money/scoring.py` | Normalization, flow score, regime classification |
| Alerts | `us_stock_money/alerts.py` | Deterministic alert-rule evaluation |
| Storage | `us_stock_money/storage.py` | SQLite history archive |

## Flow Proxy Method

The app estimates sector money movement by combining:

- short-term and medium-term price momentum
- relative strength against SPY
- dollar volume trend
- volume z-score
- group leadership across risk-on, defensive, cyclicals, and rate-sensitive sectors

The output is a market-implied money flow radar, not official ETF creation/redemption or institutional order-flow data.

## Governance

`us_stock_money/model_config.py` is the source of truth for:

- sector ETF universe
- sector group mapping
- factor weights
- alert thresholds

When weights or thresholds change, update README notes so future readers know why.

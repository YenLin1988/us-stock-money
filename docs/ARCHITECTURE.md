# US STOCK MONEY Architecture

US STOCK MONEY is a thematic-rotation monitor built around free public market data.

## Runtime Layers

| Layer | Path | Responsibility |
|---|---|---|
| Streamlit UI | `US_STOCK_MONEY.py` | Dashboard layout, charts, tables, alerts |
| Model config | `us_stock_money/model_config.py` | Theme baskets, sector ETF references, groups, weights, thresholds |
| Market data | `us_stock_money/market_data.py` | Yahoo Finance downloads and theme/component feature generation |
| Scoring | `us_stock_money/scoring.py` | Normalization, flow score, regime classification |
| Alerts | `us_stock_money/alerts.py` | Deterministic alert-rule evaluation |
| Storage | `us_stock_money/storage.py` | SQLite history archive |

## Flow Proxy Method

The app estimates thematic money movement by combining:

- short-term and medium-term price momentum
- relative strength against SPY
- dollar volume trend
- volume z-score
- group leadership across AI compute, energy/materials, defense/space, software/security, and healthcare/automation themes

The output is a market-implied money flow radar, not official ETF creation/redemption or institutional order-flow data.

## Recommendation Layer

`build_top_recommendations` ranks component stocks and ETFs by a composite score:

- 70% component flow score
- 30% average related theme score

The generated reason text explains the strongest available evidence, such as relative strength versus SPY, 20D momentum, dollar-volume trend, elevated volume, and related theme strength.

## Theme Basket Method

Each theme is a manually curated basket in `THEME_BASKETS`.

The app computes a score for every component, then averages available component scores into one theme score. This makes the model usable for themes that do not have a single clean ETF, such as optical communications, advanced packaging, drones, or space.

## Governance

`us_stock_money/model_config.py` is the source of truth for:

- theme basket universe
- sector ETF reference universe
- sector group mapping
- factor weights
- alert thresholds

When weights or thresholds change, update README notes so future readers know why.

# US STOCK MONEY Architecture

US STOCK MONEY is a thematic-rotation monitor built around free public market data.

## Runtime Layers

| Layer | Path | Responsibility |
|---|---|---|
| Streamlit UI | `US_STOCK_MONEY.py` | Dashboard layout, charts, tables, alerts |
| Model config | `us_stock_money/model_config.py` | Theme baskets, sector ETF references, groups, weights, thresholds |
| Market data | `us_stock_money/market_data.py` | Yahoo Finance downloads and theme/component feature generation (daily and intraday) |
| Scoring | `us_stock_money/scoring.py` | Normalization, flow score, regime classification, intraday entry gate |
| Alerts | `us_stock_money/alerts.py` | Deterministic alert-rule evaluation |
| Storage | `us_stock_money/storage.py` | SQLite history archive and intraday pick log |
| Pick tracking | `us_stock_money/pick_tracker.py` | Fill same-day / next-day outcomes for archived intraday picks |

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

## Intraday Method

Intraday features come from Yahoo 5-minute bars (`period="5d"`), computed per ticker in `_intraday_ticker_metrics`:

- `gap_pct` (session open versus prior session close) and `day_change_pct` (gap included) are kept separate from `day_return` (open to now), so gap-and-hold and gap-and-fade setups are distinguishable.
- `rvol` compares today's cumulative session volume against the same elapsed time on prior sessions, because intraday volume is U-shaped and a flat all-day baseline always reads "elevated" near the open.
- `vs_spy_pct` measures each component's session change against SPY downloaded in the same batch.
- `orb_high` / `orb_low` capture the first 30 minutes (6 bars) as opening-range reference levels.

`build_intraday_theme_table` aggregates component metrics into a theme board (median change, median vs-SPY, median RVOL, VWAP breadth) and a composite intraday flow score.

`intraday_entry_gate` turns benchmark pressure plus theme breadth into a go / no-go verdict for the session, with a dedicated semiconductor-chain warning (AI Compute Chain theme group).

`build_intraday_breakout_candidates` ranks candidates by cross-sectional percentile (vs-SPY 30%, RVOL 20%, post-open return 15%, 30m momentum 10%, above VWAP 15%, above opening range 10%), penalizes gap-fades, and filters rows below $3M of 30-minute dollar volume.

## Pick Tracking

The UI archives the top five live 5m candidates once per session (`HistoryStore.save_intraday_picks`, first snapshot per day per ticker wins). `pick_tracker.evaluate_intraday_picks` later joins daily closes to fill same-day and next-day returns, and `pick_hit_rate_summary` reports the realized win rate.

## Governance

`us_stock_money/model_config.py` is the source of truth for:

- theme basket universe
- sector ETF reference universe
- sector group mapping
- factor weights
- alert thresholds

When weights or thresholds change, update README notes so future readers know why.

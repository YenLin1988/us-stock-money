# US STOCK MONEY

**US Thematic Money Flow Radar** - a Streamlit dashboard for tracking where money is moving across high-conviction US-listed themes.

This project follows the same monitoring logic as AI Bubble Monitor: collect market data, convert noisy signals into normalized 0-100 scores, classify the current regime, archive history, and optionally trigger alerts.

## App Navigation

The Streamlit app uses native multi-page navigation:

| Page | Purpose |
|---|---|
| Decision Dashboard | Focused recommendations, risk watchlist, and weekly money-flow direction |
| Recommendations | Integrated rankings with factor-level scores |
| Signals | Market timing, 5-minute breakouts, and exit signals |
| Dark Pool Activity | Delayed FINRA ATS volume anomalies and weekly history |
| Stock Analysis | MA60 breakdown alerts and per-stock daily technical charts |
| Disclosures | Congressional STOCK Act and SEC Form 4 transactions |
| Market Research | Weekly money-flow trends, theme rotation, watchlist, sectors, benchmarks, and component tables |
| Full Dashboard | Complete legacy dashboard and every monitoring section |

## What It Tracks Now

The dashboard is built around thematic baskets instead of only classic sectors:

| Theme | Example Proxies |
|---|---|
| Memory / HBM | MU, WDC, STX, RMBS, SIMO, SNDK |
| Optical Communication | COHR, LITE, CIEN, FN, AAOI, NOK, MRVL, GLW, KOPN |
| Electric Power / Grid | CEG, VST, NEE, ETN, PWR, GEV, GNRC, BE, EOSE, TSLA |
| CPU / Advanced Packaging | AMD, INTC, ARM, TSM, AMKR, ASML, AMAT, KLAC, ONTO, ASX, ALAB |
| Space | SPCX, RKLB, ASTS, LUNR, IRDM, PL, VSAT, RDW, SIDU |
| Drones / Autonomy | AVAV, KTOS, RCAT, ONDS, LMT, NOC |
| Rare Earth / Strategic Metals | REMX, MP, UUUU, LAC, ALB, SCCO |
| Nuclear Energy | NLR, URA, CCJ, CEG, BWXT, OKLO, SMR, LEU, VST |
| Medical / Devices | XLV, IHI, ISRG, ABT, SYK, BSX, TMO, MDT |
| AI Infrastructure | NVDA, AVGO, ANET, VRT, DELL, SMCI, ETN, PENG, CRWV, NBIS, IREN |
| AI Software / Data | MSFT, PLTR, SNOW, MDB, NOW, ORCL, APP, FIG |
| Cybersecurity | PANW, CRWD, ZS, NET, FTNT, S |
| Robotics / Automation | BOTZ, ISRG, TER, ROK, SYM, HON |
| Defense / Aerospace | ITA, LMT, RTX, NOC, GD, HWM |

Traditional sector ETFs remain available as a reference tab.

## Sector Reference

The app also monitors sector ETFs as liquid proxies for classic US equity sector rotation:

| Sector | ETF |
|---|---|
| Technology | XLK |
| Communication Services | XLC |
| Consumer Discretionary | XLY |
| Consumer Staples | XLP |
| Energy | XLE |
| Financials | XLF |
| Health Care | XLV |
| Industrials | XLI |
| Materials | XLB |
| Utilities | XLU |
| Real Estate | XLRE |

Benchmark and style proxies:

| Proxy | Ticker |
|---|---|
| S&P 500 | SPY |
| Nasdaq 100 | QQQ |
| Russell 2000 | IWM |
| Growth | IWF |
| Value | IWD |

## Model Logic

The app estimates money flow using public market data:

- basket-level price momentum over 1D, 5D, and 20D windows
- relative strength versus SPY
- dollar volume intensity
- volume z-score versus recent history
- theme-group leadership across AI compute, energy/materials, defense/space, software/security, and healthcare/automation

Because free public data does not provide official real-time fund flows, this project uses market-implied flow proxies. It is best read as a thematic rotation radar, not a complete institutional flow tape.

## Weekly Theme Trends

The Market Research page aggregates daily data into weekly theme trends:

- weekly equal-weight theme return
- relative weekly return versus SPY
- weekly dollar-volume trend versus the prior eight weeks
- normalized weekly flow score and net inflow/outflow direction
- strongest inflow, strongest outflow, biggest gain, and biggest loss rankings
- 8, 13, 26, and 52-week heatmap windows

These weekly money-flow values are price-and-volume proxies, not official fund-flow reports.

## Top 5 Flow Candidates

The dashboard ranks five component stocks or ETFs by combining:

- the component's own money-flow score
- the average strength of its related theme baskets
- recent momentum and relative strength versus SPY
- dollar-volume and volume-confirmation signals

These candidates are research signals only. They are not buy/sell instructions.

## Integrated Recommendations

The dashboard also produces a transparent integrated ranking that combines all available signals:

- component money flow: 25%
- related theme strength: 15%
- daily momentum and relative strength: 15%
- 5-minute breakout setup: 20%
- recent congressional disclosures: 10%
- recent SEC Form 4 insider activity: 10%
- broad-market timing: 5%

Missing disclosure activity receives a neutral score rather than a penalty. An active 5-minute `Trim` or `Exit` signal applies an additional risk deduction. Congressional and insider filings are delayed supporting signals and never override weak market or price action by themselves.

Recommendation cards, ranking tables, daily flow candidates, and risk watchlists link directly to each ticker's technical analysis with daily candles, MA5, MA20, MA60, volume, RSI, and MACD.

## Market Timing Signal

Before showing candidate stocks, the dashboard checks whether broad-market conditions are suitable for new entries:

- **Stand Aside**: SPY, QQQ, and IWM are mostly falling across 1D/5D/20D windows and broad/risk-on flow is weak. The dashboard warns to wait instead of entering immediately.
- **Recovery Confirmed**: major benchmarks regain short-term strength while broad/risk-on flow improves. The dashboard flags that the market has stabilized enough to start evaluating entries.
- **Wait for Confirmation**: conditions are mixed and the dashboard waits for clearer confirmation.

This signal is a market-regime filter, not personal financial advice.

## Stock Technical Analysis

The Stock Analysis page scans the tracked universe for quarterly-line risk:

- flags stocks that crossed below MA60 during the latest five trading sessions
- separates new breakdowns from stocks that remain below MA60
- shows distance to MA60 plus 5-day and 20-day returns
- links each alert to a per-stock daily candlestick chart
- overlays MA5, MA20 monthly line, and MA60 quarterly line
- includes volume, RSI (14), MACD (12/26/9), 20-day annualized volatility, 52-week range, and drawdown
- shows trailing and forward P/E plus common valuation multiples
- compares trailing and forward P/E with other tracked stocks in the same theme
- shows theme medians and the selected stock's valuation premium or discount
- shows consensus high, low, mean, and median analyst targets
- lists dated institutional target-price actions with firm, rating, prior target, and upside versus current price

Daily prices come from Yahoo Finance and can be delayed, incomplete, or adjusted after corporate actions.
Analyst target history also comes from Yahoo Finance and may not include every institution or report.

## Dark Pool Activity

The Dark Pool Activity page uses FINRA's official weekly ATS summary:

- scans the tracked universe for ATS share volume at least 2x the prior eight-week median
- also flags statistically unusual volume with a default z-score threshold of 2.5
- applies a minimum-share threshold to reduce small-base false positives
- compares ATS shares with consolidated weekly market volume
- shows FINRA publication dates, because ATS data is delayed by approximately two to four weeks
- provides per-stock weekly ATS history and links into technical analysis

FINRA's public ATS data reports aggregate volume and trade counts, not buyer- or seller-initiated direction. The dashboard therefore labels these records as unusual activity rather than buy or sell signals.

## 5m Intraday Market Monitor

The dashboard also includes a free Yahoo Finance 5-minute intraday monitor for SPY, QQQ, and IWM:

- **Intraday Stand Aside**: most major benchmarks are down on the day, still weakening over the last 30 minutes, and trading below VWAP.
- **Intraday Recovery**: most major benchmarks stabilize above VWAP with positive short-term momentum.
- **Intraday Wait**: conditions are mixed and need more confirmation.

This monitor uses `period="5d"` and `interval="5m"` with a 5-minute Streamlit cache. Yahoo intraday data can be delayed, incomplete, or temporarily rate-limited.

## Congress Stock Trades

The dashboard tracks recently disclosed congressional stock transactions collected from public House Clerk and Senate eFD STOCK Act filings:

- filter by the last 30, 90, 180, or 365 days
- filter by House or Senate and by purchase or sale
- search by ticker
- review stock-level net buying and selling rankings before opening transaction detail
- compare estimated buy, sale, and net values using disclosed range midpoints
- compare transaction dates with filing dates and disclosure delays
- open the official filing document for each transaction

Congressional trades are not real-time signals. Covered transactions may be disclosed up to 45 days after the trade date, and transaction values are reported as ranges.

## Corporate Insider Trades

The dashboard also reads the latest SEC Form 4 filings and summarizes open-market corporate insider activity:

- total shares and estimated value bought and sold
- net insider buying or selling value
- stock-level activity rankings with distinct insider counts and latest trade dates
- a compact transaction view with value, role, price, and post-transaction holdings
- company ticker, insider name, and executive/director role
- transaction price, shares, and post-transaction ownership
- direct link to the official SEC filing

Only Form 4/4-A transaction codes `P` (open-market purchase) and `S` (open-market sale) are counted. Form 10-K filings, stock awards, option exercises, gifts, and tax withholding are excluded.

Nokia (`NOK`) is handled as a supplemental foreign-issuer case using Nokia's official Article 19 manager-transaction releases. These records show the reported transaction date, manager, role, acquisition/disposal, shares, price, currency, and official source link. Congressional source coverage can legitimately return no `NOK` records.

When a complete tracked ticker is entered in the insider search, the app also requests ticker-specific Yahoo Finance insider transactions and merges them with the recent SEC feed and available official-company supplements. This improves coverage for both foreign issuers such as TSM, ARM, NXPI, and SIMO and U.S. companies that may fall outside the short latest-filing feed window. Empty results still do not prove that no transaction occurred.

## Regimes

| Regime | Condition | Meaning |
|---|---|---|
| Risk-On Accumulation | Broad flow positive and AI compute/risk themes lead | Money is moving into growth, AI, infrastructure, and higher-beta assets |
| Defensive Rotation | Healthcare/automation leadership strong while broad flow is still stable | Money is hiding in steadier or quality growth themes |
| Broad Distribution | Broad flow negative and defensive demand is not enough | Equity market outflows / de-risking pressure |
| Mixed Rotation | No dominant flow regime | Choppy sector rotation, wait for confirmation |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run US_STOCK_MONEY.py
```

The dashboard opens at `http://localhost:8501`.

## Streamlit Cloud Deployment

Use these settings on Streamlit Community Cloud:

| Setting | Value |
|---|---|
| Repository | `YenLin1988/us-stock-money` |
| Branch | `main` |
| Main file path | `US_STOCK_MONEY.py` |
| Python runtime | `python-3.11` |

No secrets are required for the default Yahoo Finance data source.

## Project Structure

```text
us-stock-money/
  US_STOCK_MONEY.py             # Main Streamlit dashboard
  us_stock_money/
    model_config.py             # Theme baskets, sector reference universe, weights, thresholds
    scoring.py                  # Flow score, regime, decomposition helpers
    market_data.py              # Yahoo Finance data collection
    congress_trades.py          # Congressional STOCK Act disclosure data
    insider_trades.py           # SEC Form 4 open-market insider trades
    alerts.py                   # Deterministic alert rules
    storage.py                  # SQLite history store
  tests/                        # Unit tests for core logic
  docs/ARCHITECTURE.md          # Runtime architecture
  docs/REALTIME_DARK_POOL_API_PLAN.md # Planned paid real-time dark-pool integration
  .github/workflows/hourly_monitor.yml
  requirements.txt
```

## Disclaimer

This tool is for education and research only. It is not financial advice. Market data may be delayed, incomplete, or revised by the data provider.

# US STOCK MONEY

**US Equity Sector Money Flow Radar** - a Streamlit dashboard for tracking where money is moving across major US stock market sectors.

This project follows the same monitoring logic as AI Bubble Monitor: collect market data, convert noisy signals into normalized 0-100 scores, classify the current regime, archive history, and optionally trigger alerts.

## What It Tracks

The dashboard monitors sector ETFs as liquid proxies for US equity money flow:

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

- price momentum over 1D, 5D, and 20D windows
- relative strength versus SPY
- dollar volume intensity
- volume z-score versus recent history
- risk-on versus defensive sector leadership
- growth versus value rotation

Because free public data does not provide official real-time fund flows, this project uses market-implied flow proxies. It is best read as a sector rotation radar, not a complete institutional flow tape.

## Regimes

| Regime | Condition | Meaning |
|---|---|---|
| Risk-On Accumulation | Broad flow positive and cyclical/risk sectors lead | Money is moving into growth, tech, discretionary, and higher-beta assets |
| Defensive Rotation | Defensive leadership strong while broad flow is still stable | Money is hiding in staples, utilities, health care, or low-beta sectors |
| Broad Distribution | Broad flow negative and defensive demand is not enough | Equity market outflows / de-risking pressure |
| Mixed Rotation | No dominant flow regime | Choppy sector rotation, wait for confirmation |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run US_STOCK_MONEY.py
```

The dashboard opens at `http://localhost:8501`.

## Project Structure

```text
us-stock-money/
  US_STOCK_MONEY.py             # Main Streamlit dashboard
  us_stock_money/
    model_config.py             # Sector universe, weights, thresholds
    scoring.py                  # Flow score, regime, decomposition helpers
    market_data.py              # Yahoo Finance data collection
    alerts.py                   # Deterministic alert rules
    storage.py                  # SQLite history store
  tests/                        # Unit tests for core logic
  docs/ARCHITECTURE.md          # Runtime architecture
  .github/workflows/hourly_monitor.yml
  requirements.txt
```

## Disclaimer

This tool is for education and research only. It is not financial advice. Market data may be delayed, incomplete, or revised by the data provider.

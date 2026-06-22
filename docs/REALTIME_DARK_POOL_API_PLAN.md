# Real-Time Dark Pool API Plan

Status: Planned, not purchased
Decision recorded: 2026-06-22

## Goal

Add near-real-time off-exchange trade monitoring while retaining the existing delayed FINRA ATS page as the official historical baseline.

## Preferred Provider

Start with a one-month trial of Massive's real-time US stocks plan.

Estimated individual research cost at the time of this decision:

- Massive real-time individual plan: about USD 199 per month
- Approximate TWD budget: TWD 6,400 per month, before card fees and exchange-rate changes

If this Streamlit app is made public or commercial, confirm redistribution and display rights before integration. A business data license may cost around USD 2,000 per month or more.

## Alternatives

| Provider | Estimated Cost | Use Case |
|---|---:|---|
| Unusual Whales API | About USD 150/month and up | Faster integration with pre-classified dark-pool endpoints |
| Databento Standard | About USD 199/month | Professional tick data and research |
| Intrinio delayed SIP | Enterprise quote plus possible display fees | Full-market data with a 15-minute delay |

All prices are planning estimates. Recheck current pricing, exchange fees, API limits, and display licensing before purchase.

## Planned Integration

1. Store the API key in Streamlit secrets or an environment variable. Never commit it.
2. Run a persistent WebSocket ingestion worker outside the Streamlit request lifecycle.
3. Store normalized trade events in SQLite or a production time-series database.
4. Identify off-exchange/TRF prints using venue and trade-condition codes.
5. Retain the existing FINRA weekly ATS data for delayed verification and historical baselines.
6. Add a live scanner view with filters for ticker, value, size, time, and anomaly score.

## Proposed Alert Logic

Do not use volume multiple alone. Combine:

- individual trade value of at least USD 250,000
- trade size at least 3x the ticker's trailing 30-day off-exchange median print
- repeated large prints near the same price within five minutes
- abnormal intraday off-exchange share of consolidated volume
- execution price relative to contemporaneous NBBO
- exclusion of delayed, out-of-sequence, corrected, and after-hours condition codes where appropriate

Direction labels must remain estimates:

- near ask: `Buy-like`
- near bid: `Sell-like`
- between bid and ask: `Neutral`

The feed does not reveal the institution's true intent or the actual initiating buyer or seller.

## Purchase Checklist

- [ ] Confirm whether usage is private, public-display, or commercial
- [ ] Recheck current provider pricing
- [ ] Confirm WebSocket access and real-time trade-condition fields
- [ ] Confirm FINRA TRF/off-exchange coverage
- [ ] Confirm historical tick retention
- [ ] Confirm redistribution and display rights
- [ ] Purchase or start a one-month trial
- [ ] Add the secret through the deployment platform
- [ ] Implement and validate the ingestion worker
- [ ] Compare live classifications against delayed FINRA ATS summaries

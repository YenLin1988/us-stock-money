import unittest

from us_stock_money.insider_trades import (
    aggregate_insider_by_ticker,
    filter_insider_trades,
    normalize_insider_trades,
    normalize_yahoo_insider_transactions,
    parse_form4_feed,
    parse_nokia_manager_release,
    parse_nokia_release_links,
    parse_ownership_document,
    summarize_insider_trades,
)


class InsiderTradesTests(unittest.TestCase):
    def test_feed_deduplicates_filing_urls(self):
        feed = b"""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry><link href="https://sec.gov/a-index.htm"/><updated>2026-06-18T20:00:00-04:00</updated></entry>
          <entry><link href="https://sec.gov/a-index.htm"/><updated>2026-06-18T20:00:00-04:00</updated></entry>
          <entry><link href="https://sec.gov/b-index.htm"/><updated>2026-06-18T19:00:00-04:00</updated></entry>
        </feed>"""

        filings = parse_form4_feed(feed)

        self.assertEqual(len(filings), 2)

    def test_parses_only_open_market_purchase_and_sale(self):
        document = b"""<?xml version="1.0"?>
        <ownershipDocument>
          <documentType>4</documentType>
          <issuer><issuerName>Example Inc.</issuerName><issuerTradingSymbol>EXM</issuerTradingSymbol></issuer>
          <reportingOwner>
            <reportingOwnerId><rptOwnerName>JANE DOE</rptOwnerName></reportingOwnerId>
            <reportingOwnerRelationship>
              <isDirector>0</isDirector><isOfficer>1</isOfficer>
              <officerTitle>Chief Financial Officer</officerTitle>
              <isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther>
            </reportingOwnerRelationship>
          </reportingOwner>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <transactionDate><value>2026-06-17</value></transactionDate>
              <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>100</value></transactionShares>
                <transactionPricePerShare><value>25.50</value></transactionPricePerShare>
              </transactionAmounts>
              <postTransactionAmounts><sharesOwnedFollowingTransaction><value>500</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
            </nonDerivativeTransaction>
            <nonDerivativeTransaction>
              <transactionDate><value>2026-06-17</value></transactionDate>
              <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
              </transactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>"""

        rows = parse_ownership_document(
            document,
            filing_url="https://sec.gov/example",
            filing_time="2026-06-18T20:00:00-04:00",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trade_side"], "Purchase")
        self.assertEqual(rows[0]["role"], "Chief Financial Officer")
        self.assertEqual(rows[0]["estimated_value"], 2550)

    def test_skips_10k_documents(self):
        document = b"""<?xml version="1.0"?>
        <ownershipDocument>
          <documentType>10-K</documentType>
          <issuer><issuerName>Example Inc.</issuerName><issuerTradingSymbol>EXM</issuerTradingSymbol></issuer>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <transactionDate><value>2026-06-17</value></transactionDate>
              <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>100</value></transactionShares>
                <transactionPricePerShare><value>25.50</value></transactionPricePerShare>
              </transactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>"""

        rows = parse_ownership_document(
            document,
            filing_url="https://sec.gov/example-10k",
            filing_time="2026-06-18T20:00:00-04:00",
        )

        self.assertEqual(rows, [])

    def test_parses_nokia_article_19_acquisition(self):
        listing = """
        <a class="td_headlines" title="Nokia Corporation - Managers' transactions (Doe)"
           href="https://www.nokia.com/newsroom/example/">Example</a>
        """
        release = """
        <html><body>
        Nokia Corporation Managers’ transactions 31 May 2026 at 13:00 EEST
        Transaction notification under Article 19 of EU Market Abuse Regulation.
        Person subject to the notification requirement
        Name: Doe, Jane
        Position: Other senior manager
        Transaction date: 2026-05-26
        Venue: XNYS
        Instrument type: SHARE
        Nature of the transaction: ACQUISITION
        Transaction details
        (1): Volume: 22713 Unit price: 16.0179 USD
        Aggregated transactions
        About Nokia
        </body></html>
        """

        links = parse_nokia_release_links(listing)
        rows = parse_nokia_manager_release(release, filing_url=links[0])

        self.assertEqual(links, ["https://www.nokia.com/newsroom/example/"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "NOK")
        self.assertEqual(rows[0]["owner_name"], "Doe, Jane")
        self.assertEqual(rows[0]["trade_side"], "Purchase")
        self.assertEqual(rows[0]["source"], "Nokia Article 19")
        self.assertAlmostEqual(rows[0]["estimated_value"], 22713 * 16.0179)

    def test_normalizes_yahoo_purchase_and_sale_but_skips_awards(self):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "Shares": 20,
                    "Value": 7760,
                    "Text": "Purchase at price 386.00 - 390.00 per share.",
                    "Insider": "TIEN BOR-ZEN",
                    "Position": "Officer",
                    "Start Date": "2026-04-28",
                    "URL": "",
                },
                {
                    "Shares": 100,
                    "Value": 25000,
                    "Text": "Sale at price 250.00 per share.",
                    "Insider": "JANE DOE",
                    "Position": "Director",
                    "Start Date": "2026-04-20",
                    "URL": "https://example.com",
                },
                {
                    "Shares": 500,
                    "Value": 0,
                    "Text": "Stock Award(Grant) at price 0.00 per share.",
                    "Insider": "JANE DOE",
                    "Position": "Director",
                    "Start Date": "2026-04-10",
                },
            ]
        )

        rows = normalize_yahoo_insider_transactions(frame, ticker="TSM")

        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows["trade_side"]), {"Purchase", "Sale"})
        self.assertEqual(rows.iloc[0]["source"], "Yahoo Finance")
        self.assertEqual(rows.iloc[0]["ticker"], "TSM")

    def test_summary_and_filters(self):
        frame = normalize_insider_trades(
            [
                {"transaction_date": "2026-06-17", "filing_time": "2026-06-18T20:00:00Z", "ticker": "EXM", "trade_side": "Purchase", "shares": 100, "price_per_share": 25, "estimated_value": 2500, "shares_after": 500},
                {"transaction_date": "2026-06-17", "filing_time": "2026-06-18T20:00:00Z", "ticker": "EXM", "trade_side": "Purchase", "shares": 100, "price_per_share": 25, "estimated_value": 2500, "shares_after": 500},
                {"transaction_date": "2026-06-17", "filing_time": "2026-06-18T19:00:00Z", "ticker": "ABC", "trade_side": "Sale", "shares": 50, "price_per_share": 40, "estimated_value": 2000, "shares_after": 100},
            ]
        )

        filtered = filter_insider_trades(frame, sides=["Purchase"], ticker="EX")
        summary = summarize_insider_trades(frame)
        aggregate = aggregate_insider_by_ticker(frame)

        self.assertEqual(len(frame), 2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(summary["purchase_shares"], 100)
        self.assertEqual(summary["sale_shares"], 50)
        self.assertEqual(summary["net_value"], 500)
        self.assertEqual(summary["tickers"], 2)
        self.assertEqual(aggregate.iloc[0]["ticker"], "EXM")
        self.assertEqual(aggregate.iloc[0]["signal"], "Net Buying")


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from us_stock_money.congress_trades import filter_congress_trades, normalize_congress_trades


class CongressTradesTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "branch": "congress",
                "filing_type": "PTR",
                "transaction_date": "2026-06-16",
                "filing_date": "2026-06-17",
                "filer_name": "Jane Doe",
                "chamber": "house",
                "party": "D",
                "state": "CA",
                "ticker": "nvda",
                "asset_name": "NVIDIA",
                "transaction_type": "Purchase",
                "amount_range_label": "$1,001 - $15,000",
                "days_to_file": 1,
                "doc_url": "https://example.com/filing.pdf",
            },
            {
                "branch": "congress",
                "filing_type": "PTR",
                "transaction_date": "2026-04-01",
                "filing_date": "2026-05-10",
                "filer_name": "John Doe",
                "chamber": "senate",
                "party": "R",
                "state": "TX",
                "ticker": "MSFT",
                "asset_name": "Microsoft",
                "transaction_type": "Sale (Full)",
                "amount_range_label": "$15,001 - $50,000",
                "days_to_file": 39,
                "doc_url": "https://example.com/filing-2.pdf",
            },
        ]

    def test_normalizes_trade_side_and_chamber(self):
        frame = normalize_congress_trades(self.records)

        self.assertEqual(frame.iloc[0]["ticker"], "NVDA")
        self.assertEqual(frame.iloc[0]["chamber"], "House")
        self.assertEqual(frame.iloc[0]["trade_side"], "Purchase")
        self.assertEqual(frame.iloc[1]["trade_side"], "Sale")

    def test_filters_by_recency_side_and_ticker(self):
        frame = normalize_congress_trades(self.records)
        filtered = filter_congress_trades(
            frame,
            days=30,
            sides=["Purchase"],
            ticker="NV",
            today=pd.Timestamp("2026-06-20"),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["filer_name"], "Jane Doe")


if __name__ == "__main__":
    unittest.main()

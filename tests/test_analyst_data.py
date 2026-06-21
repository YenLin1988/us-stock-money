import unittest

import pandas as pd

from us_stock_money.analyst_data import (
    normalize_peer_valuations,
    normalize_target_history,
    normalize_valuation,
    pe_comparison_summary,
    peer_tickers_for_theme,
    themes_for_ticker,
)


class AnalystDataTests(unittest.TestCase):
    def test_normalizes_valuation_and_price_targets(self):
        valuation = normalize_valuation(
            {
                "trailingPE": 32.5,
                "forwardPE": 20.0,
                "numberOfAnalystOpinions": 42,
                "recommendationKey": "strong_buy",
                "currency": "USD",
            },
            {"high": 300, "low": 180, "mean": 250, "median": 245},
        )

        self.assertEqual(valuation["trailing_pe"], 32.5)
        self.assertEqual(valuation["forward_pe"], 20.0)
        self.assertEqual(valuation["target_mean"], 250.0)
        self.assertEqual(valuation["analyst_count"], 42)
        self.assertEqual(valuation["recommendation"], "Strong Buy")

    def test_normalizes_institution_target_dates_and_upside(self):
        frame = pd.DataFrame(
            [
                {
                    "Firm": "Example Research",
                    "ToGrade": "Buy",
                    "Action": "main",
                    "priceTargetAction": "Raises",
                    "currentPriceTarget": 240,
                    "priorPriceTarget": 200,
                },
                {
                    "Firm": "No Target",
                    "ToGrade": "Hold",
                    "Action": "reit",
                    "priceTargetAction": "",
                    "currentPriceTarget": 0,
                    "priorPriceTarget": 0,
                },
            ],
            index=pd.to_datetime(["2026-06-01", "2026-05-01"]),
        )
        frame.index.name = "GradeDate"

        history = normalize_target_history(frame, current_price=200)
        row = history.iloc[0]

        self.assertEqual(len(history), 1)
        self.assertEqual(row["firm"], "Example Research")
        self.assertEqual(row["estimate_date"], pd.Timestamp("2026-06-01"))
        self.assertEqual(row["target_action"], "Raised")
        self.assertAlmostEqual(row["target_change_pct"], 20.0)
        self.assertAlmostEqual(row["upside_pct"], 20.0)

    def test_selects_theme_peers_and_compares_positive_pe_values(self):
        self.assertIn("AI Infrastructure", themes_for_ticker("NVDA"))
        peers = peer_tickers_for_theme("NVDA", "AI Infrastructure", limit=3)
        self.assertEqual(peers[0], "NVDA")
        self.assertEqual(len(peers), 3)

        frame = normalize_peer_valuations(
            "NVDA",
            "AI Infrastructure",
            {
                "NVDA": {"shortName": "NVIDIA", "trailingPE": 30, "forwardPE": 20},
                "AVGO": {"shortName": "Broadcom", "trailingPE": 20, "forwardPE": 16},
                "SMCI": {"shortName": "Super Micro", "trailingPE": -5, "forwardPE": 12},
            },
            ticker_order=["NVDA", "AVGO", "SMCI"],
        )
        summary = pe_comparison_summary(frame, "NVDA")

        self.assertEqual(summary["trailing_median"], 25)
        self.assertEqual(summary["forward_median"], 16)
        self.assertAlmostEqual(summary["trailing_premium_pct"], 20)
        self.assertAlmostEqual(summary["forward_premium_pct"], 25)


if __name__ == "__main__":
    unittest.main()

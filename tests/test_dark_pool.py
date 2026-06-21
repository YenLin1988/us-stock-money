import unittest

import pandas as pd

from us_stock_money.dark_pool import build_ats_anomalies, normalize_ats_activity


class DarkPoolTests(unittest.TestCase):
    def test_normalizes_finra_weekly_records(self):
        frame = normalize_ats_activity(
            [
                {
                    "issueSymbolIdentifier": "nvda",
                    "issueName": "NVIDIA",
                    "tierIdentifier": "T1",
                    "summaryStartDate": "2026-05-25",
                    "totalWeeklyTradeCount": 100,
                    "totalWeeklyShareQuantity": 200000,
                    "totalNotionalSum": 5000000,
                    "initialPublishedDate": "2026-06-15",
                }
            ]
        )

        self.assertEqual(frame.iloc[0]["ticker"], "NVDA")
        self.assertEqual(frame.iloc[0]["ats_shares"], 200000)
        self.assertEqual(frame.iloc[0]["week"], pd.Timestamp("2026-05-25"))

    def test_flags_two_times_baseline_and_z_score_anomalies(self):
        weeks = pd.date_range("2026-01-05", periods=10, freq="W-MON")
        frame = pd.DataFrame(
            {
                "ticker": ["ABC"] * 10,
                "week": weeks,
                "ats_shares": [100_000, 105_000, 95_000, 110_000, 100_000, 102_000, 98_000, 101_000, 99_000, 250_000],
                "ats_trades": [1000] * 10,
                "ats_notional": [1_000_000] * 10,
                "ats_volume_pct": [10.0] * 10,
            }
        )

        history, anomalies = build_ats_anomalies(frame)
        latest = history.iloc[-1]

        self.assertTrue(latest["is_anomaly"])
        self.assertGreater(latest["volume_multiple"], 2)
        self.assertEqual(anomalies.iloc[0]["ticker"], "ABC")

    def test_does_not_flag_small_base_noise(self):
        weeks = pd.date_range("2026-01-05", periods=8, freq="W-MON")
        frame = pd.DataFrame(
            {
                "ticker": ["TINY"] * 8,
                "week": weeks,
                "ats_shares": [1000, 1100, 900, 1000, 950, 1050, 1000, 3000],
            }
        )

        _, anomalies = build_ats_anomalies(frame, minimum_shares=100_000)

        self.assertTrue(anomalies.empty)


if __name__ == "__main__":
    unittest.main()

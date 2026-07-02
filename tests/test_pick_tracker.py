import unittest

import pandas as pd

from us_stock_money.pick_tracker import pick_hit_rate_summary


class PickTrackerTests(unittest.TestCase):
    def test_hit_rate_summary_counts_wins_and_pending(self):
        picks_df = pd.DataFrame(
            [
                {"ticker": "NVDA", "close_return_pct": 2.0, "next_close_return_pct": 4.0},
                {"ticker": "MU", "close_return_pct": -1.0, "next_close_return_pct": None},
                {"ticker": "AMD", "close_return_pct": None, "next_close_return_pct": None},
            ]
        )
        summary = pick_hit_rate_summary(picks_df)
        self.assertEqual(summary["evaluated"], 2.0)
        self.assertEqual(summary["win_rate"], 50.0)
        self.assertAlmostEqual(summary["avg_return"], 0.5)
        self.assertEqual(summary["next_day_evaluated"], 1.0)
        self.assertEqual(summary["next_day_win_rate"], 100.0)

    def test_hit_rate_summary_handles_empty(self):
        summary = pick_hit_rate_summary(pd.DataFrame())
        self.assertEqual(summary["evaluated"], 0.0)
        self.assertEqual(summary["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

import datetime
import unittest

from us_stock_money.scoring import (
    broad_flow_score,
    build_top_recommendations,
    classify_regime,
    flow_delta,
    group_scores,
    normalize,
    score_sector_flow,
    theme_group_scores,
)


class ScoringTests(unittest.TestCase):
    def test_normalize_clamps_to_range(self):
        self.assertEqual(normalize(-10, 0, 10), 0)
        self.assertEqual(normalize(20, 0, 10), 100)
        self.assertEqual(normalize(5, 0, 10), 50)

    def test_sector_flow_score_rewards_positive_metrics(self):
        weak = {
            "return_1d": -1,
            "return_5d": -3,
            "return_20d": -5,
            "relative_5d": -2,
            "dollar_volume_trend": -10,
            "volume_zscore": -1,
        }
        strong = {
            "return_1d": 1,
            "return_5d": 3,
            "return_20d": 5,
            "relative_5d": 2,
            "dollar_volume_trend": 10,
            "volume_zscore": 1,
        }
        self.assertGreater(score_sector_flow(strong), score_sector_flow(weak))

    def test_group_and_broad_scores(self):
        sector_scores = {"XLK": 80, "XLY": 70, "XLP": 40, "XLV": 50, "XLU": 30}
        groups = group_scores(sector_scores)
        self.assertGreater(groups["Risk-On"], groups["Defensive"])
        self.assertEqual(broad_flow_score({"A": 25, "B": 75}), 50)

    def test_theme_group_scores(self):
        theme_scores = {
            "Memory / HBM": 80,
            "Optical Communication": 70,
            "CPU / Advanced Packaging": 65,
            "AI Infrastructure": 75,
            "Medical / Devices": 45,
        }
        groups = theme_group_scores(theme_scores)
        self.assertGreater(groups["AI Compute Chain"], groups["Healthcare / Automation"])

    def test_regime_classification(self):
        self.assertEqual(classify_regime(60, 70, 50).name, "Risk-On Accumulation")
        self.assertEqual(classify_regime(50, 45, 65).name, "Defensive Rotation")
        self.assertEqual(classify_regime(40, 35, 45).name, "Broad Distribution")
        self.assertEqual(classify_regime(52, 53, 52).name, "Mixed Rotation")

    def test_top_recommendations_include_reasons(self):
        rows = [
            {
                "ticker": "MU",
                "themes": "Memory / HBM",
                "flow_score": 90,
                "return_5d": 4,
                "return_20d": 10,
                "relative_5d": 3,
                "dollar_volume_trend": 20,
                "volume_zscore": 1.2,
            },
            {
                "ticker": "XLV",
                "themes": "Medical / Devices",
                "flow_score": 50,
                "return_5d": 1,
                "return_20d": 2,
                "relative_5d": -1,
                "dollar_volume_trend": -5,
                "volume_zscore": 0,
            },
        ]
        recs = build_top_recommendations(rows, {"Memory / HBM": 85, "Medical / Devices": 45}, limit=1)
        self.assertEqual(recs[0]["ticker"], "MU")
        self.assertIn("flow score", recs[0]["reason"])

    def test_flow_delta_uses_last_record_before_cutoff(self):
        now = datetime.datetime(2026, 5, 27, 12, 0)
        history = [
            {"time": "2026-05-26 10:00", "broad_flow_score": 45},
            {"time": "2026-05-26 12:00", "broad_flow_score": 50},
            {"time": "2026-05-27 12:00", "broad_flow_score": 60},
        ]
        self.assertEqual(flow_delta(history, 60, 24, now), 10)


if __name__ == "__main__":
    unittest.main()

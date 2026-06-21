import datetime
import unittest

from us_stock_money.scoring import (
    broad_flow_score,
    build_breakout_candidates,
    build_integrated_recommendations,
    build_intraday_breakout_candidates,
    build_risk_watchlist,
    build_top_recommendations,
    classify_regime,
    flow_delta,
    group_scores,
    intraday_market_signal,
    market_timing_signal,
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
                "open_price": 100,
                "last_price": 105,
                "open_to_current_pct": 5,
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
                "open_price": 100,
                "last_price": 98,
                "open_to_current_pct": -2,
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
        self.assertEqual(recs[0]["open_price"], 100)
        self.assertEqual(recs[0]["last_price"], 105)
        self.assertEqual(recs[0]["open_to_current_pct"], 5)
        self.assertIn("flow score", recs[0]["reason"])

    def test_integrated_recommendations_combine_market_and_disclosure_signals(self):
        components = [
            {
                "ticker": "MU",
                "themes": "Memory / HBM",
                "open_price": 100,
                "last_price": 105,
                "open_to_current_pct": 5,
                "flow_score": 90,
                "return_5d": 8,
                "return_20d": 15,
                "relative_5d": 6,
            },
            {
                "ticker": "ABC",
                "themes": "Test",
                "open_price": 100,
                "last_price": 95,
                "open_to_current_pct": -5,
                "flow_score": 45,
                "return_5d": -4,
                "return_20d": -8,
                "relative_5d": -3,
            },
        ]
        intraday = [
            {
                "ticker": "MU",
                "themes": "Memory / HBM",
                "session_open": 100,
                "last_price": 105,
                "day_return": 5,
                "return_30m": 1.5,
                "return_60m": 3,
                "vwap": 103,
                "vwap_gap_pct": 1.9,
                "below_vwap": False,
                "volume_trend": 100,
            },
            {
                "ticker": "ABC",
                "themes": "Test",
                "session_open": 100,
                "last_price": 95,
                "day_return": -5,
                "return_30m": -1,
                "return_60m": -2,
                "vwap": 98,
                "vwap_gap_pct": -3,
                "below_vwap": True,
                "volume_trend": -20,
            },
        ]
        congress = [
            {"ticker": "MU", "trade_side": "Purchase", "amount_range_low": 15_001, "amount_range_high": 50_000},
            {"ticker": "ABC", "trade_side": "Sale", "amount_range_low": 15_001, "amount_range_high": 50_000},
        ]
        insiders = [
            {"ticker": "MU", "trade_side": "Purchase", "estimated_value": 100_000},
            {"ticker": "ABC", "trade_side": "Sale", "estimated_value": 100_000},
        ]

        recommendations = build_integrated_recommendations(
            components,
            {"Memory / HBM": 85, "Test": 40},
            intraday,
            congress,
            insiders,
            market_score=80,
            limit=2,
        )

        self.assertEqual(recommendations[0]["ticker"], "MU")
        self.assertGreater(recommendations[0]["integrated_score"], recommendations[1]["integrated_score"])
        self.assertEqual(recommendations[0]["congress_buys"], 1)
        self.assertEqual(recommendations[0]["insider_buys"], 1)
        self.assertEqual(recommendations[1]["exit_signal"], "Exit")
        self.assertIn("Congress 1B/0S", recommendations[0]["reason"])

    def test_risk_watchlist_prioritizes_exit_and_weak_factors(self):
        rows = [
            {
                "ticker": "SAFE",
                "integrated_score": 82,
                "flow_score": 85,
                "momentum_score": 80,
                "intraday_score": 88,
                "insider_score": 50,
                "congress_score": 50,
                "exit_signal": "Hold",
            },
            {
                "ticker": "RISK",
                "integrated_score": 35,
                "flow_score": 30,
                "momentum_score": 25,
                "intraday_score": 15,
                "insider_score": 20,
                "congress_score": 30,
                "exit_signal": "Exit",
            },
        ]

        risks = build_risk_watchlist(rows, limit=2)

        self.assertEqual(risks[0]["ticker"], "RISK")
        self.assertEqual(risks[0]["risk_level"], "Avoid")
        self.assertIn("5m signal is Exit", risks[0]["risk_reason"])

    def test_breakout_candidates_reward_single_stock_surge(self):
        rows = [
            {
                "ticker": "NOK",
                "themes": "Optical Communication",
                "open_price": 5.10,
                "last_price": 5.55,
                "open_to_current_pct": 8.82,
                "flow_score": 97.6,
                "return_1d": 9.5,
                "return_5d": 14.0,
                "return_20d": 18.0,
                "relative_5d": 12.0,
                "dollar_volume_trend": 101.6,
                "volume_zscore": 1.7,
            },
            {
                "ticker": "NOW",
                "themes": "AI Software / Data",
                "open_price": 1000,
                "last_price": 1004,
                "open_to_current_pct": 0.4,
                "flow_score": 92.0,
                "return_1d": 0.8,
                "return_5d": 6.0,
                "return_20d": 14.0,
                "relative_5d": 4.0,
                "dollar_volume_trend": 18.0,
                "volume_zscore": 0.3,
            },
        ]
        candidates = build_breakout_candidates(rows, limit=2)
        self.assertEqual(candidates[0]["ticker"], "NOK")
        self.assertGreater(candidates[0]["breakout_score"], candidates[1]["breakout_score"])
        self.assertIn("open-to-current", candidates[0]["reason"])

    def test_intraday_breakout_candidates_reward_5m_momentum(self):
        rows = [
            {
                "ticker": "NOK",
                "themes": "Optical Communication",
                "last_time": "2026-01-02 10:30:00-05:00",
                "session_open": 5.10,
                "last_price": 5.55,
                "day_return": 8.8,
                "return_30m": 3.4,
                "return_60m": 6.2,
                "vwap": 5.40,
                "vwap_gap_pct": 2.8,
                "below_vwap": False,
                "volume_trend": 240.0,
                "recent_dollar_volume_m": 80.0,
            },
            {
                "ticker": "NOW",
                "themes": "AI Software / Data",
                "last_time": "2026-01-02 10:30:00-05:00",
                "session_open": 1000,
                "last_price": 1002,
                "day_return": 0.2,
                "return_30m": 0.1,
                "return_60m": 0.4,
                "vwap": 1003,
                "vwap_gap_pct": -0.1,
                "below_vwap": True,
                "volume_trend": 10.0,
                "recent_dollar_volume_m": 40.0,
            },
        ]
        candidates = build_intraday_breakout_candidates(rows, limit=2)
        self.assertEqual(candidates[0]["ticker"], "NOK")
        self.assertGreater(candidates[0]["breakout_score"], 90)
        self.assertEqual(candidates[0]["exit_signal"], "Hold")
        self.assertIn("last 30m momentum", candidates[0]["reason"])

    def test_intraday_breakout_candidates_include_exit_signals(self):
        rows = [
            {
                "ticker": "EXIT",
                "themes": "Test",
                "session_open": 10,
                "last_price": 10.2,
                "day_return": 2.0,
                "return_30m": -1.0,
                "return_60m": -0.7,
                "vwap": 10.4,
                "vwap_gap_pct": -1.9,
                "below_vwap": True,
                "volume_trend": 80.0,
                "recent_dollar_volume_m": 5.0,
            },
            {
                "ticker": "TRIM",
                "themes": "Test",
                "session_open": 10,
                "last_price": 10.6,
                "day_return": 6.0,
                "return_30m": -0.2,
                "return_60m": 1.0,
                "vwap": 10.3,
                "vwap_gap_pct": 2.9,
                "below_vwap": False,
                "volume_trend": 90.0,
                "recent_dollar_volume_m": 5.0,
            },
        ]
        candidates = {row["ticker"]: row for row in build_intraday_breakout_candidates(rows, limit=2)}
        self.assertEqual(candidates["EXIT"]["exit_signal"], "Exit")
        self.assertEqual(candidates["TRIM"]["exit_signal"], "Trim")
        self.assertIn("VWAP", candidates["EXIT"]["exit_reason"])

    def test_flow_delta_uses_last_record_before_cutoff(self):
        now = datetime.datetime(2026, 5, 27, 12, 0)
        history = [
            {"time": "2026-05-26 10:00", "broad_flow_score": 45},
            {"time": "2026-05-26 12:00", "broad_flow_score": 50},
            {"time": "2026-05-27 12:00", "broad_flow_score": 60},
        ]
        self.assertEqual(flow_delta(history, 60, 24, now), 10)

    def test_market_timing_warns_when_broad_market_keeps_falling(self):
        rows = [
            {"ticker": "SPY", "return_1d": -1.2, "return_5d": -3.5, "return_20d": -6.0},
            {"ticker": "QQQ", "return_1d": -1.5, "return_5d": -4.0, "return_20d": -8.0},
            {"ticker": "IWM", "return_1d": 0.2, "return_5d": -1.0, "return_20d": -2.0},
        ]
        signal = market_timing_signal(rows, broad_score=40, risk_on_score=38)
        self.assertEqual(signal.status, "stand_aside")
        self.assertIn("暫時不要進場", signal.title)

    def test_market_timing_confirms_recovery(self):
        rows = [
            {"ticker": "SPY", "return_1d": 0.8, "return_5d": 2.1, "return_20d": -1.0},
            {"ticker": "QQQ", "return_1d": 1.0, "return_5d": 3.0, "return_20d": 2.0},
            {"ticker": "IWM", "return_1d": -0.1, "return_5d": 0.5, "return_20d": -2.0},
        ]
        signal = market_timing_signal(rows, broad_score=55, risk_on_score=58)
        self.assertEqual(signal.status, "recovery_confirmed")
        self.assertIn("可以開始評估進場", signal.title)

    def test_intraday_market_signal_warns_on_5m_selloff(self):
        rows = [
            {"ticker": "SPY", "day_return": -1.1, "return_30m": -0.4, "return_60m": -0.8, "below_vwap": True},
            {"ticker": "QQQ", "day_return": -1.4, "return_30m": -0.5, "return_60m": -1.0, "below_vwap": True},
            {"ticker": "IWM", "day_return": -0.2, "return_30m": 0.1, "return_60m": -0.2, "below_vwap": False},
        ]
        signal = intraday_market_signal(rows)
        self.assertEqual(signal.status, "intraday_stand_aside")
        self.assertIn("暫時不要進場", signal.title)

    def test_intraday_market_signal_confirms_5m_recovery(self):
        rows = [
            {"ticker": "SPY", "day_return": 0.1, "return_30m": 0.3, "return_60m": 0.2, "below_vwap": False},
            {"ticker": "QQQ", "day_return": -0.1, "return_30m": 0.4, "return_60m": 0.1, "below_vwap": False},
            {"ticker": "IWM", "day_return": -0.7, "return_30m": 0.0, "return_60m": -0.5, "below_vwap": True},
        ]
        signal = intraday_market_signal(rows)
        self.assertEqual(signal.status, "intraday_recovery")
        self.assertIn("回穩訊號", signal.title)


if __name__ == "__main__":
    unittest.main()

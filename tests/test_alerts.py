import unittest

from us_stock_money.alerts import evaluate_alerts


class AlertTests(unittest.TestCase):
    def test_broad_distribution_alert(self):
        alerts = evaluate_alerts({"broad_flow_score": 30, "regime": "Broad Distribution"})
        self.assertIn("broad-distribution", {alert.key for alert in alerts})

    def test_defensive_rotation_alert(self):
        alerts = evaluate_alerts(
            {
                "broad_flow_score": 50,
                "defensive_score": 70,
                "risk_on_score": 40,
                "regime": "Defensive Rotation",
            }
        )
        self.assertIn("defensive-rotation", {alert.key for alert in alerts})

    def test_no_alerts_for_neutral_context(self):
        self.assertEqual(evaluate_alerts({"broad_flow_score": 52, "regime": "Mixed Rotation"}), [])

    def test_market_timing_alerts(self):
        alerts = evaluate_alerts(
            {
                "broad_flow_score": 50,
                "regime": "Mixed Rotation",
                "market_timing_status": "stand_aside",
                "market_timing_title": "大盤短線連續走弱，暫時不要進場",
                "market_timing_message": "Wait for recovery.",
            }
        )
        self.assertIn("market-stand-aside", {alert.key for alert in alerts})

    def test_intraday_market_alerts(self):
        alerts = evaluate_alerts(
            {
                "broad_flow_score": 50,
                "regime": "Mixed Rotation",
                "intraday_status": "intraday_stand_aside",
                "intraday_title": "盤中 5m 急跌風險，暫時不要進場",
                "intraday_message": "Wait for intraday recovery.",
            }
        )
        self.assertIn("intraday-stand-aside", {alert.key for alert in alerts})


if __name__ == "__main__":
    unittest.main()

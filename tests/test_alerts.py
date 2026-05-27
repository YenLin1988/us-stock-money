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


if __name__ == "__main__":
    unittest.main()

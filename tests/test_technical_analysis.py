import unittest

import pandas as pd

from us_stock_money.technical_analysis import build_ma60_alerts, build_stock_detail, stock_snapshot


def make_price_data(prices: list[float], ticker: str = "TEST") -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=len(prices), freq="B")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], [ticker]],
        names=["Price", "Ticker"],
    )
    data = pd.DataFrame(index=dates, columns=columns, dtype=float)
    data[("Open", ticker)] = prices
    data[("High", ticker)] = [price + 1 for price in prices]
    data[("Low", ticker)] = [price - 1 for price in prices]
    data[("Close", ticker)] = prices
    data[("Volume", ticker)] = 1_000_000.0
    return data


class TechnicalAnalysisTests(unittest.TestCase):
    def test_recent_ma60_breakdown_is_flagged(self):
        prices = [100.0] * 64 + [99.0, 98.0]
        alerts = build_ma60_alerts(make_price_data(prices), recent_sessions=5)
        row = alerts.iloc[0]

        self.assertEqual(row["status"], "New Breakdown")
        self.assertTrue(row["recent_breakdown"])
        self.assertTrue(row["below_ma60"])
        self.assertLess(row["distance_to_ma60_pct"], 0)

    def test_stock_detail_contains_common_indicators(self):
        prices = [100 + index * 0.5 for index in range(90)]
        detail = build_stock_detail(make_price_data(prices), "TEST")
        snapshot = stock_snapshot(detail)

        self.assertIn("ma20", detail)
        self.assertIn("ma60", detail)
        self.assertIn("rsi14", detail)
        self.assertIn("macd", detail)
        self.assertAlmostEqual(snapshot["last_price"], prices[-1])
        self.assertGreater(snapshot["ma60_gap"], 0)
        self.assertEqual(snapshot["high_52w"], prices[-1])


if __name__ == "__main__":
    unittest.main()

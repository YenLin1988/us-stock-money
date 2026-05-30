import unittest

import pandas as pd

from us_stock_money.market_data import build_component_table


class MarketDataTests(unittest.TestCase):
    def test_component_table_includes_open_to_current_change(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "Close", "Volume"], ["MU", "SPY"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(index=dates, columns=columns, dtype=float)
        data[("Open", "MU")] = 100.0
        data[("Close", "MU")] = [100.0 + index for index in range(30)]
        data.loc[dates[-1], ("Open", "MU")] = 120.0
        data.loc[dates[-1], ("Close", "MU")] = 126.0
        data[("Volume", "MU")] = 1_000_000.0
        data[("Open", "SPY")] = 400.0
        data[("Close", "SPY")] = 400.0
        data[("Volume", "SPY")] = 10_000_000.0

        rows = build_component_table(data)
        row = rows[rows["ticker"] == "MU"].iloc[0]

        self.assertEqual(row["open_price"], 120.0)
        self.assertEqual(row["last_price"], 126.0)
        self.assertAlmostEqual(row["open_to_current_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()

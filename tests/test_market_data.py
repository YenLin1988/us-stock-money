import unittest

import pandas as pd

from us_stock_money.market_data import build_component_table, build_intraday_component_table, build_intraday_price_table


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

    def test_intraday_component_table_includes_5m_breakout_fields(self):
        periods = pd.date_range("2026-01-02 09:30", periods=18, freq="5min")
        columns = pd.MultiIndex.from_product(
            [["Close", "Volume"], ["MU"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(index=periods, columns=columns, dtype=float)
        data[("Close", "MU")] = [100.0 + index for index in range(18)]
        data[("Volume", "MU")] = [1_000_000.0] * 12 + [3_000_000.0] * 6

        rows = build_intraday_component_table(data)
        row = rows[rows["ticker"] == "MU"].iloc[0]

        self.assertEqual(row["session_open"], 100.0)
        self.assertEqual(row["last_price"], 117.0)
        self.assertGreater(row["day_return"], 0)
        self.assertGreater(row["return_30m"], 0)
        self.assertGreater(row["vwap_gap_pct"], 0)
        self.assertGreater(row["volume_trend"], 0)

    def test_intraday_price_table_uses_latest_session_open_and_current_price(self):
        times = pd.to_datetime(
            [
                "2026-01-05 09:30",
                "2026-01-05 09:35",
                "2026-01-06 09:30",
                "2026-01-06 09:35",
                "2026-01-06 09:40",
            ]
        )
        columns = pd.MultiIndex.from_product(
            [["Open", "Close", "Volume"], ["MU"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(index=times, columns=columns, dtype=float)
        data[("Open", "MU")] = [90.0, 91.0, 100.0, 103.0, 104.0]
        data[("Close", "MU")] = [91.0, 92.0, 101.0, 104.0, 105.0]
        data[("Volume", "MU")] = 1_000_000.0

        rows = build_intraday_price_table(data, ["MU"])
        row = rows.iloc[0]

        self.assertEqual(row["ticker"], "MU")
        self.assertEqual(row["open_price"], 100.0)
        self.assertEqual(row["last_price"], 105.0)
        self.assertAlmostEqual(row["open_to_current_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()

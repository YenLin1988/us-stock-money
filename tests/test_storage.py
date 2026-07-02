import tempfile
import unittest
from pathlib import Path

from us_stock_money.storage import HistoryStore


class StorageTests(unittest.TestCase):
    def test_upsert_and_load_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HistoryStore(Path(tmpdir) / "flow.sqlite3")
            store.upsert_record({"time": "2026-05-27 12:00", "broad_flow_score": 55})
            store.upsert_record({"time": "2026-05-27 13:00", "broad_flow_score": 60})
            records = store.load_history()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[-1]["broad_flow_score"], 60)

    def test_record_requires_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HistoryStore(Path(tmpdir) / "flow.sqlite3")
            with self.assertRaises(ValueError):
                store.upsert_record({"broad_flow_score": 55})

    def test_intraday_picks_keep_first_snapshot_per_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HistoryStore(Path(tmpdir) / "flow.sqlite3")
            first = [
                {"pick_date": "2026-06-01", "ticker": "NVDA", "pick_price": 100.0, "breakout_score": 80.0, "themes": "AI Infrastructure"},
            ]
            second = [
                {"pick_date": "2026-06-01", "ticker": "NVDA", "pick_price": 105.0, "breakout_score": 90.0, "themes": "AI Infrastructure"},
                {"pick_date": "2026-06-01", "ticker": "MU", "pick_price": 50.0, "breakout_score": 70.0, "themes": "Memory / HBM"},
            ]
            self.assertEqual(store.save_intraday_picks(first, picked_at="2026-06-01 22:35"), 1)
            self.assertEqual(store.save_intraday_picks(second, picked_at="2026-06-01 23:05"), 1)

            picks = {pick["ticker"]: pick for pick in store.load_intraday_picks()}
            self.assertEqual(len(picks), 2)
            self.assertEqual(picks["NVDA"]["pick_price"], 100.0)  # first snapshot wins
            self.assertEqual(picks["NVDA"]["picked_at"], "2026-06-01 22:35")

    def test_intraday_pick_outcomes_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HistoryStore(Path(tmpdir) / "flow.sqlite3")
            store.save_intraday_picks(
                [{"pick_date": "2026-06-01", "ticker": "NVDA", "pick_price": 100.0, "breakout_score": 80.0}],
                picked_at="2026-06-01 22:35",
            )
            updated = store.update_pick_outcomes(
                [
                    {
                        "pick_date": "2026-06-01",
                        "ticker": "NVDA",
                        "close_price": 103.0,
                        "close_return_pct": 3.0,
                        "next_close_price": 101.0,
                        "next_close_return_pct": 1.0,
                    }
                ]
            )
            self.assertEqual(updated, 1)
            pick = store.load_intraday_picks()[0]
            self.assertEqual(pick["close_return_pct"], 3.0)
            self.assertEqual(pick["next_close_return_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()

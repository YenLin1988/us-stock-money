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


if __name__ == "__main__":
    unittest.main()

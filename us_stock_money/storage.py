"""SQLite-backed history storage for money-flow observations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flow_history (
                    time TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert_record(self, record: dict[str, Any]) -> None:
        if "time" not in record:
            raise ValueError("history record requires a time field")
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO flow_history (time, payload) VALUES (?, ?)",
                (str(record["time"]), payload),
            )
            conn.commit()

    def load_history(self, limit: int = 2000) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT payload
                FROM flow_history
                ORDER BY time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

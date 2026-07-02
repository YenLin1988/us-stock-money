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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS intraday_picks (
                    pick_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    picked_at TEXT NOT NULL,
                    pick_price REAL NOT NULL,
                    breakout_score REAL NOT NULL,
                    themes TEXT NOT NULL DEFAULT '',
                    close_price REAL,
                    close_return_pct REAL,
                    next_close_price REAL,
                    next_close_return_pct REAL,
                    PRIMARY KEY (pick_date, ticker)
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

    def save_intraday_picks(self, picks: list[dict[str, Any]], picked_at: str) -> int:
        """Archive intraday picks, keeping only the first snapshot per day per ticker.

        The first logged pick is the actual decision point, so later refreshes
        in the same session must not overwrite it (INSERT OR IGNORE).
        """
        saved = 0
        with closing(self._connect()) as conn:
            for pick in picks:
                pick_date = str(pick.get("pick_date", ""))
                ticker = str(pick.get("ticker", ""))
                if not pick_date or not ticker:
                    continue
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO intraday_picks
                        (pick_date, ticker, picked_at, pick_price, breakout_score, themes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pick_date,
                        ticker,
                        picked_at,
                        float(pick.get("pick_price", 0.0)),
                        float(pick.get("breakout_score", 0.0)),
                        str(pick.get("themes", "")),
                    ),
                )
                saved += cursor.rowcount if cursor.rowcount > 0 else 0
            conn.commit()
        return saved

    def load_intraday_picks(self, limit: int = 500) -> list[dict[str, Any]]:
        columns = [
            "pick_date",
            "ticker",
            "picked_at",
            "pick_price",
            "breakout_score",
            "themes",
            "close_price",
            "close_return_pct",
            "next_close_price",
            "next_close_return_pct",
        ]
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT {', '.join(columns)}
                FROM intraday_picks
                ORDER BY pick_date DESC, breakout_score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def update_pick_outcomes(self, outcomes: list[dict[str, Any]]) -> int:
        updated = 0
        with closing(self._connect()) as conn:
            for outcome in outcomes:
                cursor = conn.execute(
                    """
                    UPDATE intraday_picks
                    SET close_price = ?,
                        close_return_pct = ?,
                        next_close_price = ?,
                        next_close_return_pct = ?
                    WHERE pick_date = ? AND ticker = ?
                    """,
                    (
                        outcome.get("close_price"),
                        outcome.get("close_return_pct"),
                        outcome.get("next_close_price"),
                        outcome.get("next_close_return_pct"),
                        str(outcome.get("pick_date", "")),
                        str(outcome.get("ticker", "")),
                    ),
                )
                updated += cursor.rowcount if cursor.rowcount > 0 else 0
            conn.commit()
        return updated

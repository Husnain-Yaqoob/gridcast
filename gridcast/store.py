"""SQLite storage for grid readings.

SQLite rather than a server database, on purpose. This pipeline runs on a
laptop, collects a few thousand rows a day, and needs to survive being cloned
by someone who wants to look at it. A dependency on a running SQL Server
instance would make "does it work on your machine" a real question. If the
project outgrows this, the store interface is the only thing that changes.

The single most important property here is idempotency: running the ingest
twice must produce the same database, not two copies of every reading.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, Sequence

from .models import Reading

SCHEMA = """
CREATE TABLE IF NOT EXISTS reading (
    area        TEXT    NOT NULL,
    region      TEXT    NOT NULL,
    ts_utc      TEXT    NOT NULL,   -- ISO-8601, always UTC, always sortable
    value       REAL,               -- NULL is meaningful: not collected
    fetched_at  TEXT    NOT NULL,

    PRIMARY KEY (area, region, ts_utc)
);

CREATE INDEX IF NOT EXISTS ix_reading_area_ts ON reading (area, region, ts_utc);

-- Every run leaves a record of itself. An unattended pipeline that silently
-- stops is indistinguishable from one that is working, right up until someone
-- notices the dashboard has not moved in three weeks.
CREATE TABLE IF NOT EXISTS ingest_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    command       TEXT NOT NULL,
    rows_seen     INTEGER,
    rows_written  INTEGER,
    nulls_seen    INTEGER,
    status        TEXT,
    detail        TEXT
);
"""


@dataclass
class IngestResult:
    rows_seen: int = 0
    rows_written: int = 0
    nulls_seen: int = 0

    @property
    def summary(self) -> str:
        return (
            f"{self.rows_written:,} rows written from {self.rows_seen:,} seen"
            f" ({self.nulls_seen:,} null)"
        )


def _iso(moment: datetime) -> str:
    """UTC, to the second, in a form that sorts lexicographically."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            # Write-ahead logging so a long backfill does not block a reader,
            # which matters once a scheduled poll and a notebook are both open.
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    # ------------------------------------------------------------------ writes
    def upsert(self, readings: Iterable[Reading]) -> IngestResult:
        """Insert or update readings. Safe to run on overlapping ranges.

        A later fetch of the same period wins, which is the behaviour we want:
        EirGrid revises recent figures as they settle, so the newest answer for
        a given timestamp is the most correct one. The composite primary key
        makes a duplicate physically impossible rather than merely unlikely.
        """
        batch: list[tuple] = []
        result = IngestResult()
        now = _iso(datetime.now(timezone.utc))

        for reading in readings:
            result.rows_seen += 1
            if reading.is_missing:
                result.nulls_seen += 1
            batch.append((
                reading.area,
                reading.region,
                _iso(reading.timestamp_utc),
                reading.value,
                now,
            ))

        if not batch:
            return result

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO reading (area, region, ts_utc, value, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(area, region, ts_utc) DO UPDATE SET
                    value = excluded.value,
                    fetched_at = excluded.fetched_at
                """,
                batch,
            )
        result.rows_written = len(batch)
        return result

    def start_run(self, command: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO ingest_log (started_at, command) VALUES (?, ?)",
                (_iso(datetime.now(timezone.utc)), command),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, result: IngestResult,
                   status: str, detail: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_log
                   SET finished_at = ?, rows_seen = ?, rows_written = ?,
                       nulls_seen = ?, status = ?, detail = ?
                 WHERE id = ?
                """,
                (_iso(datetime.now(timezone.utc)), result.rows_seen,
                 result.rows_written, result.nulls_seen, status,
                 detail or None, run_id),
            )

    # ------------------------------------------------------------------- reads
    def latest_timestamp(self, area: str, region: str) -> datetime | None:
        """Newest reading that actually has a value.

        Deliberately ignores null rows. EirGrid publishes placeholder rows for
        periods that have not settled, so asking for the newest row of any kind
        would report the pipeline as up to date while holding nothing but gaps.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(ts_utc) AS newest
                  FROM reading
                 WHERE area = ? AND region = ? AND value IS NOT NULL
                """,
                (area, region),
            ).fetchone()

        if not row or not row["newest"]:
            return None
        return datetime.strptime(row["newest"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )

    def coverage(self) -> Sequence[sqlite3.Row]:
        """One row per series: how much is held, how much of it is real."""
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT area, region,
                       COUNT(*)                                   AS rows_held,
                       SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS nulls,
                       MIN(ts_utc)                                AS first_ts,
                       MAX(ts_utc)                                AS last_ts
                  FROM reading
                 GROUP BY area, region
                 ORDER BY area, region
                """
            ).fetchall()

    def recent_runs(self, limit: int = 10) -> Sequence[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT id, started_at, finished_at, command,
                       rows_seen, rows_written, nulls_seen, status, detail
                  FROM ingest_log
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM reading"
            ).fetchone()[0])

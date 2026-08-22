"""Storage tests, centred on idempotency.

A scheduled pipeline gets run twice. It gets retried after a timeout. It gets
kicked off manually by someone who forgot cron already did it. If any of those
doubles the data, every figure downstream is wrong and nothing announces it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gridcast.models import Reading
from gridcast.store import Store


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def reading(hour=9, value=100.0, area="wind", region="ALL"):
    return Reading(
        area=area,
        region=region,
        timestamp_utc=datetime(2026, 1, 15, hour, 0, tzinfo=timezone.utc),
        value=value,
    )


def test_upsert_writes_rows(store):
    result = store.upsert([reading(9), reading(10)])
    assert result.rows_written == 2
    assert store.count() == 2


def test_running_twice_does_not_duplicate(store):
    """The property the whole pipeline depends on."""
    batch = [reading(9), reading(10), reading(11)]
    store.upsert(batch)
    store.upsert(batch)
    assert store.count() == 3


def test_later_fetch_overwrites_earlier_value(store):
    """EirGrid revises provisional figures; the newest answer should win."""
    store.upsert([reading(9, value=100.0)])
    store.upsert([reading(9, value=142.5)])

    assert store.count() == 1
    with store._connect() as connection:
        row = connection.execute("SELECT value FROM reading").fetchone()
    assert row["value"] == pytest.approx(142.5)


def test_null_survives_into_the_database(store):
    store.upsert([reading(9, value=None)])
    with store._connect() as connection:
        row = connection.execute("SELECT value FROM reading").fetchone()
    assert row["value"] is None


def test_null_counted_separately(store):
    result = store.upsert([reading(9, value=None), reading(10, value=50.0)])
    assert result.nulls_seen == 1
    assert result.rows_seen == 2


def test_series_are_stored_independently(store):
    """Same timestamp, different series — both must be kept."""
    store.upsert([reading(9, area="wind"), reading(9, area="demand")])
    assert store.count() == 2


def test_regions_are_stored_independently(store):
    store.upsert([reading(9, region="ROI"), reading(9, region="NI")])
    assert store.count() == 2


# ------------------------------------------------------------------ watermarks
def test_latest_timestamp_ignores_nulls(store):
    """A trailing run of placeholder rows must not read as 'up to date'.

    If it did, the incremental update would start after the nulls and never go
    back for the real values once they settle — leaving a permanent hole.
    """
    store.upsert([
        reading(9, value=100.0),
        reading(10, value=None),
        reading(11, value=None),
    ])
    newest = store.latest_timestamp("wind", "ALL")
    assert newest == datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)


def test_latest_timestamp_is_none_when_empty(store):
    assert store.latest_timestamp("wind", "ALL") is None


def test_latest_timestamp_is_per_series(store):
    store.upsert([
        reading(9, area="wind", value=1.0),
        reading(15, area="demand", value=2.0),
    ])
    assert store.latest_timestamp("wind", "ALL").hour == 9
    assert store.latest_timestamp("demand", "ALL").hour == 15


# --------------------------------------------------------------------- logging
def test_run_is_logged_start_to_finish(store):
    run_id = store.start_run("update ALL")
    result = store.upsert([reading(9)])
    store.finish_run(run_id, result, "success")

    runs = store.recent_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "success"
    assert runs[0]["finished_at"] is not None
    assert runs[0]["rows_written"] == 1


def test_coverage_reports_nulls(store):
    store.upsert([reading(9, value=1.0), reading(10, value=None)])
    rows = store.coverage()
    assert len(rows) == 1
    assert rows[0]["rows_held"] == 2
    assert rows[0]["nulls"] == 1

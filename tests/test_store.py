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


def test_coverage_separates_held_rows_from_real_data(store):
    """`last_ts` counts placeholders; `real_last_ts` must not.

    This is the bug that made a stalled series look healthier than a working
    one. EirGrid keeps publishing empty rows ahead of settlement, so the series
    that stopped receiving values grows the longest tail of nulls and reports
    the newest `last_ts` of anything in the database.
    """
    store.upsert([
        reading(9, value=100.0),
        reading(10, value=None),
        reading(11, value=None),
    ])
    row = store.coverage()[0]
    assert row["last_ts"].startswith("2026-01-15T11:00")
    assert row["real_last_ts"].startswith("2026-01-15T09:00")


def test_a_stalled_series_does_not_outrank_a_healthy_one(store):
    """The inversion itself, written down as a test.

    Observed in production: wind had received nothing for fourteen hours and
    displayed as four hours fresher than SNSP, which was current. Ordering by
    `last_ts` reproduces that. Ordering by `real_last_ts` does not.
    """
    store.upsert([
        # Healthy: current to 12:00, no placeholder tail.
        reading(12, value=500.0, area="snsp"),
        # Stalled: real data stopped at 09:00, placeholders run on to 18:00.
        reading(9, value=100.0, area="wind"),
        reading(18, value=None, area="wind"),
    ])
    by_area = {r["area"]: r for r in store.coverage()}

    assert by_area["wind"]["last_ts"] > by_area["snsp"]["last_ts"]
    assert by_area["wind"]["real_last_ts"] < by_area["snsp"]["real_last_ts"]


def test_real_last_ts_is_null_when_a_series_holds_only_placeholders(store):
    store.upsert([reading(9, value=None), reading(10, value=None)])
    assert store.coverage()[0]["real_last_ts"] is None

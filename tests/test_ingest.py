"""Ingestion tests.

These exist because two real backfills failed in two different ways.

The first fetched all chunks for a series and wrote them only once the last
succeeded, so an interrupt partway through discarded everything.

The second was throttled by EirGrid after about fifteen requests. Without a
record of which windows had already been retrieved, every re-run would have
re-requested the same early months and never reached the later ones.

The tests below pin down both fixes.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from gridcast import ingest
from gridcast.client import GridClientError
from gridcast.models import Reading
from gridcast.store import Store


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


class FakeClient:
    """Mimics GridClient's two-part interface: pure `windows`, fetching `fetch`."""

    def __init__(self, windows=3, fail_on=None, interrupt_after=None,
                 fail_after=None):
        self.window_count = windows
        self.fail_on = fail_on or set()
        self.interrupt_after = interrupt_after
        self.fail_after = fail_after
        self.fetches = []            # (area, window_start) actually requested

    def windows(self, start, end):
        for offset in range(self.window_count):
            day = start + timedelta(days=offset)
            yield day, day

    def fetch(self, area_label, region, window_start, window_end):
        if area_label in self.fail_on:
            raise GridClientError(f"{area_label} is unavailable")
        if self.interrupt_after is not None and len(self.fetches) >= self.interrupt_after:
            raise KeyboardInterrupt()
        if self.fail_after is not None and len(self.fetches) >= self.fail_after:
            raise GridClientError("HTTP 403: still throttled")

        self.fetches.append((area_label, window_start))
        return [
            Reading(
                area=area_label,
                region=region,
                timestamp_utc=datetime.combine(
                    window_start, datetime.min.time(), tzinfo=timezone.utc
                ) + timedelta(hours=hour),
                value=float(hour),
            )
            for hour in range(4)
        ]


# A range safely in the past, so the "window ends today" rule does not apply.
START, END = date(2026, 1, 1), date(2026, 1, 3)


def test_backfill_stores_every_series(store):
    result = ingest.backfill(store, FakeClient(), START, END,
                             areas=("wind", "demand"))
    assert result.rows_written == 2 * 3 * 4
    assert store.count() == 2 * 3 * 4


def test_each_window_is_committed_as_it_arrives(store):
    """The property that was missing, and that cost a full backfill."""
    client = FakeClient(windows=5, interrupt_after=3)

    with pytest.raises(ingest.Cancelled):
        ingest.backfill(store, client, START, END, areas=("wind",))

    assert store.count() == 3 * 4


def test_interrupt_is_recorded_as_cancelled_not_left_open(store):
    """A run log entry with no finish time reads as 'still running' forever."""
    with pytest.raises(ingest.Cancelled):
        ingest.backfill(store, FakeClient(interrupt_after=1), START, END,
                        areas=("wind",))

    runs = store.recent_runs()
    assert runs[0]["status"] == "cancelled"
    assert runs[0]["finished_at"] is not None
    assert runs[0]["rows_written"] == 4


def test_one_failing_series_does_not_abandon_the_others(store):
    result = ingest.backfill(store, FakeClient(fail_on={"wind"}), START, END,
                             areas=("wind", "demand", "snsp"))

    assert result.rows_written == 2 * 3 * 4
    runs = store.recent_runs()
    assert runs[0]["status"] == "partial"
    assert "wind" in runs[0]["detail"]


def test_partial_series_keeps_what_it_managed_before_failing(store):
    """Throttled at window three: the first two must survive."""
    ingest.backfill(store, FakeClient(windows=5, fail_after=2), START, END,
                    areas=("wind",))
    assert store.count() == 2 * 4
    assert store.recent_runs()[0]["status"] == "partial"


# ------------------------------------------------------------------- resuming
def test_rerun_skips_windows_already_fetched(store):
    """The fix for being throttled: never ask twice for the same window."""
    first = FakeClient()
    ingest.backfill(store, first, START, END, areas=("wind",))
    assert len(first.fetches) == 3

    second = FakeClient()
    ingest.backfill(store, second, START, END, areas=("wind",))
    assert second.fetches == [], "re-run should have requested nothing"


def test_rerun_after_throttling_continues_where_it_stopped(store):
    """The scenario that motivated the ledger."""
    throttled = FakeClient(windows=5, fail_after=2)
    ingest.backfill(store, throttled, START, END, areas=("wind",))
    assert len(throttled.fetches) == 2

    resumed = FakeClient(windows=5)
    ingest.backfill(store, resumed, START, END, areas=("wind",))

    # Only the three it never got to.
    assert len(resumed.fetches) == 3
    assert store.count() == 5 * 4


def test_rerunning_backfill_does_not_duplicate(store):
    ingest.backfill(store, FakeClient(), START, END, areas=("wind",))
    first = store.count()
    ingest.backfill(store, FakeClient(), START, END, areas=("wind",))
    assert store.count() == first


def test_window_ending_today_is_not_marked_complete(store):
    """Today is still being published; recording it as done would freeze it."""
    today = datetime.now(timezone.utc).date()
    client = FakeClient(windows=1)
    ingest.backfill(store, client, today, today, areas=("wind",))

    assert store.windows_fetched("wind", "ALL") == 0

    again = FakeClient(windows=1)
    ingest.backfill(store, again, today, today, areas=("wind",))
    assert len(again.fetches) == 1, "today should be re-fetched, not skipped"


def test_ledger_is_per_series(store):
    ingest.backfill(store, FakeClient(), START, END, areas=("wind",))
    assert store.windows_fetched("wind", "ALL") == 3
    assert store.windows_fetched("demand", "ALL") == 0


def test_update_skips_a_series_that_is_already_current(store, monkeypatch):
    """window_for can return a start after the end; that is not an error."""
    monkeypatch.setattr(
        store, "latest_timestamp",
        lambda area, region: datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    result = ingest.update(store, FakeClient(), areas=("wind",))
    assert result.rows_written == 0
    assert store.recent_runs()[0]["status"] == "success"

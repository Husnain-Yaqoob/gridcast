"""Orchestration: decide what to fetch, fetch it, store it, record the run."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from .client import GridClient, GridClientError
from .config import AREAS, DEFAULT_REGION
from .store import IngestResult, Store

log = logging.getLogger(__name__)

# How far back to re-fetch on an incremental run.
#
# EirGrid revises recent figures as metering settles, so the last few hours of
# any earlier fetch are provisional. Re-reading a day's overlap costs one small
# request per series and means the database converges on the settled values
# instead of preserving whatever happened to be published first.
OVERLAP_DAYS = 1


def backfill(store: Store, client: GridClient, start: date, end: date,
             region: str = DEFAULT_REGION,
             areas: tuple[str, ...] | None = None) -> IngestResult:
    """Load a fixed historical range for every series."""
    labels = areas or tuple(a.label for a in AREAS)
    run_id = store.start_run(f"backfill {start}..{end} {region}")
    total = IngestResult()
    failures: list[str] = []

    try:
        for label in labels:
            try:
                readings = list(client.fetch_range(label, region, start, end))
            except GridClientError as exc:
                # One unavailable series should not abandon the other five.
                log.error("%s failed: %s", label, exc)
                failures.append(f"{label}: {exc}")
                continue

            result = store.upsert(readings)
            total.rows_seen += result.rows_seen
            total.rows_written += result.rows_written
            total.nulls_seen += result.nulls_seen
            log.info("%s: %s", label, result.summary)

        status = "partial" if failures else "success"
        store.finish_run(run_id, total, status, "; ".join(failures)[:900])
        return total

    except Exception as exc:
        store.finish_run(run_id, total, "failed", str(exc)[:900])
        raise


def update(store: Store, client: GridClient, region: str = DEFAULT_REGION,
           areas: tuple[str, ...] | None = None,
           default_lookback_days: int = 7) -> IngestResult:
    """Fetch whatever has appeared since the last run, per series.

    Each series carries its own high-water mark. They are not always in step —
    carbon intensity has been seen to lag wind by a settlement period — and a
    single global watermark would either re-fetch everything or silently skip
    the series that lags.
    """
    labels = areas or tuple(a.label for a in AREAS)
    run_id = store.start_run(f"update {region}")
    total = IngestResult()
    failures: list[str] = []
    today = datetime.now(timezone.utc).date()

    try:
        for label in labels:
            newest = store.latest_timestamp(label, region)
            if newest is None:
                start = today - timedelta(days=default_lookback_days)
                log.info("%s: no data held, starting %s", label, start)
            else:
                start = (newest - timedelta(days=OVERLAP_DAYS)).date()
                log.info("%s: newest %s, re-reading from %s",
                         label, newest.isoformat(), start)

            try:
                readings = list(client.fetch_range(label, region, start, today))
            except GridClientError as exc:
                log.error("%s failed: %s", label, exc)
                failures.append(f"{label}: {exc}")
                continue

            result = store.upsert(readings)
            total.rows_seen += result.rows_seen
            total.rows_written += result.rows_written
            total.nulls_seen += result.nulls_seen
            log.info("%s: %s", label, result.summary)

        status = "partial" if failures else "success"
        store.finish_run(run_id, total, status, "; ".join(failures)[:900])
        return total

    except Exception as exc:
        store.finish_run(run_id, total, "failed", str(exc)[:900])
        raise

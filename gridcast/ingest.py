"""Orchestration: decide what to fetch, fetch it, store it, record the run.

The governing principle here is that work already done must not be thrown
away. A backfill makes over three hundred requests across twenty minutes.
Anything that discards all of that because request 280 failed — or because
someone pressed Ctrl+C — is badly built, however tidy the code looks.

So every chunk is committed as it arrives, a failed series does not abandon
the others, and an interrupt records what was achieved before exiting.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from .client import GridClient, GridClientError, ThrottledOut
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


class Cancelled(Exception):
    """The operator interrupted the run. Not an error, but not a success."""


def _accumulate(total: IngestResult, part: IngestResult) -> None:
    total.rows_seen += part.rows_seen
    total.rows_written += part.rows_written
    total.nulls_seen += part.nulls_seen


def _run_series(store: Store, client: GridClient, label: str, region: str,
                start: date, end: date, total: IngestResult,
                resume: bool = True) -> None:
    """Fetch one series across a range, committing each window as it lands.

    Two properties make this survivable against a service that rate-limits.

    Committing per window rather than per series means an interruption keeps
    the work already done. The first version accumulated a year in memory and
    wrote only after every request succeeded, so being throttled at request 16
    discarded the previous 15.

    Skipping windows already in the fetch ledger means a re-run resumes rather
    than restarting. Without it, a throttled backfill re-requests the same
    early months on every attempt and never reaches the later ones, however
    many times it is run.
    """
    today = datetime.now(timezone.utc).date()

    for window_start, window_end in client.windows(start, end):
        if resume and store.window_fetched(label, region, window_start, window_end):
            log.info("  %s %s..%s: already held, skipping",
                     label, window_start, window_end)
            continue

        readings = client.fetch(label, region, window_start, window_end)
        result = store.upsert(readings)
        _accumulate(total, result)
        store.mark_window_fetched(label, region, window_start, window_end,
                                  result.rows_seen)

        # A window that runs up to today is not finished — the rest of today
        # has not been published yet. Recording it as done would freeze the
        # series at this moment and no later run would ever revisit it.
        if window_end >= today:
            store.forget_window(label, region, window_start, window_end)

        log.info("  %s %s..%s: %s", label, window_start, window_end, result.summary)


def _run(store: Store, client: GridClient, command: str, labels: tuple[str, ...],
         region: str, window_for, resume: bool = True) -> IngestResult:
    """Shared body of backfill and update.

    `window_for(label)` returns the (start, end) this series needs, which is
    the only thing that differs between the two commands.
    """
    run_id = store.start_run(command)
    total = IngestResult()
    failures: list[str] = []

    try:
        for label in labels:
            start, end = window_for(label)
            if start > end:
                log.info("%s: already up to date", label)
                continue
            try:
                _run_series(store, client, label, region, start, end, total,
                            resume=resume)
            except ThrottledOut as exc:
                # The limit is on us, not on this series. Trying the remaining
                # five would just collect five more refusals and waste another
                # ten minutes of cool-downs.
                log.error("stopping: %s", exc)
                failures.append(str(exc))
                break
            except GridClientError as exc:
                # One unavailable series must not abandon the other five, and
                # whatever it managed to store before failing is already
                # committed.
                log.error("%s failed: %s", label, exc)
                failures.append(f"{label}: {exc}")

        status = "partial" if failures else "success"
        store.finish_run(run_id, total, status, "; ".join(failures)[:900])
        return total

    except KeyboardInterrupt:
        # KeyboardInterrupt is a BaseException, so a bare `except Exception`
        # never sees it — the process dies where it stands and the run log is
        # left open forever, reading as though the job never finished.
        #
        # Everything fetched so far is already committed. All that is needed
        # is an honest note in the log saying the run was stopped.
        log.warning("interrupted — %s", total.summary)
        store.finish_run(run_id, total, "cancelled",
                         "interrupted by operator; data fetched so far is committed")
        raise Cancelled() from None

    except Exception as exc:
        store.finish_run(run_id, total, "failed", str(exc)[:900])
        raise


def backfill(store: Store, client: GridClient, start: date, end: date,
             region: str = DEFAULT_REGION,
             areas: tuple[str, ...] | None = None,
             resume: bool = True) -> IngestResult:
    """Load a fixed historical range for every series.

    `resume=True` skips windows already in the fetch ledger, which is what
    makes a throttled backfill finishable across several attempts. Pass False
    to re-request everything — useful only if the stored data is suspect.
    """
    labels = areas or tuple(a.label for a in AREAS)
    return _run(
        store, client, f"backfill {start}..{end} {region}", labels, region,
        window_for=lambda label: (start, end),
        resume=resume,
    )


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
    today = datetime.now(timezone.utc).date()

    def window_for(label: str) -> tuple[date, date]:
        newest = store.latest_timestamp(label, region)
        if newest is None:
            start = today - timedelta(days=default_lookback_days)
            log.info("%s: no data held, starting %s", label, start)
        else:
            start = (newest - timedelta(days=OVERLAP_DAYS)).date()
            log.info("%s: newest %s, re-reading from %s",
                     label, newest.isoformat(), start)
        return start, today

    return _run(store, client, f"update {region}", labels, region, window_for)

"""HTTP access to the EirGrid Smart Grid Dashboard.

Everything that knows about the network lives here. The rest of the package
receives Readings and never learns where they came from, which is what makes
the store and the feature code testable without touching the internet.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Iterator

import requests

from .config import (
    API_DATE_FORMAT,
    AREAS_BY_LABEL,
    BACKOFF_BASE_SECONDS,
    BASE_URL,
    MAX_DAYS_PER_REQUEST,
    MAX_RETRIES,
    COOLDOWN_TICK_SECONDS,
    MAX_THROTTLE_WAITS,
    REQUEST_INTERVAL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_STATUS_CODES,
    THROTTLE_COOLDOWN_SECONDS,
    THROTTLE_STATUS_CODES,
    USER_AGENT,
)
from .models import ParseError, Reading, parse_response

log = logging.getLogger(__name__)


class GridClientError(Exception):
    """The dashboard could not be reached, or refused to answer usefully."""


class ThrottledOut(GridClientError):
    """The run's cool-down allowance is spent.

    Distinct from a one-off failure: there is no point trying the next series,
    because the limit applies to the client, not to the endpoint. The ingest
    layer stops the whole run when it sees this rather than grinding through
    five more series that will all be refused.
    """


class GridClient:
    """A polite client for the dashboard service.

    Deliberately synchronous. Concurrency here would buy nothing — the data
    updates every fifteen minutes — while making it far easier to breach the
    licence's fair-use provision by accident.
    """

    def __init__(self, session: requests.Session | None = None,
                 interval_seconds: float = REQUEST_INTERVAL_SECONDS,
                 sleep=time.sleep) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        self.interval_seconds = interval_seconds
        self._sleep = sleep          # injected so tests do not actually wait
        self._last_request_at: float | None = None

        # Cool-downs are budgeted across the client's whole life, not per
        # request. See MAX_THROTTLE_WAITS.
        self._throttle_waits = 0

    @property
    def throttle_budget_spent(self) -> bool:
        return self._throttle_waits >= MAX_THROTTLE_WAITS

    def _cool_down(self, seconds: float, area: str | None) -> None:
        """Wait, saying so as it goes.

        A silent two-minute sleep looks exactly like a crash. Sleeping in
        slices and logging the remaining time is the difference between "this
        is working" and "I think it has hung, I will kill it"."""
        remaining = seconds
        while remaining > 0:
            slice_seconds = min(COOLDOWN_TICK_SECONDS, remaining)
            log.warning("  throttled on %s — resuming in %.0fs",
                        area or "?", remaining)
            self._sleep(slice_seconds)
            remaining -= slice_seconds

    # ---------------------------------------------------------------- internals
    def _throttle(self) -> None:
        """Leave a fixed gap between requests, however fast the caller loops."""
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def _get(self, params: dict) -> dict:
        """One request, distinguishing three kinds of failure.

        Transient (500, 502, 503, 504) — the service is busy. Retry on a
        widening backoff of seconds.

        Throttled (403, 429) — we have asked too often. The identical request
        succeeds a minute later, so this is not a permanent refusal, but
        retrying it two seconds later is both useless and rude. The client
        stops, waits a couple of minutes, and tries again a small number of
        times before giving up on the series.

        Permanent (everything else 4xx) — a 404 will still be a 404 in four
        seconds. Fail immediately.

        Collapsing the first two into one policy is what got this client
        throttled out of a whole backfill.
        """
        last_error: Exception | None = None
        attempt = 0

        while attempt < MAX_RETRIES:
            self._throttle()
            try:
                response = self.session.get(
                    BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
                )
                self._last_request_at = time.monotonic()
                status = response.status_code

                if status in THROTTLE_STATUS_CODES:
                    if self.throttle_budget_spent:
                        raise ThrottledOut(
                            f"HTTP {status}: throttled {self._throttle_waits} "
                            f"times this run — today's allowance looks spent. "
                            f"Everything fetched is saved; re-run later and it "
                            f"will resume."
                        )
                    self._throttle_waits += 1
                    wait = THROTTLE_COOLDOWN_SECONDS * self._throttle_waits
                    log.warning("throttled (HTTP %d) on %s — cool-down %d of %d",
                                status, params.get("area"),
                                self._throttle_waits, MAX_THROTTLE_WAITS)
                    self._cool_down(wait, params.get("area"))
                    continue        # does not consume a retry attempt

                if status in RETRY_STATUS_CODES:
                    last_error = GridClientError(f"HTTP {status} from dashboard")
                    log.warning("attempt %d/%d: HTTP %d for %s",
                                attempt + 1, MAX_RETRIES, status, params.get("area"))
                elif status >= 400:
                    raise GridClientError(
                        f"HTTP {status} from dashboard for "
                        f"area={params.get('area')!r} — not retryable"
                    )
                else:
                    return response.json()

            except requests.RequestException as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                log.warning("attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, exc)
            except ValueError as exc:  # JSON decode
                self._last_request_at = time.monotonic()
                raise GridClientError(
                    "dashboard returned a response that was not JSON"
                ) from exc

            attempt += 1
            if attempt < MAX_RETRIES:
                self._sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        raise GridClientError(
            f"gave up after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------ public
    def fetch(self, area_label: str, region: str,
              start: date, end: date) -> list[Reading]:
        """Fetch one series for one region over an inclusive date range."""
        try:
            area = AREAS_BY_LABEL[area_label]
        except KeyError:
            raise GridClientError(f"unknown area {area_label!r}") from None

        params = {
            "area": area.key,
            "region": region,
            "datefrom": datetime(start.year, start.month, start.day, 0, 0)
                        .strftime(API_DATE_FORMAT),
            "dateto": datetime(end.year, end.month, end.day, 23, 59)
                      .strftime(API_DATE_FORMAT),
        }

        payload = self._get(params)
        try:
            return list(parse_response(payload, area.label, region))
        except ParseError as exc:
            raise GridClientError(f"could not parse {area_label} response: {exc}") from exc

    @staticmethod
    def windows(start: date, end: date) -> Iterator[tuple[date, date]]:
        """The request windows a range is split into. No network access.

        Exposed separately so the caller can decide whether a window is worth
        requesting — checking the fetch ledger first — rather than discovering
        that only after the request has been made.
        """
        if end < start:
            raise GridClientError(f"end {end} is before start {start}")

        window_start = start
        while window_start <= end:
            window_end = min(
                window_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end
            )
            yield window_start, window_end
            window_start = window_end + timedelta(days=1)

    def iter_chunks(self, area_label: str, region: str,
                    start: date, end: date) -> Iterator[tuple[date, date, list[Reading]]]:
        """Fetch a long range one bounded request at a time.

        Yields `(window_start, window_end, readings)` so the caller can commit
        each window as it arrives rather than holding a year in memory and
        writing nothing until the end.

        Chunking is not an optimisation. A single request for two years of
        15-minute data asks the service to build a response with tens of
        thousands of rows in it, which is precisely the behaviour the fair-use
        clause is aimed at.
        """
        for window_start, window_end in self.windows(start, end):
            log.info("fetching %s/%s %s to %s",
                     area_label, region, window_start, window_end)
            readings = self.fetch(area_label, region, window_start, window_end)
            yield window_start, window_end, readings

    def fetch_range(self, area_label: str, region: str,
                    start: date, end: date) -> Iterator[Reading]:
        """Every reading across a range, flattened.

        Convenience for callers that genuinely want the whole range at once.
        The ingest path deliberately does not use this — see `iter_chunks`.
        """
        for _, _, readings in self.iter_chunks(area_label, region, start, end):
            yield from readings

    def probe(self) -> tuple[bool, str]:
        """Check the service is answering, without writing anything.

        The first thing to run on a new machine. The endpoint was returning
        503s during development, so "is it up?" is a real question and
        deserves its own command rather than being discovered halfway through
        a backfill.
        """
        today = date.today()
        try:
            readings = self.fetch("wind", "ALL", today, today)
        except GridClientError as exc:
            return False, str(exc)

        if not readings:
            return False, "service answered but returned no rows"

        present = [r for r in readings if not r.is_missing]
        newest = max((r.timestamp_utc for r in present), default=None)
        return True, (
            f"{len(readings)} rows, {len(present)} with values, "
            f"most recent {newest:%Y-%m-%d %H:%M} UTC"
            if newest else f"{len(readings)} rows, all null so far today"
        )

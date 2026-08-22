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
    REQUEST_INTERVAL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_STATUS_CODES,
    USER_AGENT,
)
from .models import ParseError, Reading, parse_response

log = logging.getLogger(__name__)


class GridClientError(Exception):
    """The dashboard could not be reached, or refused to answer usefully."""


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
        """One request, with backoff on the failures that are worth retrying.

        A 503 from this service means "busy", not "gone" — it returned exactly
        that during development. Retrying with a widening gap is the correct
        response; retrying immediately in a tight loop is how a client gets
        itself throttled.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                response = self.session.get(
                    BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
                )
                self._last_request_at = time.monotonic()

                if response.status_code in RETRY_STATUS_CODES:
                    last_error = GridClientError(
                        f"HTTP {response.status_code} from dashboard"
                    )
                    log.warning(
                        "attempt %d/%d: HTTP %d for %s",
                        attempt + 1, MAX_RETRIES, response.status_code,
                        params.get("area"),
                    )
                elif response.status_code >= 400:
                    # Not in the retry set, so this is a permanent answer: a
                    # 404 will still be a 404 in four seconds. Retrying it only
                    # delays the error and adds load the licence asks us not to.
                    raise GridClientError(
                        f"HTTP {response.status_code} from dashboard for "
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

            if attempt < MAX_RETRIES - 1:
                self._sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

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

    def fetch_range(self, area_label: str, region: str,
                    start: date, end: date) -> Iterator[Reading]:
        """Fetch a long range as a series of bounded requests.

        Chunking is not an optimisation. A single request for two years of
        15-minute data asks the service to build a response with tens of
        thousands of rows in it, which is precisely the behaviour the fair-use
        clause is aimed at.
        """
        if end < start:
            raise GridClientError(f"end {end} is before start {start}")

        window_start = start
        while window_start <= end:
            window_end = min(
                window_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end
            )
            log.info("fetching %s/%s %s to %s",
                     area_label, region, window_start, window_end)
            yield from self.fetch(area_label, region, window_start, window_end)
            window_start = window_end + timedelta(days=1)

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

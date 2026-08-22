"""The one shape every reading takes, and the parsing that produces it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from .config import API_DATE_FORMAT, GRID_TIMEZONE

DUBLIN = ZoneInfo(GRID_TIMEZONE)


class ParseError(Exception):
    """The response did not look like what the dashboard normally returns."""


@dataclass(frozen=True)
class Reading:
    """A single measurement of one series, at one moment, for one region.

    `value` is deliberately Optional. EirGrid returns rows with a null value
    for periods that have not settled yet, and for gaps in metering. Those
    nulls must survive all the way into the database as NULL.

    Storing a missing reading as 0 would say the wind stopped blowing. On a
    grid where wind routinely covers a third of demand, that is not a rounding
    error — it is a fabricated blackout, and any model trained on it learns
    that outages happen every night at midnight.
    """

    area: str                 # canonical label, e.g. "wind"
    region: str               # ROI | NI | ALL
    timestamp_utc: datetime   # tz-aware, always UTC
    value: float | None

    @property
    def is_missing(self) -> bool:
        return self.value is None


def parse_effective_time(raw: str) -> datetime:
    """Turn EirGrid's local-time string into an aware UTC datetime.

    The API reports Irish local time. Ireland observes daylight saving, which
    means that on one night each October the same local time occurs twice.
    Python resolves that with `fold`, and we take the first occurrence — the
    alternative is silently shifting an hour of October data by an hour every
    single year.

    Storing UTC rather than local time is the whole reason this function
    exists. Lag features on a series stored in local time are wrong twice a
    year, and wrong in a way that is very hard to see in a chart.
    """
    text = (raw or "").strip()
    if not text:
        raise ParseError("empty EffectiveTime")

    parsed = None
    # The service has been seen to return both with and without seconds.
    for fmt in (API_DATE_FORMAT + ":%S", API_DATE_FORMAT, "%d-%b-%Y %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        raise ParseError(f"unrecognised EffectiveTime format: {raw!r}")

    local = parsed.replace(tzinfo=DUBLIN, fold=0)
    return local.astimezone(timezone.utc)


def _coerce_value(raw: Any) -> float | None:
    """Null stays null. Anything unparseable is treated as null, not as zero."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", "")
    if not text or text.lower() in {"null", "none", "-", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_response(payload: dict, area_label: str, region: str) -> Iterator[Reading]:
    """Yield Readings from one dashboard response.

    The service reports failure inside a 200 response rather than by status
    code, so the body has to be inspected before the rows are trusted.
    """
    if not isinstance(payload, dict):
        raise ParseError(f"expected a JSON object, got {type(payload).__name__}")

    status = payload.get("Status")
    if status is not None and str(status).lower() != "success":
        message = payload.get("ErrorMessage") or "no message given"
        raise ParseError(f"service reported status {status!r}: {message}")

    rows = payload.get("Rows")
    if rows is None:
        raise ParseError("response contained no 'Rows' key")
    if not isinstance(rows, list):
        raise ParseError(f"'Rows' was {type(rows).__name__}, expected a list")

    for row in rows:
        if not isinstance(row, dict):
            continue
        effective = row.get("EffectiveTime")
        if not effective:
            continue
        yield Reading(
            area=area_label,
            region=row.get("Region") or region,
            timestamp_utc=parse_effective_time(effective),
            value=_coerce_value(row.get("Value")),
        )

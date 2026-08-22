"""Parsing tests.

The null handling tests are the important ones. Every other bug in this
project is visible; a null silently becoming zero is not, and it corrupts
every model trained afterwards.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gridcast.models import ParseError, Reading, parse_effective_time, parse_response


# ------------------------------------------------------------------ timestamps
def test_parses_effective_time_to_utc():
    # January: Ireland is on UTC, so the clock time is unchanged.
    assert parse_effective_time("15-Jan-2026 09:30") == datetime(
        2026, 1, 15, 9, 30, tzinfo=timezone.utc
    )


def test_converts_irish_summer_time_to_utc():
    # July: IST is UTC+1, so 09:30 local is 08:30 UTC. Storing the local time
    # here would put every summer reading an hour late relative to winter.
    assert parse_effective_time("15-Jul-2026 09:30") == datetime(
        2026, 7, 15, 8, 30, tzinfo=timezone.utc
    )


def test_accepts_seconds_in_timestamp():
    assert parse_effective_time("15-Jan-2026 09:30:00") == datetime(
        2026, 1, 15, 9, 30, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("bad", ["", "   ", "not a date", "2026-01-15 09:30"])
def test_rejects_unparseable_timestamps(bad):
    with pytest.raises(ParseError):
        parse_effective_time(bad)


# ----------------------------------------------------------------------- nulls
def _payload(rows):
    return {"Status": "Success", "ErrorMessage": None, "Rows": rows}


def test_null_value_stays_none():
    readings = list(parse_response(
        _payload([{"EffectiveTime": "15-Jan-2026 09:30", "Region": "ALL", "Value": None}]),
        "wind", "ALL",
    ))
    assert len(readings) == 1
    assert readings[0].value is None
    assert readings[0].is_missing


@pytest.mark.parametrize("raw", [None, "", "null", "N/A", "-", "  "])
def test_missing_markers_never_become_zero(raw):
    """The whole point. A null wind reading is 'not measured', not 'no wind'."""
    readings = list(parse_response(
        _payload([{"EffectiveTime": "15-Jan-2026 09:30", "Region": "ALL", "Value": raw}]),
        "wind", "ALL",
    ))
    assert readings[0].value is None, f"{raw!r} was coerced to a number"


def test_genuine_zero_is_preserved():
    """A real measured zero must survive. Wind output can genuinely be ~0."""
    readings = list(parse_response(
        _payload([{"EffectiveTime": "15-Jan-2026 09:30", "Region": "ALL", "Value": 0}]),
        "wind", "ALL",
    ))
    assert readings[0].value == 0.0
    assert not readings[0].is_missing


def test_parses_numeric_strings_with_separators():
    readings = list(parse_response(
        _payload([{"EffectiveTime": "15-Jan-2026 09:30", "Region": "ALL", "Value": "1,234.5"}]),
        "wind", "ALL",
    ))
    assert readings[0].value == pytest.approx(1234.5)


# -------------------------------------------------------------------- envelope
def test_rejects_failure_status_inside_a_200_response():
    """The service reports errors in the body, not the status code."""
    payload = {"Status": "Failure", "ErrorMessage": "bad area", "Rows": []}
    with pytest.raises(ParseError, match="bad area"):
        list(parse_response(payload, "wind", "ALL"))


def test_rejects_missing_rows_key():
    with pytest.raises(ParseError, match="Rows"):
        list(parse_response({"Status": "Success"}, "wind", "ALL"))


def test_skips_rows_without_a_timestamp():
    readings = list(parse_response(
        _payload([
            {"EffectiveTime": None, "Value": 100},
            {"EffectiveTime": "15-Jan-2026 09:30", "Region": "ALL", "Value": 100},
        ]),
        "wind", "ALL",
    ))
    assert len(readings) == 1


def test_falls_back_to_requested_region_when_row_omits_it():
    readings = list(parse_response(
        _payload([{"EffectiveTime": "15-Jan-2026 09:30", "Value": 1}]),
        "wind", "ROI",
    ))
    assert readings[0].region == "ROI"

"""Client tests. No network access — the session is a stub.

A test suite that needs the internet is a test suite that fails on a train,
and one that hits a public service on every run is exactly the behaviour the
fair-use clause is about.
"""

from __future__ import annotations

from datetime import date

import pytest
import requests

from gridcast.client import GridClient, GridClientError


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "Status": "Success", "ErrorMessage": None, "Rows": []
        }

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Replays a queue of responses and records what was asked for."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(params or {})
        if not self._responses:
            return FakeResponse()
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(responses):
    slept = []
    client = GridClient(
        session=FakeSession(responses),
        interval_seconds=0,
        sleep=slept.append,
    )
    return client, slept


ROW = {"EffectiveTime": "15-Jan-2026 09:30", "Region": "ALL", "Value": 1234.5}


def test_fetch_returns_readings():
    client, _ = make_client([FakeResponse(payload={
        "Status": "Success", "Rows": [ROW]
    })])
    readings = client.fetch("wind", "ALL", date(2026, 1, 15), date(2026, 1, 15))
    assert len(readings) == 1
    assert readings[0].value == pytest.approx(1234.5)
    assert readings[0].area == "wind"


def test_sends_the_expected_query_parameters():
    client, _ = make_client([FakeResponse(payload={"Status": "Success", "Rows": []})])
    client.fetch("wind", "ROI", date(2026, 1, 15), date(2026, 1, 16))

    params = client.session.calls[0]
    assert params["area"] == "windactual"      # config maps label -> API key
    assert params["region"] == "ROI"
    assert params["datefrom"] == "15-Jan-2026 00:00"
    assert params["dateto"] == "16-Jan-2026 23:59"


def test_unknown_area_is_rejected_before_any_request():
    client, _ = make_client([])
    with pytest.raises(GridClientError, match="unknown area"):
        client.fetch("nonsense", "ALL", date(2026, 1, 15), date(2026, 1, 15))
    assert client.session.calls == []


# --------------------------------------------------------------------- retries
def test_retries_on_503_then_succeeds():
    """503 is the failure this service actually produces when busy."""
    client, slept = make_client([
        FakeResponse(status_code=503),
        FakeResponse(status_code=503),
        FakeResponse(payload={"Status": "Success", "Rows": [ROW]}),
    ])
    readings = client.fetch("wind", "ALL", date(2026, 1, 15), date(2026, 1, 15))
    assert len(readings) == 1
    assert len(client.session.calls) == 3


def test_backoff_widens_between_attempts():
    client, slept = make_client([
        FakeResponse(status_code=503),
        FakeResponse(status_code=503),
        FakeResponse(payload={"Status": "Success", "Rows": []}),
    ])
    client.fetch("wind", "ALL", date(2026, 1, 15), date(2026, 1, 15))
    waits = [s for s in slept if s > 0]
    assert waits == sorted(waits), "backoff should not shrink"
    assert len(waits) >= 2


def test_gives_up_after_max_retries():
    client, _ = make_client([FakeResponse(status_code=503)] * 10)
    with pytest.raises(GridClientError, match="gave up"):
        client.fetch("wind", "ALL", date(2026, 1, 15), date(2026, 1, 15))


def test_retries_on_connection_error():
    client, _ = make_client([
        requests.ConnectionError("network down"),
        FakeResponse(payload={"Status": "Success", "Rows": [ROW]}),
    ])
    readings = client.fetch("wind", "ALL", date(2026, 1, 15), date(2026, 1, 15))
    assert len(readings) == 1


def test_does_not_retry_a_404():
    """A 404 will still be a 404 in four seconds. Retrying only wastes time."""
    client, _ = make_client([FakeResponse(status_code=404)] * 5)
    with pytest.raises(GridClientError):
        client.fetch("wind", "ALL", date(2026, 1, 15), date(2026, 1, 15))
    assert len(client.session.calls) == 1


# --------------------------------------------------------------------- chunking
def test_long_range_is_split_into_bounded_requests():
    client, _ = make_client([
        FakeResponse(payload={"Status": "Success", "Rows": []}) for _ in range(10)
    ])
    list(client.fetch_range("wind", "ALL", date(2026, 1, 1), date(2026, 1, 21)))
    # 21 days at 7 days per request
    assert len(client.session.calls) == 3


def test_chunks_do_not_overlap_or_leave_gaps():
    client, _ = make_client([
        FakeResponse(payload={"Status": "Success", "Rows": []}) for _ in range(10)
    ])
    list(client.fetch_range("wind", "ALL", date(2026, 1, 1), date(2026, 1, 21)))

    starts = [c["datefrom"] for c in client.session.calls]
    ends = [c["dateto"] for c in client.session.calls]
    assert starts[0] == "01-Jan-2026 00:00"
    assert ends[0] == "07-Jan-2026 23:59"
    assert starts[1] == "08-Jan-2026 00:00"     # day after, no overlap, no gap
    assert ends[-1] == "21-Jan-2026 23:59"


def test_backwards_range_is_rejected():
    client, _ = make_client([])
    with pytest.raises(GridClientError, match="before start"):
        list(client.fetch_range("wind", "ALL", date(2026, 1, 21), date(2026, 1, 1)))

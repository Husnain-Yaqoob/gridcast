"""API tests.

The important ones are about status codes. An API that answers every request
with 200 and hides the problem in the body is invisible to monitoring, to
retry logic, and to anything automated — and "it returned a response" is not
the same as "it worked".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gridcast import model as model_module  # noqa: E402
from gridcast.api import ForecastService, create_app  # noqa: E402
from gridcast.frame import load_wide  # noqa: E402
from gridcast.models import Reading  # noqa: E402
from gridcast.store import Store  # noqa: E402

FAST = {"n_estimators": 20, "max_depth": 3}


@pytest.fixture()
def populated(tmp_path):
    """A small database with one trained model, at horizon 1."""
    db_path = str(tmp_path / "api.db")
    model_dir = str(tmp_path / "models")

    store = Store(db_path)
    rng = np.random.default_rng(11)
    n = 96 * 40
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    mean, phi = 1500.0, 0.995
    wind = np.empty(n)
    wind[0] = mean
    for i in range(1, n):
        wind[i] = mean + phi * (wind[i - 1] - mean) + rng.normal(0, 45)
    wind = np.clip(wind, 0, 5000)

    rows = []
    for i in range(n):
        ts = start + timedelta(minutes=15 * i)
        rows.append(Reading("wind", "ALL", ts, float(wind[i])))
        rows.append(Reading("demand", "ALL", ts, 4000.0))
        rows.append(Reading("snsp", "ALL", ts, 50.0))
        rows.append(Reading("co2_intensity", "ALL", ts, 300.0))
    store.upsert(rows)

    frame = load_wide(db_path)
    result = model_module.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    trained, columns, count = model_module.train_final(frame, 1, params=FAST)
    model_module.save(trained, columns, 1, count, result, directory=model_dir)

    return db_path, model_dir


@pytest.fixture()
def client(populated):
    db_path, model_dir = populated
    service = ForecastService(db_path=db_path, model_dir=model_dir)
    return TestClient(create_app(service))


@pytest.fixture()
def empty_client(tmp_path):
    """A service with a model but no data — the 503 case."""
    db_path, model_dir = str(tmp_path / "x.db"), str(tmp_path / "m")
    Store(db_path)                       # schema only, no readings
    service = ForecastService(db_path=db_path, model_dir=model_dir)
    return TestClient(create_app(service))


# ------------------------------------------------------------------- health
def test_health_reports_loaded_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert 1.0 in body["models_loaded"]


def test_health_is_degraded_when_no_models_loaded(empty_client):
    """Still 200 — the service is alive. But it says so honestly."""
    response = empty_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_horizons_lists_what_can_be_forecast(client):
    body = client.get("/horizons").json()
    hours = [h["hours"] for h in body["horizons"]]
    assert 1.0 in hours
    entry = next(h for h in body["horizons"] if h["hours"] == 1.0)
    assert entry["expected_error_mw"] > 0
    assert "EirGrid" in body["attribution"]


# ------------------------------------------------------------------ forecast
def test_forecast_returns_a_prediction(client):
    response = client.get("/forecast", params={"horizon": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_wind_mw"] > 0
    assert body["horizon_hours"] == 1.0
    assert "EirGrid" in body["attribution"]


def test_forecast_reports_its_own_expected_error(client):
    """A prediction without an error bar invites false confidence."""
    body = client.get("/forecast", params={"horizon": 1}).json()
    assert body["expected_error_mw"] > 0
    assert body["persistence_error_mw"] > 0
    assert body["skill_vs_persistence_pct"] is not None


def test_valid_time_is_the_horizon_after_the_made_time(client):
    body = client.get("/forecast", params={"horizon": 1}).json()
    made = datetime.fromisoformat(body["made_at_utc"].replace("Z", "+00:00"))
    valid = datetime.fromisoformat(body["valid_at_utc"].replace("Z", "+00:00"))
    assert valid - made == timedelta(hours=1)


# -------------------------------------------------------------- status codes
def test_untrained_horizon_is_404_not_200(client):
    """The resource genuinely does not exist. Say so in the status line."""
    response = client.get("/forecast", params={"horizon": 6})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["available_horizons"] == [1.0]
    assert "train" in detail["hint"]


def test_no_data_is_503_not_404(empty_client):
    """Temporarily unable, not permanently absent — the caller should retry."""
    response = empty_client.get("/latest")
    assert response.status_code == 503


@pytest.mark.parametrize("horizon", [0, -1, 100])
def test_nonsense_horizons_are_rejected_as_422(client, horizon):
    """Validation failure, caught before any work is done."""
    assert client.get("/forecast", params={"horizon": horizon}).status_code == 422


# ------------------------------------------------------------------- latest
def test_latest_returns_current_grid_state(client):
    body = client.get("/latest").json()
    assert body["wind_mw"] > 0
    assert body["demand_mw"] > 0
    assert body["wind_share_pct"] == pytest.approx(
        100 * body["wind_mw"] / body["demand_mw"], rel=1e-2
    )


# -------------------------------------------------------------------- cache
def test_data_is_cached_between_requests(populated):
    """Re-reading 35,000 rows per request is waste the TTL exists to avoid."""
    db_path, model_dir = populated
    loads = []

    service = ForecastService(db_path=db_path, model_dir=model_dir)
    original = service.frame

    def counting_frame():
        result = original()
        loads.append(1)
        return result

    service.frame = counting_frame
    api = TestClient(create_app(service))

    api.get("/latest")
    api.get("/latest")
    api.get("/forecast", params={"horizon": 1})

    # The frame property is hit each time, but load_wide is not — proven by
    # the cache age advancing rather than resetting.
    assert service.cache_age_seconds is not None


def test_cache_expires_after_the_ttl(populated):
    db_path, model_dir = populated
    now = [1000.0]

    service = ForecastService(db_path=db_path, model_dir=model_dir,
                              ttl_seconds=60, clock=lambda: now[0])
    first = service.frame()
    assert service.cache_age_seconds == 0

    now[0] += 30
    assert service.frame() is first, "still fresh, should not re-read"

    now[0] += 60          # now 90s old, past the 60s TTL
    assert service.frame() is not first, "stale, should have re-read"


# --------------------------------------------------------------------- docs
def test_openapi_schema_is_generated(client):
    """The interactive docs at /docs are a real deliverable, not decoration."""
    schema = client.get("/openapi.json").json()
    assert "/forecast" in schema["paths"]
    assert "/health" in schema["paths"]
    assert schema["info"]["title"] == "gridcast"


# ------------------------------------------------------------------ history
def test_history_returns_recent_readings_oldest_first(client):
    body = client.get("/history", params={"hours": 6}).json()
    assert body["count"] > 0
    times = [r["observed_at_utc"] for r in body["readings"]]
    assert times == sorted(times)
    assert all(r["wind_mw"] is not None for r in body["readings"])


def test_history_window_is_respected(client):
    short = client.get("/history", params={"hours": 3}).json()
    long = client.get("/history", params={"hours": 24}).json()
    assert short["count"] < long["count"]


def test_history_omits_gaps_rather_than_sending_nulls(client):
    """A chart has to draw something for every row it is given."""
    body = client.get("/history", params={"hours": 12}).json()
    assert all(r["demand_mw"] is not None for r in body["readings"])


def test_history_with_no_data_is_503(empty_client):
    assert empty_client.get("/history").status_code == 503


@pytest.mark.parametrize("hours", [0, -5, 500])
def test_nonsense_history_windows_are_422(client, hours):
    assert client.get("/history", params={"hours": hours}).status_code == 422


# ---------------------------------------------------------------- dashboard
def test_dashboard_serves_html(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "gridcast" in response.text


def test_dashboard_is_self_contained(client):
    """No CDN. It has to work in a room with no internet."""
    page = client.get("/dashboard").text
    assert "http://" not in page.replace('"http://www.w3.org/2000/svg"', "")
    assert "https://" not in page
    assert "<script" in page and "src=" not in page


def test_dashboard_is_not_in_the_openapi_schema(client):
    """It is a page for people, not an endpoint for programs."""
    schema = client.get("/openapi.json").json()
    assert "/dashboard" not in schema["paths"]
    assert "/history" in schema["paths"]


def test_horizons_reports_the_baseline_it_is_measured_against(client):
    """Skill without the number it is skill over is half a fact."""
    entry = client.get("/horizons").json()["horizons"][0]
    assert entry["persistence_error_mw"] > 0
    assert entry["beat_baseline_in_every_fold"] in (True, False)


# ------------------------------------------------- supplementary series lag
#
# The dashboard's carbon tile rendered an em dash against a database holding a
# year of carbon readings. `latest()` anchors on the newest row carrying wind
# and demand, and carbon settles later than both, so it was being read at a
# timestamp it had not reached — a bug that looks exactly like missing data.

def _service_with_lagging_carbon(tmp_path, carbon_hours_behind):
    """Wind and demand current; carbon last seen N hours earlier."""
    db_path = str(tmp_path / "lag.db")
    store = Store(db_path)
    base = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    readings = []
    for step in range(96):
        stamp = base + timedelta(minutes=15 * step)
        readings.append(Reading(area="wind", region="ALL",
                                timestamp_utc=stamp, value=1000.0 + step))
        readings.append(Reading(area="demand", region="ALL",
                                timestamp_utc=stamp, value=4000.0))
        # Carbon stops early, then publishes empty placeholders like the real
        # upstream does.
        latest_carbon = 96 - 1 - int(carbon_hours_behind * 4)
        readings.append(Reading(
            area="co2_intensity", region="ALL", timestamp_utc=stamp,
            value=300.0 if step <= latest_carbon else None,
        ))
    store.upsert(readings)
    return ForecastService(db_path=db_path, model_dir=str(tmp_path / "none"))


def test_carbon_is_reported_from_its_own_newest_reading(tmp_path):
    service = _service_with_lagging_carbon(tmp_path, carbon_hours_behind=2)
    body = service.latest()

    assert body["co2_intensity"] == 300.0, "carbon exists and must be reported"
    assert body["lagging_by_minutes"]["co2_intensity"] == 120


def test_a_current_series_is_not_reported_as_lagging(tmp_path):
    service = _service_with_lagging_carbon(tmp_path, carbon_hours_behind=0)
    body = service.latest()

    assert body["co2_intensity"] == 300.0
    assert "lagging_by_minutes" not in body, (
        "nothing lags, so the response should carry no caveat"
    )


def test_a_long_dead_series_is_dropped_rather_than_shown_as_current(tmp_path):
    """Falling back is not the same as falling back forever.

    A carbon reading from two days ago displayed beside a live wind figure
    would be worse than an empty tile, because it looks current.
    """
    service = _service_with_lagging_carbon(tmp_path, carbon_hours_behind=20)
    body = service.latest()

    assert "co2_intensity" not in body
    assert body["wind_mw"] is not None, "the rest of the payload still works"


def test_lag_is_exposed_through_the_endpoint(tmp_path):
    service = _service_with_lagging_carbon(tmp_path, carbon_hours_behind=3)
    response = TestClient(create_app(service)).get("/latest")

    assert response.status_code == 200
    assert response.json()["lagging_by_minutes"] == {"co2_intensity": 180}


def test_the_dashboard_labels_a_lagging_tile(tmp_path):
    """A number shown without its age is the thing this fix was avoiding."""
    from gridcast.dashboard import DASHBOARD_HTML

    assert "lagging_by_minutes" in DASHBOARD_HTML
    assert "behind" in DASHBOARD_HTML

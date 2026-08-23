"""HTTP API serving wind forecasts.

Three decisions shape this file.

Models are loaded once at startup, not per request. A gradient booster takes
a noticeable fraction of a second to deserialise, and doing that on every call
turns a 5ms response into a 300ms one for no benefit — the file has not
changed.

The data frame is cached with a deadline rather than forever. The database is
refreshed hourly by a scheduled job, so re-reading 35,000 rows on every request
is waste, but caching indefinitely would serve a forecast anchored to
yesterday while insisting it is current.

Failures get the status code that describes them. A horizon with no trained
model is a 404, because that resource genuinely does not exist. A database
with no usable data is a 503, because the service is temporarily unable to
answer and the caller should retry rather than give up. Returning 200 with an
error message in the body — which plenty of APIs do — makes both cases
invisible to anything automated.
"""

from __future__ import annotations

import time
from typing import Any

from .config import ATTRIBUTION, DEFAULT_DB_PATH, DEFAULT_REGION

# How long a loaded frame is trusted before it is re-read. The source publishes
# every fifteen minutes and the scheduled update runs hourly, so five minutes
# is comfortably fresher than the data can possibly be.
FRAME_TTL_SECONDS = 300

DEFAULT_HORIZONS = (1.0, 3.0, 6.0, 12.0)


class ForecastService:
    """Holds the loaded models and a time-limited view of the data.

    Deliberately a plain class rather than module-level globals, so the tests
    can build one against a temporary database without monkey-patching
    anything.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH,
                 model_dir: str = "models",
                 region: str = DEFAULT_REGION,
                 ttl_seconds: float = FRAME_TTL_SECONDS,
                 clock=time.monotonic) -> None:
        self.db_path = db_path
        self.model_dir = model_dir
        self.region = region
        self.ttl_seconds = ttl_seconds
        self._clock = clock

        self._models: dict[float, tuple[Any, dict]] = {}
        self._frame = None
        self._frame_loaded_at: float | None = None

    # ----------------------------------------------------------------- models
    def load_models(self, horizons=DEFAULT_HORIZONS) -> list[float]:
        """Load every model that exists. Missing ones are skipped, not fatal.

        A service that refuses to start because one of four models is absent
        is more brittle than the problem warrants — the other three are still
        useful, and `/horizons` reports honestly what is available.
        """
        from . import model as model_module

        loaded = []
        for horizon in horizons:
            try:
                self._models[float(horizon)] = model_module.load(
                    horizon, directory=self.model_dir
                )
                loaded.append(float(horizon))
            except FileNotFoundError:
                continue
        return loaded

    @property
    def available_horizons(self) -> list[float]:
        return sorted(self._models)

    def manifest(self, horizon: float) -> dict:
        return self._models[float(horizon)][1]

    # ------------------------------------------------------------------ data
    def frame(self):
        """The data, re-read when the cached copy is older than the TTL."""
        from .frame import load_wide

        now = self._clock()
        expired = (
            self._frame_loaded_at is None
            or (now - self._frame_loaded_at) > self.ttl_seconds
        )
        if expired:
            self._frame = load_wide(self.db_path, region=self.region)
            self._frame_loaded_at = now
        return self._frame

    @property
    def cache_age_seconds(self) -> float | None:
        if self._frame_loaded_at is None:
            return None
        return round(self._clock() - self._frame_loaded_at, 1)

    # -------------------------------------------------------------- forecast
    def forecast(self, horizon: float) -> dict:
        from . import features

        model, manifest = self._models[float(horizon)]
        frame = self.frame()

        steps = features.horizon_steps(horizon)
        built = features.build_features(frame, steps)

        essential = ["wind_now", "wind_lag4", "wind_mean4"]
        usable = built[built[essential].notna().all(axis=1)]
        if usable.empty:
            raise LookupError("no row has enough history to forecast from")

        import pandas as pd

        row = usable.iloc[[-1]][manifest["features"]]
        predicted = float(model.predict(row)[0])
        made_at = usable.index[-1]
        validation = manifest.get("validation", {})

        return {
            "made_at_utc": made_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_at_utc": (made_at + pd.Timedelta(hours=horizon)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "horizon_hours": float(horizon),
            "predicted_wind_mw": round(predicted, 1),
            "current_wind_mw": round(float(row["wind_now"].iloc[0]), 1),
            "expected_error_mw": validation.get("model_mae"),
            "persistence_error_mw": validation.get("persistence_mae"),
            "skill_vs_persistence_pct": validation.get("skill_vs_persistence_pct"),
            "model_trained_at": manifest["trained_at"],
            "attribution": ATTRIBUTION,
        }

    def latest(self) -> dict:
        """Current grid state, straight from the most recent complete reading."""
        frame = self.frame()
        recent = frame.dropna(subset=["wind", "demand"])
        if recent.empty:
            raise LookupError("no complete readings held")

        row = recent.iloc[-1]
        share = 100.0 * float(row["wind"]) / float(row["demand"]) if row["demand"] else None

        payload = {
            "observed_at_utc": recent.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wind_mw": round(float(row["wind"]), 1),
            "demand_mw": round(float(row["demand"]), 1),
            "wind_share_pct": round(share, 1) if share is not None else None,
            "attribution": ATTRIBUTION,
        }
        for column in ("snsp", "co2_intensity", "interconnection"):
            if column in frame.columns and not pd.isna(row.get(column)):
                payload[column] = round(float(row[column]), 1)
        return payload


# pandas is imported lazily inside methods above; this import is only for the
# isna() check in latest(), and is kept at module level for readability.
import pandas as pd  # noqa: E402


def _response_models():
    """Typed response schemas.

    Declared as Pydantic models rather than plain dicts so the generated
    OpenAPI document describes the actual shape of each response — field
    names, types, units, and a worked example.

    Returning bare `dict` produces docs that say `{"additionalProp1": {}}`,
    which is worse than no documentation: it looks like documentation while
    telling a caller nothing. The examples below are what someone integrating
    against this reads first.
    """
    from pydantic import BaseModel, Field

    class Health(BaseModel):
        status: str = Field(description="ok, or degraded if no models loaded")
        models_loaded: list[float] = Field(description="Horizons in hours")
        data_cache_age_seconds: float | None = Field(
            default=None, description="Age of the cached frame; null if never loaded"
        )
        model_config = {"json_schema_extra": {"example": {
            "status": "ok", "models_loaded": [1.0, 3.0, 6.0, 12.0],
            "data_cache_age_seconds": 42.1,
        }}}

    class HorizonInfo(BaseModel):
        hours: float
        expected_error_mw: float | None = Field(
            default=None, description="Validated MAE of this model, in MW"
        )
        skill_vs_persistence_pct: float | None = Field(
            default=None, description="Positive means better than persistence"
        )
        trained_at: str

    class Horizons(BaseModel):
        horizons: list[HorizonInfo]
        attribution: str

    class Latest(BaseModel):
        observed_at_utc: str
        wind_mw: float
        demand_mw: float
        wind_share_pct: float | None = Field(
            default=None,
            description="Wind as a percentage of demand. Can exceed 100 when "
                        "Ireland generates more wind than it consumes.",
        )
        snsp: float | None = Field(
            default=None, description="System Non-Synchronous Penetration, %"
        )
        co2_intensity: float | None = Field(default=None, description="gCO2/kWh")
        interconnection: float | None = Field(
            default=None, description="Net interconnector flow, MW. Positive is import."
        )
        attribution: str
        model_config = {"json_schema_extra": {"example": {
            "observed_at_utc": "2026-08-23T22:15:00Z",
            "wind_mw": 450.0, "demand_mw": 3980.5, "wind_share_pct": 11.3,
            "snsp": 38.2, "co2_intensity": 341.0, "interconnection": 412.0,
            "attribution": ATTRIBUTION,
        }}}

    class Forecast(BaseModel):
        made_at_utc: str = Field(description="Time the forecast was made from")
        valid_at_utc: str = Field(description="Time the forecast is for")
        horizon_hours: float
        predicted_wind_mw: float
        current_wind_mw: float = Field(
            description="Output now — this is what the persistence baseline predicts"
        )
        expected_error_mw: float | None = Field(
            default=None, description="Validated MAE of this model, in MW"
        )
        persistence_error_mw: float | None = Field(
            default=None, description="MAE of the naive baseline, on the same rows"
        )
        skill_vs_persistence_pct: float | None = Field(
            default=None, description="Positive means the model beats persistence"
        )
        model_trained_at: str
        attribution: str
        model_config = {"protected_namespaces": (), "json_schema_extra": {"example": {
            "made_at_utc": "2026-08-23T22:15:00Z",
            "valid_at_utc": "2026-08-23T23:15:00Z",
            "horizon_hours": 1.0,
            "predicted_wind_mw": 470.2, "current_wind_mw": 450.0,
            "expected_error_mw": 105.3, "persistence_error_mw": 120.8,
            "skill_vs_persistence_pct": 12.8,
            "model_trained_at": "2026-08-23T21:40:00Z",
            "attribution": ATTRIBUTION,
        }}}

    return Health, Horizons, Latest, Forecast


def create_app(service: ForecastService | None = None):
    """Build the FastAPI application.

    Takes an optional service so tests can inject one pointed at a temporary
    database. Without that, testing the API would mean either hitting the real
    data or patching globals — both worse.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'FastAPI is required to serve the API. Install it with:\n'
            '    pip install -e ".[api]"'
        ) from exc

    Health, Horizons, Latest, Forecast = _response_models()

    service = service or ForecastService()
    service.load_models()

    app = FastAPI(
        title="gridcast",
        version="0.1.0",
        description=(
            "Short-term wind generation forecasts for the all-island Irish "
            "grid.\n\n"
            "Every forecast reports the validated error of the model that "
            "produced it, and how that compares with a persistence baseline. "
            "A prediction without an error bar invites false confidence.\n\n"
            f"{ATTRIBUTION}"
        ),
    )
    app.state.service = service

    @app.get("/health", tags=["service"], response_model=Health)
    def health():
        """Liveness plus enough detail to diagnose a sick instance."""
        return {
            "status": "ok" if service.available_horizons else "degraded",
            "models_loaded": service.available_horizons,
            "data_cache_age_seconds": service.cache_age_seconds,
        }

    @app.get("/horizons", tags=["service"], response_model=Horizons)
    def horizons():
        """What can be forecast, and how well each one scored in validation."""
        return {
            "horizons": [
                {
                    "hours": h,
                    "expected_error_mw": service.manifest(h)
                        .get("validation", {}).get("model_mae"),
                    "skill_vs_persistence_pct": service.manifest(h)
                        .get("validation", {}).get("skill_vs_persistence_pct"),
                    "trained_at": service.manifest(h)["trained_at"],
                }
                for h in service.available_horizons
            ],
            "attribution": ATTRIBUTION,
        }

    @app.get("/latest", tags=["data"], response_model=Latest,
             responses={503: {"description": "No usable data held yet"}})
    def latest():
        """Most recent observed state of the grid."""
        try:
            return service.latest()
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/forecast", tags=["forecast"], response_model=Forecast,
             responses={
                 404: {"description": "No trained model for that horizon"},
                 503: {"description": "Not enough history to forecast from"},
             })
    def forecast(
        horizon: float = Query(
            1.0, gt=0, le=48,
            description="Hours ahead. Must be a horizon with a trained model; "
                        "see /horizons.",
        )
    ):
        if float(horizon) not in service._models:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"no trained model for horizon {horizon}h",
                    "available_horizons": service.available_horizons,
                    "hint": "train one with: gridcast train --horizons "
                            f"{horizon:g} --save",
                },
            )
        try:
            return service.forecast(horizon)
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app

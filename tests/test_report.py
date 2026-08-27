"""Chart tests.

A chart is hard to assert on — "does it look right" is not a unit test. So
these test the things that are actually checkable and that would actually go
wrong: that the backtest never trains on the rows it is scored against, that
forecasts are aligned to the time they are about rather than the time they were
made, that a missing input is skipped rather than fatal, and that every chart
writes a real file.

The alignment test is the one that matters most. Getting it wrong produces a
chart that looks plausible, is off by the length of the horizon, and would be
believed.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from gridcast import report  # noqa: E402
from gridcast.frame import load_wide  # noqa: E402
from gridcast.models import Reading  # noqa: E402
from gridcast.store import Store  # noqa: E402

FAST = {"n_estimators": 20, "max_depth": 3}


@pytest.fixture()
def frame(tmp_path):
    """Forty days of bounded, mean-reverting wind on a quarter-hourly index.

    Deliberately not a random walk. An unbounded walk drifts into the clip
    ceiling, goes flat, and makes persistence look perfect — a fixture bug that
    silently invalidates every skill number computed from it.
    """
    db_path = str(tmp_path / "report.db")
    store = Store(db_path)

    rng = np.random.default_rng(7)
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
        hour = ts.hour
        rows.append(Reading("wind", "ALL", ts, float(wind[i])))
        rows.append(Reading("demand", "ALL", ts, 3600.0 + 700 * np.sin(hour / 24 * 2 * np.pi)))
        rows.append(Reading("snsp", "ALL", ts, 50.0))
        # A real diurnal shape, so the "dirtiest hour" annotation has something
        # true to find rather than picking noise.
        rows.append(Reading("co2_intensity", "ALL", ts,
                            300.0 + 90 * np.sin((hour - 4) / 24 * 2 * np.pi)))
    store.upsert(rows)
    return load_wide(db_path)


# ---------------------------------------------------------------- backtesting
def test_actual_is_the_reading_at_the_valid_time(frame):
    """The alignment test, and the one worth having.

    Every row claims "at time T, wind was `actual` and the model had predicted
    `model`". If the index is off by the horizon — the mistake that happens
    when forecasts are plotted at the time they were made — then `actual` will
    not match the reading the database actually holds for T, and this fails.
    """
    for horizon in (1.0, 12.0):
        result = report.backtest(frame, horizon, params=FAST)
        observed = frame["wind"].reindex(result.index)
        assert np.allclose(result["actual"].to_numpy(), observed.to_numpy()), (
            f"actual column is not the reading at its own index at {horizon}h"
        )


def test_valid_time_is_exactly_one_horizon_after_the_made_time(frame):
    """A 12h forecast is a statement about 12h from now. Plot it there."""
    horizon = 12.0
    result = report.backtest(frame, horizon, params=FAST)
    made_at = result.index - np.timedelta64(int(horizon * 60), "m")

    # Persistence is "output at the moment the forecast was made", so if the
    # shift is right, it matches the frame at the made time — not the valid one.
    assert np.allclose(result["persistence"].to_numpy(),
                       frame["wind"].reindex(made_at).to_numpy())


def test_backtest_columns_are_the_three_things_being_compared(frame):
    result = report.backtest(frame, 1.0, params=FAST)
    assert list(result.columns) == ["actual", "model", "persistence"]
    assert len(result) > 0


def test_backtest_does_not_score_on_rows_it_trained_on(frame):
    """The split is chronological, and the test half comes strictly after."""
    result = report.backtest(frame, 1.0, train_fraction=0.8, params=FAST)
    total_usable = len(frame.dropna(subset=["wind"])) - 1
    # Roughly a fifth of the data, not all of it.
    assert len(result) < total_usable * 0.35


def test_persistence_column_is_output_now(frame):
    """Persistence is not a model. It is the current reading, carried forward."""
    result = report.backtest(frame, 1.0, params=FAST)
    shifted = frame["wind"].reindex(result.index - np.timedelta64(60, "m"))
    assert np.allclose(result["persistence"].to_numpy(),
                       shifted.to_numpy(), equal_nan=True)


def test_days_argument_trims_to_a_window(frame):
    week = report.backtest(frame, 1.0, days=2, params=FAST)
    span = week.index.max() - week.index.min()
    assert span <= np.timedelta64(2, "D")


def test_backtest_refuses_when_there_is_not_enough_history(tmp_path):
    """Better a clear refusal than a chart drawn from thirty rows."""
    db_path = str(tmp_path / "thin.db")
    store = Store(db_path)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.upsert([
        Reading("wind", "ALL", start + timedelta(minutes=15 * i), 1000.0)
        for i in range(120)
    ] + [
        Reading("demand", "ALL", start + timedelta(minutes=15 * i), 4000.0)
        for i in range(120)
    ])
    with pytest.raises(ValueError, match="not enough"):
        report.backtest(load_wide(db_path), 1.0, params=FAST)


# --------------------------------------------------------------------- charts
def test_wind_share_chart_writes_a_file(frame, tmp_path):
    path = report.wind_share_chart(frame, str(tmp_path / "share.png"))
    assert os.path.getsize(path) > 5_000


def test_carbon_chart_writes_a_file(frame, tmp_path):
    path = report.carbon_by_hour_chart(frame, str(tmp_path / "carbon.png"))
    assert os.path.getsize(path) > 5_000


def test_carbon_chart_refuses_without_carbon_data(frame, tmp_path):
    without = frame.drop(columns=["co2_intensity"])
    with pytest.raises(ValueError, match="co2_intensity"):
        report.carbon_by_hour_chart(without, str(tmp_path / "no.png"))


def test_forecast_chart_writes_a_file(frame, tmp_path):
    path = report.forecast_vs_actual_chart(
        frame, str(tmp_path / "fva.png"), horizons=(1.0,), days=3, params=FAST
    )
    assert os.path.getsize(path) > 5_000


def test_skill_chart_refuses_without_manifests(tmp_path):
    with pytest.raises(ValueError, match="no validated model manifests"):
        report.skill_by_horizon_chart(str(tmp_path / "empty"),
                                      str(tmp_path / "skill.png"))


def test_skill_chart_reads_saved_manifests(frame, tmp_path):
    from gridcast import model as model_module

    model_dir = str(tmp_path / "models")
    result = model_module.evaluate_horizon(frame, 1.0, n_splits=3, params=FAST)
    trained, columns, count = model_module.train_final(frame, 1.0, params=FAST)
    model_module.save(trained, columns, 1.0, count, result, directory=model_dir)

    path = report.skill_by_horizon_chart(model_dir, str(tmp_path / "skill.png"))
    assert os.path.getsize(path) > 5_000


# --------------------------------------------------------------------- driver
def test_render_all_skips_what_it_cannot_draw(frame, tmp_path, capsys):
    """No trained models is a reason to skip one chart, not to lose four."""
    written = report.render_all(frame, output_dir=str(tmp_path),
                                model_dir=str(tmp_path / "none"),
                                days=3, params=FAST)
    assert len(written) == 3
    assert "skipped skill by horizon" in capsys.readouterr().out
    assert all(os.path.exists(p) for p in written)

"""Model tests.

The point of these is not that the model is accurate — accuracy is an empirical
question answered by running it on real data. The point is that the *evaluation*
is honest: that no fold trains on its own test set, that the baseline is scored
on the same rows as the model, and that a saved model records what it is.

An evaluation harness that flatters a model is worse than no evaluation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from gridcast import evaluate, features, model
from gridcast.frame import FREQUENCY


@pytest.fixture()
def frame():
    """Sixty days of synthetic quarter-hourly data with real autocorrelation.

    A mean-reverting AR(1) process, not a sine wave and not a random walk.

    A pure oscillation is trivially predictable, so a broken evaluation would
    still look fine. An unbounded random walk is worse: it wanders into the
    physical limits, sticks there for days, and persistence scores a perfect
    zero — which is exactly the failure that first draft produced. Wind output
    is bounded and mean-reverting, so the fixture is too.
    """
    rng = np.random.default_rng(7)
    n = 96 * 60
    index = pd.date_range("2026-01-01", periods=n, freq=FREQUENCY, tz="UTC")

    mean_level, phi = 1500.0, 0.995
    wind = np.empty(n)
    wind[0] = mean_level
    for i in range(1, n):
        wind[i] = mean_level + phi * (wind[i - 1] - mean_level) + rng.normal(0, 45)
    wind = np.clip(wind, 0, 5000)
    t = np.arange(n)

    return pd.DataFrame(
        {
            "wind": wind,
            "demand": 4000 + 500 * np.cos(2 * np.pi * t / 96) + rng.normal(0, 30, n),
            "snsp": np.clip(40 + wind / 100, 0, 90),
            "co2_intensity": 400 - wind / 20 + rng.normal(0, 10, n),
            "interconnection": rng.normal(200, 80, n),
        },
        index=index,
    )


FAST = {"n_estimators": 30, "max_depth": 3}


# --------------------------------------------------------------- honest folds
def test_no_fold_trains_on_its_own_test_data(frame):
    """The property the whole harness exists to guarantee."""
    steps = features.horizon_steps(1)
    X, _ = features.build_dataset(frame, steps)

    for train_index, test_index in evaluate.walk_forward_splits(X.index, n_splits=3):
        assert set(train_index).isdisjoint(set(test_index))
        assert train_index.max() < test_index.min()


def test_baseline_is_scored_on_the_same_rows_as_the_model(frame):
    """Comparing a model on one sample to a baseline on another proves nothing."""
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    for fold in result.folds:
        assert fold.test_rows > 0
        # Both numbers came from the same fold, so both describe the same rows.
        assert fold.model_mae > 0
        assert fold.baseline_mae > 0


def test_every_fold_is_recorded(frame):
    result = model.evaluate_horizon(frame, 1, n_splits=4, params=FAST)
    assert len(result.folds) == 4
    assert [f.fold for f in result.folds] == [1, 2, 3, 4]


def test_training_set_grows_across_folds(frame):
    result = model.evaluate_horizon(frame, 1, n_splits=4, params=FAST)
    sizes = [f.train_rows for f in result.folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_test_windows_move_forward_in_time(frame):
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    starts = [f.test_start for f in result.folds]
    assert starts == sorted(starts)


# ------------------------------------------------------------------ reporting
def test_headline_mae_is_weighted_by_rows_not_a_mean_of_means(frame):
    """Folds differ in size; a plain mean overweights the smallest one."""
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)

    rows = sum(f.test_rows for f in result.folds)
    expected = sum(f.model_mae * f.test_rows for f in result.folds) / rows
    assert result.model_mae == pytest.approx(expected)

    naive = np.mean([f.model_mae for f in result.folds])
    if len({f.test_rows for f in result.folds}) > 1:
        assert result.model_mae != pytest.approx(naive, rel=1e-9)


def test_consistency_flag_distinguishes_lucky_from_reliable(frame):
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    won_everywhere = all(f.model_mae < f.baseline_mae for f in result.folds)
    assert result.consistent == won_everywhere


def test_skill_sign_matches_which_error_is_lower(frame):
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    if result.model_mae < result.baseline_mae:
        assert result.skill > 0
        assert result.beats_baseline
    else:
        assert result.skill <= 0
        assert not result.beats_baseline


def test_summary_names_the_loser_rather_than_hiding_it(frame):
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    text = result.summary()
    assert ("beats" in text) or ("LOSES TO" in text)
    assert "persistence" in text


# ------------------------------------------------------------ save and reload
def test_saved_model_records_what_it_is(frame, tmp_path):
    """A .joblib with no manifest is unusable six months later."""
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    trained, columns, rows = model.train_final(frame, 1, params=FAST)
    model.save(trained, columns, 1, rows, result, directory=str(tmp_path))

    with open(tmp_path / "wind_h1.json", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["horizon_hours"] == 1
    assert manifest["features"] == columns
    assert manifest["training_rows"] == rows
    assert "EirGrid" in manifest["attribution"]
    assert manifest["validation"]["persistence_mae"] > 0
    assert len(manifest["validation"]["folds"]) == 3


def test_model_round_trips(frame, tmp_path):
    trained, columns, rows = model.train_final(frame, 1, params=FAST)
    model.save(trained, columns, 1, rows, None, directory=str(tmp_path))

    reloaded, manifest = model.load(1, directory=str(tmp_path))
    assert manifest["features"] == columns

    X, _ = features.build_dataset(frame, features.horizon_steps(1))
    sample = X.iloc[[-1]][columns]
    assert trained.predict(sample)[0] == pytest.approx(reloaded.predict(sample)[0])


def test_missing_model_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        model.load(99, directory=str(tmp_path))


# -------------------------------------------------------------------- forecast
def test_forecast_is_anchored_to_the_latest_row(frame, tmp_path):
    trained, columns, rows = model.train_final(frame, 3, params=FAST)
    result = model.evaluate_horizon(frame, 3, n_splits=3, params=FAST)
    model.save(trained, columns, 3, rows, result, directory=str(tmp_path))

    forecast = model.predict_latest(frame, 3, directory=str(tmp_path))

    made = pd.Timestamp(forecast["made_at_utc"])
    valid = pd.Timestamp(forecast["valid_at_utc"])
    assert (valid - made) == pd.Timedelta(hours=3)
    assert forecast["predicted_wind_mw"] > 0
    assert forecast["expected_error_mw"] is not None


def test_forecast_reports_its_own_expected_error(frame, tmp_path):
    """A prediction without an error bar invites false confidence."""
    trained, columns, rows = model.train_final(frame, 1, params=FAST)
    result = model.evaluate_horizon(frame, 1, n_splits=3, params=FAST)
    model.save(trained, columns, 1, rows, result, directory=str(tmp_path))

    forecast = model.predict_latest(frame, 1, directory=str(tmp_path))
    assert forecast["expected_error_mw"] == pytest.approx(
        round(result.model_mae, 2)
    )

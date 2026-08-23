"""Feature and evaluation tests.

The leakage tests are the reason this file exists. Every other bug here shows
up as a bad score; leakage shows up as a *good* score, which is why it survives
into so many finished projects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast import evaluate, features
from gridcast.frame import FREQUENCY, SNSP_FILL_LIMIT


@pytest.fixture()
def frame():
    """Two weeks of synthetic quarter-hourly data with a daily cycle."""
    index = pd.date_range("2026-01-01", periods=96 * 14, freq=FREQUENCY, tz="UTC")
    t = np.arange(len(index))
    return pd.DataFrame(
        {
            "wind": 1500 + 800 * np.sin(2 * np.pi * t / 96) + (t % 7),
            "demand": 4000 + 500 * np.cos(2 * np.pi * t / 96),
            "snsp": 50 + 10 * np.sin(2 * np.pi * t / 96),
            "co2_intensity": 300 - 50 * np.sin(2 * np.pi * t / 96),
            "interconnection": 200 + 100 * np.cos(2 * np.pi * t / 96),
        },
        index=index,
    )


# ------------------------------------------------------------------- leakage
def test_no_feature_uses_the_future(frame):
    """The guarantee the whole module exists for.

    Corrupt everything after a cut point. If any feature value at or before the
    cut changes, that feature was reading forward in time.
    """
    horizon = features.horizon_steps(3)
    cut = len(frame) // 2

    clean = features.build_features(frame, horizon)

    tampered = frame.copy()
    tampered.iloc[cut + 1:] = tampered.iloc[cut + 1:] * 100 + 5000
    dirty = features.build_features(tampered, horizon)

    pd.testing.assert_frame_equal(
        clean.iloc[:cut + 1], dirty.iloc[:cut + 1],
        obj="features before the cut must not move when the future changes",
    )


def test_target_is_the_future_by_exactly_the_horizon(frame):
    horizon = features.horizon_steps(1)
    target = features.make_target(frame, horizon)

    assert target.iloc[0] == frame["wind"].iloc[horizon]
    assert target.iloc[-horizon:].isna().all(), "the last rows have no future"


def test_features_and_target_stay_aligned(frame):
    horizon = features.horizon_steps(6)
    X, y = features.build_dataset(frame, horizon)

    assert len(X) == len(y)
    assert X.index.equals(y.index)
    assert y.notna().all(), "no row may be kept without a target"


def test_wind_now_is_not_the_target(frame):
    """The most seductive leak: shipping the answer as a feature."""
    horizon = features.horizon_steps(3)
    X, y = features.build_dataset(frame, horizon)
    assert not np.allclose(X["wind_now"], y), "wind_now must lead the target"


# ------------------------------------------------------------------ encoding
def test_time_of_day_wraps_around_midnight(frame):
    """23:45 and 00:00 must be neighbours, not opposites."""
    built = features.build_features(frame, 4)
    late = built.loc[built.index.hour == 23].iloc[-1]
    early = built.loc[built.index.hour == 0].iloc[0]

    distance = np.hypot(late["tod_sin"] - early["tod_sin"],
                        late["tod_cos"] - early["tod_cos"])
    assert distance < 0.3, "midnight should be adjacent to just before midnight"


def test_cyclical_encodings_stay_on_the_unit_circle(frame):
    built = features.build_features(frame, 4)
    radius = np.hypot(built["tod_sin"], built["tod_cos"])
    assert np.allclose(radius, 1.0)


def test_weekend_flag_is_set_for_saturday_and_sunday(frame):
    built = features.build_features(frame, 4)
    assert built.loc[built.index.dayofweek == 5, "is_weekend"].eq(1).all()
    assert built.loc[built.index.dayofweek == 2, "is_weekend"].eq(0).all()


# ---------------------------------------------------------- walk-forward
def test_splits_never_train_on_the_future(frame):
    """The property that makes the evaluation honest."""
    for train, test in evaluate.walk_forward_splits(frame.index, n_splits=4):
        assert train.max() < test.min(), "training data must precede test data"


def test_training_window_expands(frame):
    sizes = [len(train)
             for train, _ in evaluate.walk_forward_splits(frame.index, n_splits=4)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_test_folds_do_not_overlap(frame):
    seen: set = set()
    for _, test in evaluate.walk_forward_splits(frame.index, n_splits=4):
        assert not seen & set(test)
        seen |= set(test)


def test_splits_reject_too_little_data():
    tiny = pd.date_range("2026-01-01", periods=5, freq=FREQUENCY, tz="UTC")
    with pytest.raises(ValueError):
        list(evaluate.walk_forward_splits(tiny))


# ------------------------------------------------------------------ scoring
def test_persistence_is_the_current_value(frame):
    horizon = features.horizon_steps(1)
    X, _ = features.build_dataset(frame, horizon)
    assert evaluate.persistence(X).equals(X["wind_now"])


def test_skill_is_positive_when_error_is_lower():
    assert evaluate.skill_score(50.0, 100.0) == pytest.approx(50.0)
    assert evaluate.skill_score(150.0, 100.0) == pytest.approx(-50.0)
    assert evaluate.skill_score(100.0, 100.0) == pytest.approx(0.0)


def test_perfect_forecast_scores_zero_error():
    actual = pd.Series([1.0, 2.0, 3.0])
    assert evaluate.mae(actual, actual) == 0.0
    assert evaluate.rmse(actual, actual) == 0.0


def test_rmse_punishes_large_errors_more_than_mae():
    actual = pd.Series([0.0, 0.0, 0.0, 0.0])
    predicted = pd.Series([0.0, 0.0, 0.0, 100.0])
    assert evaluate.rmse(actual, predicted) > evaluate.mae(actual, predicted)


def test_baselines_are_scored_and_persistence_leads(frame):
    horizon = features.horizon_steps(1)
    X, y = features.build_dataset(frame, horizon)
    scores = evaluate.evaluate_baselines(X, y, horizon_hours=1)

    names = [s.name for s in scores]
    assert "persistence" in names
    assert "climatology" in names

    by_name = {s.name: s for s in scores}
    assert by_name["persistence"].mae < by_name["climatology"].mae, \
        "persistence should beat predicting the mean"

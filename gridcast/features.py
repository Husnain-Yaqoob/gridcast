"""Features for forecasting wind generation h steps ahead.

The single rule this module exists to enforce: **a feature may only use
information that was available at the moment the forecast is made.**

That sounds obvious and is violated constantly. It is why a model can score
brilliantly in a notebook and be worthless in production — it was quietly shown
the answer. Every function here takes the horizon explicitly so the shift can
be checked rather than assumed, and there is a test that fails if any feature
correlates with the future in a way it should not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "wind"

# Lags in steps of the frame's frequency (15 minutes).
#   1 = 15 min, 4 = 1 h, 96 = 24 h, 672 = 7 days
LAG_STEPS: tuple[int, ...] = (1, 2, 4, 8, 12, 24, 48, 96, 672)
ROLLING_WINDOWS: tuple[int, ...] = (4, 12, 96)

STEPS_PER_HOUR = 4


def horizon_steps(hours: float) -> int:
    return int(round(hours * STEPS_PER_HOUR))


def make_target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """What we are predicting: wind output `horizon` steps from now.

    Shifting the target backwards, rather than shifting every feature forwards,
    keeps each row anchored to the time the forecast is *made*. That is the
    only framing in which "no feature may see the future" is checkable at a
    glance.
    """
    return frame[TARGET].shift(-horizon).rename(f"{TARGET}_t+{horizon}")


def build_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Predictors known at time t, for a forecast of t + horizon.

    Nothing here is shifted forward. Every column is a fact about the present
    or the past, which is what makes the leakage guarantee mechanical rather
    than a matter of care.
    """
    features = pd.DataFrame(index=frame.index)

    # --- the target's own history -----------------------------------------
    # Wind is strongly autocorrelated at short range; these carry most of the
    # signal, and the persistence baseline is simply lag 0.
    features[f"{TARGET}_now"] = frame[TARGET]
    for lag in LAG_STEPS:
        features[f"{TARGET}_lag{lag}"] = frame[TARGET].shift(lag)

    for window in ROLLING_WINDOWS:
        rolled = frame[TARGET].rolling(window, min_periods=max(2, window // 2))
        features[f"{TARGET}_mean{window}"] = rolled.mean()
        features[f"{TARGET}_std{window}"] = rolled.std()

    # --- ramp rates --------------------------------------------------------
    # How fast output is moving, and whether that movement is accelerating.
    # Ramps are what make wind hard to schedule around, so a model that ignores
    # them is blind to the cases anybody actually cares about.
    features[f"{TARGET}_ramp1"] = frame[TARGET].diff(1)
    features[f"{TARGET}_ramp4"] = frame[TARGET].diff(4)
    features[f"{TARGET}_ramp_accel"] = features[f"{TARGET}_ramp1"].diff(1)

    # --- other series, present values only ---------------------------------
    for column in ("demand", "snsp", "co2_intensity", "interconnection"):
        if column in frame.columns:
            features[column] = frame[column]
            features[f"{column}_lag4"] = frame[column].shift(4)

    if {"wind", "demand"}.issubset(frame.columns):
        features["wind_share"] = 100.0 * frame["wind"] / frame["demand"].replace(0, np.nan)

    # --- calendar ----------------------------------------------------------
    # Encoded as sine/cosine pairs rather than raw integers. Hour 23 and hour 0
    # are adjacent in reality; as plain numbers they are twenty-three apart,
    # and a linear model is told midnight is the opposite of 11pm.
    minutes = frame.index.hour * 60 + frame.index.minute
    features["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    features["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)

    day_of_year = frame.index.dayofyear
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    features["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)

    return features


def build_dataset(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned (X, y), with rows unusable for training removed.

    Rows are dropped only where the target is missing or the target's own
    history is absent. Gaps in the auxiliary series are left as NaN for the
    model to handle, because dropping a row whenever any of six series has a
    gap discards far more data than the gaps are worth.
    """
    features = build_features(frame, horizon)
    target = make_target(frame, horizon)

    essential = [f"{TARGET}_now", f"{TARGET}_lag4", f"{TARGET}_mean4"]
    usable = target.notna() & features[essential].notna().all(axis=1)

    return features.loc[usable], target.loc[usable]

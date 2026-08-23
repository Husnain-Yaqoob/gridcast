"""Baselines and honest evaluation.

Two ideas carry this module, and both exist to stop a model looking better
than it is.

**Walk-forward validation, never a random split.** Shuffling a time series and
holding out 20% at random lets the model learn from Thursday to predict
Wednesday. Scores rise, and the model is worthless. Every split here trains
strictly on the past and tests strictly on the future.

**A persistence baseline, reported at every horizon.** "Wind in an hour will be
what it is now" is astonishingly hard to beat at short range. A model reported
without it is a number with no meaning. Reporting the horizons where the model
*loses* is the difference between analysis and marketing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Score:
    name: str
    horizon_hours: float
    mae: float
    rmse: float
    n: int
    skill_vs_persistence: float | None = None

    def __str__(self) -> str:
        skill = ("      —" if self.skill_vs_persistence is None
                 else f"{self.skill_vs_persistence:+6.1f}%")
        return (f"{self.name:<24} {self.horizon_hours:>5.1f}h  "
                f"MAE {self.mae:>8.1f}  RMSE {self.rmse:>8.1f}  "
                f"skill {skill}  n={self.n:,}")


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def skill_score(model_error: float, baseline_error: float) -> float:
    """Percentage improvement over the baseline. Negative means worse.

    Reported as a percentage of the baseline's error rather than as raw MAE,
    because "48 MW" means nothing without knowing what the naive answer costs.
    """
    if baseline_error == 0:
        return 0.0
    return 100.0 * (baseline_error - model_error) / baseline_error


# --------------------------------------------------------------- baselines
def persistence(features: pd.DataFrame) -> pd.Series:
    """The forecast to beat: output will not change.

    Trivial, and at one hour ahead it is a genuinely strong forecast. Any model
    that cannot beat this is not earning its complexity.
    """
    return features["wind_now"]


def seasonal_naive(features: pd.DataFrame) -> pd.Series:
    """Yesterday at this time.

    A useful second reference: it tests whether daily shape carries information
    that recent level does not. For wind — unlike demand — it usually does not,
    and showing that is itself a finding.
    """
    return features["wind_lag96"]


def climatology(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    """The long-run mean. Deliberately terrible, included for scale.

    It anchors the other numbers: knowing the mean-only forecast's MAE tells a
    reader how much of the problem the interesting models actually solve.
    """
    return pd.Series(target.mean(), index=features.index)


# ------------------------------------------------------- walk-forward split
def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = 5,
                        min_train_fraction: float = 0.4):
    """Expanding-window splits: train on everything before, test on what follows.

    Expanding rather than sliding, because more history genuinely helps here
    and discarding it to keep the window fixed costs accuracy for no benefit.

    Yields (train_index, test_index) pairs in chronological order.
    """
    n = len(index)
    if n < 10:
        raise ValueError(f"need at least 10 rows to split, got {n}")

    start = int(n * min_train_fraction)
    fold_size = (n - start) // n_splits
    if fold_size < 1:
        raise ValueError(f"{n_splits} splits is too many for {n} rows")

    for fold in range(n_splits):
        train_end = start + fold * fold_size
        test_end = train_end + fold_size if fold < n_splits - 1 else n
        yield index[:train_end], index[train_end:test_end]


def evaluate_baselines(features: pd.DataFrame, target: pd.Series,
                       horizon_hours: float) -> list[Score]:
    """Score every baseline on the whole set.

    Baselines need no training, so there is nothing to hold out from — the
    walk-forward machinery is for models that learn.
    """
    scores: list[Score] = []

    candidates = {
        "persistence": persistence(features),
        "seasonal naive (24h)": seasonal_naive(features),
        "climatology": climatology(features, target),
    }

    baseline_mae = None
    for name, predicted in candidates.items():
        usable = predicted.notna() & target.notna()
        if not usable.any():
            continue
        actual_u, predicted_u = target[usable], predicted[usable]
        this_mae = mae(actual_u, predicted_u)

        if name == "persistence":
            baseline_mae = this_mae

        scores.append(Score(
            name=name,
            horizon_hours=horizon_hours,
            mae=this_mae,
            rmse=rmse(actual_u, predicted_u),
            n=int(usable.sum()),
            skill_vs_persistence=(
                None if baseline_mae is None or name == "persistence"
                else skill_score(this_mae, baseline_mae)
            ),
        ))

    return scores

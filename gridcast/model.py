"""Training and evaluating a wind forecasting model.

One model is trained per horizon, not one model that takes horizon as an
input. Predicting one hour ahead and predicting twelve hours ahead are
genuinely different problems: at one hour, current output dominates and almost
nothing else matters; at twelve, current output is nearly irrelevant and the
model has to lean on season, time of day and recent variability. A single
model forced to serve both learns a compromise that is good at neither.

Every result is produced by walk-forward validation against a persistence
baseline scored on exactly the same rows. A model evaluated on a different
sample than its baseline is not being compared to anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import evaluate, features

DEFAULT_MODEL_DIR = "models"

# Deliberately modest. The dataset is ~35,000 rows with strong autocorrelation,
# which is a setting where a large ensemble memorises recent noise and looks
# excellent in training while adding nothing out of sample.
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_leaf": 50,
    "subsample": 0.8,
    "random_state": 42,
}


@dataclass
class FoldResult:
    fold: int
    train_rows: int
    test_rows: int
    test_start: str
    test_end: str
    model_mae: float
    model_rmse: float
    baseline_mae: float
    baseline_rmse: float

    @property
    def skill(self) -> float:
        return evaluate.skill_score(self.model_mae, self.baseline_mae)


@dataclass
class HorizonResult:
    horizon_hours: float
    n_features: int
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def model_mae(self) -> float:
        """Rows-weighted, not a mean of means.

        Folds differ in size — the first test window is smaller than the last —
        so averaging the per-fold MAEs would quietly overweight the smallest
        fold. Weighting by rows gives the error an average prediction actually
        carries.
        """
        rows = sum(f.test_rows for f in self.folds)
        return sum(f.model_mae * f.test_rows for f in self.folds) / rows

    @property
    def baseline_mae(self) -> float:
        rows = sum(f.test_rows for f in self.folds)
        return sum(f.baseline_mae * f.test_rows for f in self.folds) / rows

    @property
    def model_rmse(self) -> float:
        rows = sum(f.test_rows for f in self.folds)
        return sum(f.model_rmse * f.test_rows for f in self.folds) / rows

    @property
    def skill(self) -> float:
        return evaluate.skill_score(self.model_mae, self.baseline_mae)

    @property
    def beats_baseline(self) -> bool:
        return self.model_mae < self.baseline_mae

    @property
    def consistent(self) -> bool:
        """Did it win in every fold, or only on average?

        A model that wins overall but loses in two folds out of five is not
        reliably better — it got lucky on a stretch of weather. That is worth
        knowing before anyone quotes the headline number.
        """
        return all(f.model_mae < f.baseline_mae for f in self.folds)

    def summary(self) -> str:
        verdict = "beats" if self.beats_baseline else "LOSES TO"
        stability = "" if self.consistent else "  [not in every fold]"
        return (
            f"{self.horizon_hours:>5.1f}h  model MAE {self.model_mae:>7.1f}  "
            f"persistence {self.baseline_mae:>7.1f}  "
            f"{verdict} by {abs(self.skill):>5.1f}%{stability}"
        )


def _make_regressor(params: dict | None = None):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'scikit-learn is required. Install it with:\n'
            '    pip install -e ".[analysis]"'
        ) from exc

    settings = {**DEFAULT_PARAMS, **(params or {})}
    # HistGradientBoosting handles NaN natively, which matters here: the
    # auxiliary series have real gaps, and imputing them would invent readings
    # the grid never published. Letting the model learn "this was missing" is
    # both more honest and more accurate.
    return HistGradientBoostingRegressor(
        max_iter=settings["n_estimators"],
        learning_rate=settings["learning_rate"],
        max_depth=settings["max_depth"],
        min_samples_leaf=settings["min_samples_leaf"],
        random_state=settings["random_state"],
        early_stopping=False,
    )


def evaluate_horizon(frame: pd.DataFrame, horizon_hours: float,
                     n_splits: int = 5, params: dict | None = None) -> HorizonResult:
    """Walk-forward evaluation of one horizon against persistence."""
    steps = features.horizon_steps(horizon_hours)
    X, y = features.build_dataset(frame, steps)

    result = HorizonResult(horizon_hours=horizon_hours, n_features=X.shape[1])

    for fold, (train_index, test_index) in enumerate(
        evaluate.walk_forward_splits(X.index, n_splits=n_splits), start=1
    ):
        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_test, y_test = X.loc[test_index], y.loc[test_index]

        model = _make_regressor(params)
        model.fit(X_train, y_train)
        predicted = pd.Series(model.predict(X_test), index=X_test.index)

        # The baseline is scored on exactly these rows. Comparing a model on
        # one sample to a baseline on another is not a comparison.
        baseline = evaluate.persistence(X_test)

        result.folds.append(FoldResult(
            fold=fold,
            train_rows=len(X_train),
            test_rows=len(X_test),
            test_start=str(X_test.index.min()),
            test_end=str(X_test.index.max()),
            model_mae=evaluate.mae(y_test, predicted),
            model_rmse=evaluate.rmse(y_test, predicted),
            baseline_mae=evaluate.mae(y_test, baseline),
            baseline_rmse=evaluate.rmse(y_test, baseline),
        ))

    return result


def train_final(frame: pd.DataFrame, horizon_hours: float,
                params: dict | None = None):
    """Fit on all available history, for serving rather than for scoring.

    Trained on everything, so it has no honest test set — which is exactly why
    it is never the thing that produces a reported number. Evaluation comes
    from `evaluate_horizon`; this is only what the API loads.
    """
    steps = features.horizon_steps(horizon_hours)
    X, y = features.build_dataset(frame, steps)
    model = _make_regressor(params)
    model.fit(X, y)
    return model, list(X.columns), len(X)


def feature_importance(frame: pd.DataFrame, horizon_hours: float,
                       n_repeats: int = 3, sample: int = 4000,
                       params: dict | None = None) -> pd.Series:
    """Permutation importance on a held-out tail.

    Permutation rather than the tree's own split counts. Split-based importance
    inflates high-cardinality continuous features simply because there are more
    places to cut them, which here would flatter the lag columns. Permutation
    asks the only question that matters: how much worse does it get if this
    column is shuffled?
    """
    from sklearn.inspection import permutation_importance

    steps = features.horizon_steps(horizon_hours)
    X, y = features.build_dataset(frame, steps)

    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    if len(X_test) > sample:
        X_test, y_test = X_test.iloc[-sample:], y_test.iloc[-sample:]

    model = _make_regressor(params)
    model.fit(X_train, y_train)

    scores = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        random_state=42, scoring="neg_mean_absolute_error",
    )
    return pd.Series(
        scores.importances_mean, index=X.columns
    ).sort_values(ascending=False)


# ----------------------------------------------------------------- persistence
def save(model, columns: list[str], horizon_hours: float, rows: int,
         result: HorizonResult | None, directory: str = DEFAULT_MODEL_DIR) -> str:
    """Write the model and a manifest describing what it is.

    The manifest matters as much as the model file. A .joblib on disk with no
    record of which columns it expects, which horizon it serves or how well it
    scored is an object nobody can safely use six months later.
    """
    import joblib

    os.makedirs(directory, exist_ok=True)
    stem = f"wind_h{horizon_hours:g}"
    model_path = os.path.join(directory, f"{stem}.joblib")
    joblib.dump(model, model_path)

    manifest = {
        "horizon_hours": horizon_hours,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_rows": rows,
        "features": columns,
        "target": "wind generation (MW) at t + horizon",
        "units": "MW",
        "attribution": "Supported by EirGrid Group Data",
    }
    if result is not None:
        manifest["validation"] = {
            "method": f"walk-forward, {len(result.folds)} expanding folds",
            "model_mae": round(result.model_mae, 2),
            "model_rmse": round(result.model_rmse, 2),
            "persistence_mae": round(result.baseline_mae, 2),
            "skill_vs_persistence_pct": round(result.skill, 2),
            "beat_baseline_in_every_fold": result.consistent,
            "folds": [asdict(f) for f in result.folds],
        }

    with open(os.path.join(directory, f"{stem}.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return model_path


def load(horizon_hours: float, directory: str = DEFAULT_MODEL_DIR):
    """Load a saved model and its manifest."""
    import joblib

    stem = f"wind_h{horizon_hours:g}"
    model_path = os.path.join(directory, f"{stem}.joblib")
    manifest_path = os.path.join(directory, f"{stem}.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"no model for horizon {horizon_hours}h in {directory}/. "
            f"Train one with:  python -m gridcast train --horizons {horizon_hours:g}"
        )

    model = joblib.load(model_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return model, manifest


def predict_latest(frame: pd.DataFrame, horizon_hours: float,
                   directory: str = DEFAULT_MODEL_DIR) -> dict:
    """Forecast from the most recent complete row in the frame."""
    model, manifest = load(horizon_hours, directory)

    steps = features.horizon_steps(horizon_hours)
    built = features.build_features(frame, steps)

    essential = ["wind_now", "wind_lag4", "wind_mean4"]
    usable = built[built[essential].notna().all(axis=1)]
    if usable.empty:
        raise ValueError("no row has enough history to forecast from")

    row = usable.iloc[[-1]][manifest["features"]]
    predicted = float(model.predict(row)[0])
    made_at = usable.index[-1]

    return {
        "made_at_utc": made_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_at_utc": (
            made_at + pd.Timedelta(hours=horizon_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": horizon_hours,
        "predicted_wind_mw": round(predicted, 1),
        "current_wind_mw": round(float(row["wind_now"].iloc[0]), 1),
        "expected_error_mw": manifest.get("validation", {}).get("model_mae"),
        "model_trained_at": manifest["trained_at"],
        "attribution": manifest["attribution"],
    }

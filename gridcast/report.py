"""Charts.

Everything else in this project reports its findings as numbers. A mean
absolute error of 105.3 MW is the honest result, and it is also invisible to
anyone who does not already know what a mean absolute error is. These charts
exist so the findings can be *seen*, and they are held to the same standard as
the numbers: nothing is drawn that the data does not support, and the horizons
where the model loses are drawn as prominently as the one where it wins.

Two decisions here are worth reading before changing anything.

**Forecasts are plotted against the time they are for, not the time they were
made.** A twelve-hour forecast made at noon is a statement about midnight, and
drawing it at noon would put it half a day to the left of the reality it is
predicting. Every comparison line here is shifted onto its valid time, which is
the only alignment in which "the model tracks" or "the model drifts" means
anything.

**The backtest model never sees the period it is drawn on.** `backtest()` fits
on the earlier part of the series and predicts the later part. Plotting a model
against data it was trained on produces a chart that looks superb and proves
nothing, and that chart is one of the most common ways portfolio projects
quietly mislead.

Hour-of-day analysis is done in Irish local time rather than UTC. Carbon
intensity peaks because of when people cook dinner, and people cook dinner at
a local clock time — through the summer, a UTC hour label would be an hour off
the human behaviour it is describing.
"""

from __future__ import annotations

import json
import os
from glob import glob

import numpy as np
import pandas as pd

from .config import ATTRIBUTION, GRID_TIMEZONE
from . import evaluate, features

DEFAULT_OUTPUT_DIR = "docs"

# Palette. Taken from a validated categorical set rather than chosen by eye:
# the three series colours clear colour-vision-deficiency separation against
# each other and contrast against the surface. The lines are also direct
# labelled, so identity never rests on colour alone.
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

ACTUAL = "#2a78d6"        # blue
MODEL = "#eb6834"         # orange
PERSISTENCE = "#1baf7a"   # aqua
NEGATIVE = "#e34948"      # red, for skill below zero

FIGSIZE = (10, 5)
DPI = 150


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")          # no display on a server or in CI
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required to draw charts. Install it with:\n"
            '    pip install -e ".[analysis]"'
        ) from exc


def _style(ax, title: str, subtitle: str = "", ylabel: str = "") -> None:
    """Recessive chrome, so the data is the loudest thing in the frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="#52514e", fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold",
                     loc="left", pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                color="#52514e", fontsize=10, va="bottom")


def _finish(fig, path: str) -> str:
    # Below the figure, not inside it. At (0.01, 0.01) it collided with
    # multi-line tick labels; a negative y lets the tight bounding box grow
    # to include it rather than letting the two overlap.
    fig.text(0.01, -0.045, ATTRIBUTION, color=INK_MUTED, fontsize=8)
    fig.patch.set_facecolor(SURFACE)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# --------------------------------------------------------------- backtesting
def backtest(frame: pd.DataFrame, horizon_hours: float,
             train_fraction: float = 0.8, params: dict | None = None,
             days: int | None = None) -> pd.DataFrame:
    """Predictions on data the model was not trained on, indexed by valid time.

    Returns one row per forecast with three columns — what actually happened,
    what the model said would happen, and what persistence said would happen —
    all aligned on the moment being predicted.

    The split is chronological, never random. Shuffling a time series before
    splitting lets the model train on Thursday and test on Wednesday, which on
    a quantity this autocorrelated is close to handing it the answer.
    """
    from . import model as model_module

    steps = features.horizon_steps(horizon_hours)
    X, y = features.build_dataset(frame, steps)
    if len(X) < 200:
        raise ValueError(
            f"only {len(X)} usable rows at {horizon_hours}h — not enough to "
            f"backtest. Collect more history first."
        )

    split = int(len(X) * train_fraction)
    regressor = model_module._make_regressor(params)
    regressor.fit(X.iloc[:split], y.iloc[:split])

    X_test, y_test = X.iloc[split:], y.iloc[split:]
    predicted = regressor.predict(X_test)

    # Shift onto valid time. Up to here every row is stamped with the moment
    # the forecast was made; the chart needs the moment it is about.
    valid_at = X_test.index + pd.Timedelta(hours=horizon_hours)

    result = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "model": predicted,
            # Persistence is simply "output now", carried forward. It is the
            # thing to beat, so it is drawn rather than described.
            "persistence": X_test["wind_now"].to_numpy(),
        },
        index=valid_at,
    )
    result.index.name = "valid_at_utc"

    if days is not None:
        cutoff = result.index.max() - pd.Timedelta(days=days)
        result = result[result.index >= cutoff]
    return result


# -------------------------------------------------------------------- charts
def wind_share_chart(frame: pd.DataFrame,
                     path: str = f"{DEFAULT_OUTPUT_DIR}/wind_share.png") -> str:
    """How much of Irish demand wind actually covered, day by day.

    Drawn as a daily mean with the day's range behind it. The mean alone would
    hide the fact that a 40%-average day still contains hours near zero and
    hours above 100 — and that spread is the whole operational problem with
    wind, so a chart that smooths it away is telling a comfortable lie.
    """
    plt = _require_matplotlib()

    share = (100.0 * frame["wind"] / frame["demand"].replace(0, np.nan)).dropna()
    if share.empty:
        raise ValueError("no rows with both wind and demand — nothing to plot")

    daily = share.resample("D")
    mean, low, high = daily.mean(), daily.min(), daily.max()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(mean.index, low, high, color=ACTUAL, alpha=0.15, linewidth=0)
    ax.plot(mean.index, mean, color=ACTUAL, linewidth=2)

    # 100% is a real threshold, not a decoration: above it, Ireland is
    # generating more wind than it is consuming and exporting the surplus.
    if float(high.max()) > 100:
        ax.axhline(100, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(mean.index[-1], 102, "wind exceeds total demand  ",
                color="#52514e", fontsize=9, ha="right")

    hours_above = int((share > 100).sum())
    _style(
        ax,
        "Wind as a share of Irish electricity demand",
        f"Daily mean, with each day's full range behind it. "
        f"{hours_above:,} quarter-hours above 100% across the period.",
        "% of demand",
    )
    ax.set_ylim(bottom=0)
    return _finish(fig, path)


def carbon_by_hour_chart(frame: pd.DataFrame,
                         path: str = f"{DEFAULT_OUTPUT_DIR}/carbon_by_hour.png") -> str:
    """When the grid is dirtiest, by local hour of day.

    The band is the interquartile range, not the full range. Carbon intensity
    has genuine outliers — a still, cold evening with the interconnectors
    importing coal — and letting those set the band would make every hour look
    equally uncertain when they are not.
    """
    plt = _require_matplotlib()

    if "co2_intensity" not in frame.columns:
        raise ValueError("no co2_intensity column — nothing to plot")

    series = frame["co2_intensity"].dropna()
    if series.empty:
        raise ValueError("no carbon intensity readings held")

    local_hour = series.index.tz_convert(GRID_TIMEZONE).hour
    grouped = series.groupby(local_hour)
    median = grouped.median()
    q1, q3 = grouped.quantile(0.25), grouped.quantile(0.75)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(median.index, q1, q3, color=ACTUAL, alpha=0.15, linewidth=0)
    ax.plot(median.index, median, color=ACTUAL, linewidth=2, marker="o",
            markersize=4)

    dirtiest, cleanest = int(median.idxmax()), int(median.idxmin())
    for hour, label, offset in ((dirtiest, "dirtiest", 14), (cleanest, "cleanest", -26)):
        # Annotations near either end of the axis are anchored inwards, or the
        # text runs off the edge of the figure.
        align = "left" if hour <= 4 else "right" if hour >= 19 else "center"
        ax.annotate(
            f"{label}  {hour:02d}:00\n{median[hour]:.0f} gCO₂/kWh",
            xy=(hour, median[hour]), xytext=(0, offset),
            textcoords="offset points", ha=align, fontsize=9,
            color=INK, fontweight="bold",
        )

    _style(
        ax,
        "Carbon intensity of Irish electricity, by hour of day",
        "Median with interquartile range. Hours are Irish local time — the "
        "evening peak is a human habit, not a UTC one.",
        "gCO₂ per kWh",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day (Europe/Dublin)", color="#52514e", fontsize=10)
    return _finish(fig, path)


def forecast_vs_actual_chart(
    frame: pd.DataFrame,
    path: str = f"{DEFAULT_OUTPUT_DIR}/forecast_vs_actual.png",
    horizons: tuple[float, ...] = (1.0, 12.0),
    days: int = 4,
    params: dict | None = None,
) -> str:
    """The finding, drawn: short horizons track, long horizons drift.

    Two panels sharing one y-axis, because the comparison between them is the
    point. A separate scale per panel would let the twelve-hour errors look the
    same size as the one-hour errors, which is the opposite of what happened.
    """
    plt = _require_matplotlib()

    panels = [(h, backtest(frame, h, days=days, params=params)) for h in horizons]

    fig, axes = plt.subplots(len(panels), 1, figsize=(FIGSIZE[0], 3.4 * len(panels)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    # Each panel carries a title and a subtitle, which need room. The
    # default spacing puts the lower panel's title on the upper panel's
    # bottom gridline.
    fig.subplots_adjust(hspace=0.42)

    for ax, (horizon, data) in zip(axes, panels):
        model_mae = float(np.abs(data["actual"] - data["model"]).mean())
        naive_mae = float(np.abs(data["actual"] - data["persistence"]).mean())
        skill = evaluate.skill_score(model_mae, naive_mae)

        ax.plot(data.index, data["actual"], color=ACTUAL, linewidth=2.2,
                label="actual", zorder=3)
        ax.plot(data.index, data["model"], color=MODEL, linewidth=1.8,
                label="model forecast", zorder=2)
        ax.plot(data.index, data["persistence"], color=PERSISTENCE, linewidth=1.4,
                linestyle="--", label="persistence baseline", zorder=1)

        verdict = "beats persistence" if skill > 0 else "loses to persistence"
        unit = "hour" if horizon == 1 else "hours"
        _style(
            ax,
            f"{horizon:g} {unit} ahead",
            f"MAE {model_mae:.0f} MW vs persistence {naive_mae:.0f} MW — "
            f"{verdict} by {abs(skill):.1f}%",
            "wind generation (MW)",
        )

    # One legend for the whole figure, below the panels.
    #
    # Per-axes legends were the first version and they collided with the next
    # panel's title; right-hand direct labels were the second and they collided
    # with each other, because at the right edge all three lines are within a
    # few tens of MW of one another. A figure-level legend is the only
    # placement that cannot overlap the data.
    #
    # Persistence is also dashed, so the three series are separable without
    # relying on colour at all.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=10,
               labelcolor=INK, loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.autofmt_xdate()
    fig.suptitle(
        f"Forecast against what actually happened — last {days} days of a "
        f"held-out test period",
        color=INK, fontsize=14, fontweight="bold", x=0.005, ha="left", y=1.02,
    )
    return _finish(fig, path)


def skill_by_horizon_chart(model_dir: str = "models",
                           path: str = f"{DEFAULT_OUTPUT_DIR}/skill_by_horizon.png") -> str:
    """Where the model stops being worth its complexity.

    A diverging bar around zero, because the quantity has a sign that means
    something: above the line the model earns its place, below it a one-line
    baseline would have done better. Losses are drawn in red at full size —
    the crossover is the most interesting thing this project found, and
    cropping it would be dishonest.
    """
    plt = _require_matplotlib()

    entries = []
    for manifest_path in sorted(glob(os.path.join(model_dir, "wind_h*.json"))):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        validation = manifest.get("validation")
        if not validation:
            continue
        entries.append((
            float(manifest["horizon_hours"]),
            float(validation["skill_vs_persistence_pct"]),
            float(validation["model_mae"]),
            float(validation["persistence_mae"]),
            bool(validation.get("beat_baseline_in_every_fold")),
        ))

    if not entries:
        raise ValueError(
            f"no validated model manifests in {model_dir}/. "
            f"Train some first:  python -m gridcast train --save"
        )

    entries.sort()
    skills = [s for _, s, *_ in entries]
    colours = [ACTUAL if s > 0 else NEGATIVE for s in skills]

    # The supporting numbers go under the tick labels rather than floating near
    # the zero line. Anything placed near zero collides with whichever bar
    # starts there — which is every bar.
    labels = [
        f"{horizon:g}h ahead\n{model_mae:.0f} vs {naive_mae:.0f} MW"
        + ("\nwins every fold" if consistent and skill > 0 else "")
        for horizon, skill, model_mae, naive_mae, consistent in entries
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, skills, color=colours, width=0.5, zorder=3)
    ax.axhline(0, color="#52514e", linewidth=1.2, zorder=4)

    for bar, skill in zip(bars, skills):
        above = skill > 0
        ax.annotate(
            f"{skill:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, skill),
            xytext=(0, 8 if above else -8), textcoords="offset points",
            ha="center", va="bottom" if above else "top",
            fontsize=12, fontweight="bold", color=INK,
        )

    _style(
        ax,
        "Does the model beat doing nothing clever?",
        "Skill against a persistence baseline, by forecast horizon. "
        "Above the line the model earns its complexity; below it, it does not.",
        "skill vs persistence (%)",
    )
    ax.tick_params(axis="x", labelsize=10, colors="#52514e", length=0)
    span = max(abs(min(skills)), abs(max(skills))) * 1.35 or 1
    ax.set_ylim(-span, span)
    return _finish(fig, path)


# --------------------------------------------------------------------- driver
def render_all(frame: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR,
               model_dir: str = "models", days: int = 4,
               params: dict | None = None) -> list[str]:
    """Draw everything that can be drawn from what is held.

    A chart that cannot be drawn — no carbon readings, no trained models — is
    reported and skipped rather than raised. Losing four good charts because
    the fifth had no data would be a poor trade, and the caller is told exactly
    what was missing.
    """
    written: list[str] = []
    jobs = [
        ("wind share", lambda: wind_share_chart(
            frame, os.path.join(output_dir, "wind_share.png"))),
        ("carbon by hour", lambda: carbon_by_hour_chart(
            frame, os.path.join(output_dir, "carbon_by_hour.png"))),
        ("forecast vs actual", lambda: forecast_vs_actual_chart(
            frame, os.path.join(output_dir, "forecast_vs_actual.png"),
            days=days, params=params)),
        ("skill by horizon", lambda: skill_by_horizon_chart(
            model_dir, os.path.join(output_dir, "skill_by_horizon.png"))),
    ]

    for name, job in jobs:
        try:
            written.append(job())
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"  skipped {name}: {exc}")

    return written

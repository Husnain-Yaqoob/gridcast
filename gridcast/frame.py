"""Turn stored readings into a regular, modellable time series.

This is the boundary between "data we collected" and "data we can learn from",
and three decisions are made here that everything downstream inherits.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from .config import DEFAULT_DB_PATH, DEFAULT_REGION

# The grid publishes most series every 15 minutes. That is the frame's rate.
FREQUENCY = "15min"

# SNSP is published half-hourly, so on a quarter-hourly index every other slot
# is empty. It is forward-filled by exactly one step — never more.
#
# One step says "the SNSP measured at 10:00 still applied at 10:15", which is
# true of a quantity that moves slowly. An unlimited fill would say the last
# reading before a three-day outage applied for three days, which is invention
# dressed up as data.
SNSP_FILL_LIMIT = 1


def load_wide(db_path: str = DEFAULT_DB_PATH,
              region: str = DEFAULT_REGION) -> pd.DataFrame:
    """One row per timestamp, one column per series, on a regular index.

    Filtering by region matters more than it looks. Most series are published
    only for the all-island system, but `interconnection` returns separate ROI,
    NI and ALL rows whatever region is requested — so an unfiltered pivot
    silently triples those timestamps and quietly corrupts every join.
    """
    connection = sqlite3.connect(db_path)
    try:
        raw = pd.read_sql_query(
            """
            SELECT area, ts_utc, value
              FROM reading
             WHERE region = ?
             ORDER BY ts_utc
            """,
            connection,
            params=(region,),
        )
    finally:
        connection.close()

    if raw.empty:
        raise ValueError(
            f"no readings for region {region!r} in {db_path}. "
            f"Run 'python -m gridcast backfill' first."
        )

    raw["ts_utc"] = pd.to_datetime(raw["ts_utc"], utc=True, format="ISO8601")
    wide = raw.pivot_table(index="ts_utc", columns="area", values="value",
                           aggfunc="last")
    wide.columns.name = None

    # Reindex onto a complete regular grid. A missing timestamp and a present
    # timestamp holding NaN are the same fact — "not measured" — and lag
    # features are only meaningful when a row's position implies its time.
    full_index = pd.date_range(wide.index.min(), wide.index.max(),
                               freq=FREQUENCY, tz="UTC")
    wide = wide.reindex(full_index)
    wide.index.name = "ts_utc"

    if "snsp" in wide.columns:
        wide["snsp"] = wide["snsp"].ffill(limit=SNSP_FILL_LIMIT)

    return wide


def coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    """How much of each column is actually present. Read this before modelling."""
    total = len(frame)
    return pd.DataFrame({
        "present": frame.notna().sum(),
        "missing": frame.isna().sum(),
        "pct_present": (100 * frame.notna().sum() / total).round(2),
        "first": frame.apply(lambda s: s.first_valid_index()),
        "last": frame.apply(lambda s: s.last_valid_index()),
    })


def wind_share(frame: pd.DataFrame) -> pd.Series:
    """Wind generation as a percentage of demand.

    Can exceed 100: at times Ireland generates more wind than it consumes and
    exports the surplus. Capping it at 100 would hide the most interesting
    hours on the grid, so it is left alone.
    """
    return 100.0 * frame["wind"] / frame["demand"].replace(0, pd.NA)

"""Command line for the pipeline.

    python -m gridcast probe                 is the service answering?
    python -m gridcast backfill --days 365   load a year of history
    python -m gridcast update                fetch whatever is new
    python -m gridcast status                what is held, and did the last run work?
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from . import ingest
from .client import GridClient
from .config import (ATTRIBUTION, AREAS, DEFAULT_DB_PATH, DEFAULT_REGION,
                     MAX_DAYS_PER_REQUEST, REQUEST_INTERVAL_SECONDS, REGIONS)
from .store import Store


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_probe(args) -> int:
    client = GridClient()
    ok, detail = client.probe()
    if ok:
        print(f"Dashboard is answering — {detail}")
        return 0
    print(f"Dashboard did not answer usefully: {detail}", file=sys.stderr)
    print(
        "\nThis endpoint returns 503 when busy. Wait a minute and try again "
        "before assuming anything is broken.",
        file=sys.stderr,
    )
    return 1


def cmd_backfill(args) -> int:
    store = Store(args.db)
    client = GridClient()
    end = date.today()
    start = end - timedelta(days=args.days)

    print(f"Backfilling {start} to {end} for region {args.region}")
    print(f"{len(AREAS)} series in {MAX_DAYS_PER_REQUEST}-day windows, "
          f"{REQUEST_INTERVAL_SECONDS:.0f}s apart.")
    print("Each window is saved as it arrives, and windows already held are "
          "skipped,\nso this is safe to stop and re-run until it completes.\n")

    try:
        result = ingest.backfill(store, client, start, end, region=args.region,
                                 resume=not args.fresh)
    except ingest.Cancelled:
        print(f"\nStopped. {store.count():,} readings are saved — re-run "
              f"'backfill' to continue.", file=sys.stderr)
        return 130      # conventional exit code for SIGINT

    print(f"\n{result.summary}")
    print(f"Database now holds {store.count():,} readings.")
    return 0


def cmd_update(args) -> int:
    store = Store(args.db)
    client = GridClient()
    try:
        result = ingest.update(store, client, region=args.region)
    except ingest.Cancelled:
        print(f"\nStopped. {store.count():,} readings are saved.", file=sys.stderr)
        return 130
    print(result.summary)
    return 0


def cmd_baseline(args) -> int:
    """Score the naive forecasts on the collected data.

    Deliberately the first modelling command. Before any model is built, these
    numbers say what the problem is worth solving and how much of it is already
    solved by doing nothing clever.
    """
    try:
        from .frame import coverage_report, load_wide, wind_share
        from . import evaluate, features
    except ImportError:
        print("This needs the analysis extras:\n"
              '    pip install -e ".[analysis]"', file=sys.stderr)
        return 1

    frame = load_wide(args.db, region=args.region)

    print(f"{len(frame):,} rows, {frame.index.min():%Y-%m-%d} to "
          f"{frame.index.max():%Y-%m-%d}\n")
    print(coverage_report(frame).to_string())

    share = wind_share(frame)
    print(f"\nWind as a share of demand: mean {share.mean():.1f}%, "
          f"max {share.max():.1f}%")
    if "co2_intensity" in frame:
        by_hour = frame.groupby(frame.index.hour)["co2_intensity"].mean()
        print(f"Carbon intensity: dirtiest hour {by_hour.idxmax():02d}:00 "
              f"({by_hour.max():.0f} gCO2/kWh), "
              f"cleanest {by_hour.idxmin():02d}:00 ({by_hour.min():.0f})")

    print("\nBaseline forecasts, by horizon")
    print("=" * 78)
    for hours in args.horizons:
        steps = features.horizon_steps(hours)
        X, y = features.build_dataset(frame, steps)
        print(f"\n{hours}h ahead — {len(X):,} usable rows")
        for score in evaluate.evaluate_baselines(X, y, hours):
            print(f"  {score}")

    print("\nSkill is measured against persistence. A model that cannot beat")
    print("these numbers is not earning its complexity.")
    print(f"\n{ATTRIBUTION}")
    return 0


def cmd_train(args) -> int:
    """Train and evaluate a model per horizon, against persistence."""
    try:
        from .frame import load_wide
        from . import model as model_module
    except ImportError:
        print('This needs the analysis extras:\n    pip install -e ".[analysis]"',
              file=sys.stderr)
        return 1

    frame = load_wide(args.db, region=args.region)
    print(f"{len(frame):,} rows, {frame.index.min():%Y-%m-%d} to "
          f"{frame.index.max():%Y-%m-%d}")
    print(f"Walk-forward validation, {args.folds} expanding folds. "
          f"Training on the past, testing on the future, every time.\n")

    results = []
    for hours in args.horizons:
        print(f"--- {hours}h ahead")
        result = model_module.evaluate_horizon(frame, hours, n_splits=args.folds)
        results.append(result)

        for fold in result.folds:
            arrow = "+" if fold.skill > 0 else " "
            print(f"    fold {fold.fold}  train {fold.train_rows:>6,}  "
                  f"test {fold.test_rows:>5,}  "
                  f"MAE {fold.model_mae:>7.1f} vs {fold.baseline_mae:>7.1f}  "
                  f"{arrow}{fold.skill:>5.1f}%")

        if args.save:
            trained, columns, rows = model_module.train_final(frame, hours)
            path = model_module.save(trained, columns, hours, rows, result,
                                     directory=args.model_dir)
            print(f"    saved {path}")
        print()

    print("Summary")
    print("=" * 78)
    for result in results:
        print(f"  {result.summary()}")

    winners = [r for r in results if r.beats_baseline]
    print(f"\nBeats persistence at {len(winners)} of {len(results)} horizons.")
    if winners:
        best = max(winners, key=lambda r: r.skill)
        print(f"Best gain: {best.skill:.1f}% at {best.horizon_hours:g}h.")
    losers = [r for r in results if not r.beats_baseline]
    for result in losers:
        print(f"Loses to persistence at {result.horizon_hours:g}h — reported "
              f"because hiding it would make the rest meaningless.")

    print(f"\n{ATTRIBUTION}")
    return 0


def cmd_importance(args) -> int:
    """Which features the model actually relies on."""
    try:
        from .frame import load_wide
        from . import model as model_module
    except ImportError:
        print('This needs the analysis extras:\n    pip install -e ".[analysis]"',
              file=sys.stderr)
        return 1

    frame = load_wide(args.db, region=args.region)
    print(f"Permutation importance at {args.horizon}h "
          f"(MAE increase when a column is shuffled)\n")

    scores = model_module.feature_importance(frame, args.horizon)
    for name, value in scores.head(args.top).items():
        bar = "#" * int(max(0, value) / max(scores.max(), 1e-9) * 40)
        print(f"  {name:<24} {value:>8.2f}  {bar}")

    print(f"\n{ATTRIBUTION}")
    return 0


def cmd_forecast(args) -> int:
    """Predict from the most recent data held."""
    try:
        from .frame import load_wide
        from . import model as model_module
    except ImportError:
        print('This needs the analysis extras:\n    pip install -e ".[analysis]"',
              file=sys.stderr)
        return 1

    frame = load_wide(args.db, region=args.region)
    try:
        forecast = model_module.predict_latest(frame, args.horizon,
                                               directory=args.model_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Forecast made at    {forecast['made_at_utc']}")
    print(f"Valid at            {forecast['valid_at_utc']}")
    print(f"Current wind        {forecast['current_wind_mw']:,.1f} MW")
    print(f"Predicted wind      {forecast['predicted_wind_mw']:,.1f} MW")
    if forecast["expected_error_mw"]:
        print(f"Typical error       +/- {forecast['expected_error_mw']:,.1f} MW "
              f"(validated MAE)")
    print(f"\n{forecast['attribution']}")
    return 0


def _print_runs(store: Store, limit: int = 5) -> None:
    runs = store.recent_runs(limit)
    if not runs:
        return
    print("Recent runs:")
    for run in runs:
        state = run["status"] or "running"
        print(f"  #{run['id']:<4} {run['started_at'][:16]}  {run['command']:<28} "
              f"{state:<8} {run['rows_written'] or 0:>7,} rows")
        if run["detail"]:
            print(f"        {run['detail'][:100]}")


def cmd_status(args) -> int:
    store = Store(args.db)
    rows = store.coverage()

    if not rows:
        print("No readings held yet.\n")
        # Show the run log anyway. An empty database with three failed runs
        # behind it is a completely different situation from an empty database
        # that has never been used, and "run backfill" is unhelpful advice for
        # the first one.
        _print_runs(store)
        print("\nIf there are no runs above, start with 'probe' then 'backfill'.")
        return 0

    print(f"{'series':<16} {'region':<7} {'rows':>8} {'null':>7} "
          f"{'from':<17} {'to':<17}")
    print("-" * 78)
    for row in rows:
        print(f"{row['area']:<16} {row['region']:<7} {row['rows_held']:>8,} "
              f"{row['nulls']:>7,} {row['first_ts'][:16]:<17} {row['last_ts'][:16]:<17}")

    print(f"\nTotal: {store.count():,} readings\n")
    _print_runs(store)
    print(f"\n{ATTRIBUTION}")
    return 0


# Shared flag defaults, applied after parsing rather than by argparse itself.
# See _common_options for why.
SHARED_DEFAULTS = {"db": DEFAULT_DB_PATH, "verbose": False}


def _common_options() -> argparse.ArgumentParser:
    """Flags every subcommand accepts, in either position.

    Two argparse behaviours collide here, and the fix needs both halves.

    First: a flag defined only on the top-level parser is rejected after the
    subcommand, so `gridcast backfill -v` fails while `gridcast -v backfill`
    works. Sharing the flags with every subparser via `parents=` fixes that.

    Second, and less obvious: once both parsers define the same flag, the
    subparser applies its own default *after* the top-level parser has already
    stored the real value — so `gridcast -v backfill` silently reverts verbose
    to False. This is long-standing argparse behaviour, not a mistake in the
    calling code.

    Suppressing the defaults means an unsupplied flag leaves no attribute at
    all, so nothing can be overwritten. The defaults are then applied once,
    after parsing, in `_apply_shared_defaults`.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS,
                        help=f"SQLite database path (default: {DEFAULT_DB_PATH})")
    common.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS,
                        help="log each request as it is made")
    return common


def _apply_shared_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in SHARED_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def main(argv: list[str] | None = None) -> int:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="gridcast",
        description="Ingest and forecast Irish grid data from EirGrid.",
        epilog=ATTRIBUTION,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="check the dashboard is answering",
                       parents=[common])
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("backfill", help="load historical data", parents=[common])
    p.add_argument("--days", type=int, default=365, help="how far back to go")
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.add_argument("--fresh", action="store_true",
                   help="re-request windows already fetched (ignores the ledger)")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("update", help="fetch anything new since the last run",
                       parents=[common])
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("status", help="show what is held and how recent runs went",
                       parents=[common])
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("baseline", help="score the naive forecasts on the data",
                       parents=[common])
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.add_argument("--horizons", type=float, nargs="+", default=[1, 3, 6, 12],
                   help="forecast horizons in hours")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("train", help="train and evaluate models against persistence",
                       parents=[common])
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.add_argument("--horizons", type=float, nargs="+", default=[1, 3, 6, 12])
    p.add_argument("--folds", type=int, default=5, help="walk-forward folds")
    p.add_argument("--save", action="store_true", help="save the trained models")
    p.add_argument("--model-dir", default="models")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("importance", help="which features the model relies on",
                       parents=[common])
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.add_argument("--horizon", type=float, default=6)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_importance)

    p = sub.add_parser("forecast", help="predict from the most recent data",
                       parents=[common])
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.add_argument("--horizon", type=float, default=6)
    p.add_argument("--model-dir", default="models")
    p.set_defaults(func=cmd_forecast)

    args = _apply_shared_defaults(parser.parse_args(argv))
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

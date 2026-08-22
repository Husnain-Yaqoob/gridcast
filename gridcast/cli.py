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
from .config import ATTRIBUTION, AREAS, DEFAULT_DB_PATH, DEFAULT_REGION, REGIONS
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
    print(f"{len(AREAS)} series, chunked into weekly requests. This will take a while.\n")

    result = ingest.backfill(store, client, start, end, region=args.region)
    print(f"\n{result.summary}")
    print(f"Database now holds {store.count():,} readings.")
    return 0


def cmd_update(args) -> int:
    store = Store(args.db)
    client = GridClient()
    result = ingest.update(store, client, region=args.region)
    print(result.summary)
    return 0


def cmd_status(args) -> int:
    store = Store(args.db)
    rows = store.coverage()

    if not rows:
        print("No data held yet. Run 'probe', then 'backfill'.")
        return 0

    print(f"{'series':<16} {'region':<7} {'rows':>8} {'null':>7} "
          f"{'from':<17} {'to':<17}")
    print("-" * 78)
    for row in rows:
        print(f"{row['area']:<16} {row['region']:<7} {row['rows_held']:>8,} "
              f"{row['nulls']:>7,} {row['first_ts'][:16]:<17} {row['last_ts'][:16]:<17}")

    print(f"\nTotal: {store.count():,} readings\n")

    runs = store.recent_runs(5)
    if runs:
        print("Recent runs:")
        for run in runs:
            state = run["status"] or "running"
            print(f"  #{run['id']:<4} {run['started_at'][:16]}  {run['command']:<28} "
                  f"{state:<8} {run['rows_written'] or 0:>7,} rows")
            if run["detail"]:
                print(f"        {run['detail'][:100]}")

    print(f"\n{ATTRIBUTION}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gridcast",
        description="Ingest and forecast Irish grid data from EirGrid.",
        epilog=ATTRIBUTION,
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="log each request")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="check the dashboard is answering")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("backfill", help="load historical data")
    p.add_argument("--days", type=int, default=365, help="how far back to go")
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("update", help="fetch anything new since the last run")
    p.add_argument("--region", default=DEFAULT_REGION, choices=REGIONS)
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("status", help="show what is held and how recent runs went")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

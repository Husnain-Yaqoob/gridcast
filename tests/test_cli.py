"""Argument parsing tests.

These exist because of a real bug. The shared flags were defined only on the
top-level parser, so `gridcast backfill -v` failed with "unrecognized
arguments: -v" while `gridcast -v backfill` worked. Both orderings are
reasonable to type, and the error message explains nothing.
"""

from __future__ import annotations

import argparse

import pytest

from gridcast.cli import main


def _parse(argv):
    """Parse without running anything, by intercepting the dispatch."""
    captured = {}

    def fake(args):
        captured["args"] = args
        return 0

    # Build the parser the same way main() does, then swap the handler.
    import gridcast.cli as cli

    original = {
        name: getattr(cli, name)
        for name in ("cmd_probe", "cmd_backfill", "cmd_update", "cmd_status")
    }
    for name in original:
        setattr(cli, name, fake)
    try:
        main(argv)
    finally:
        for name, func in original.items():
            setattr(cli, name, func)
    return captured.get("args")


@pytest.mark.parametrize("argv", [
    ["-v", "backfill"],          # flag before the subcommand
    ["backfill", "-v"],          # flag after it — the ordering that used to fail
])
def test_verbose_accepted_in_either_position(argv):
    args = _parse(argv)
    assert args.verbose is True


@pytest.mark.parametrize("argv", [
    ["--db", "x.db", "status"],
    ["status", "--db", "x.db"],
])
def test_db_accepted_in_either_position(argv):
    args = _parse(argv)
    assert args.db == "x.db"


def test_defaults_are_sane():
    args = _parse(["backfill"])
    assert args.verbose is False
    assert args.days == 365
    assert args.region == "ALL"


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_region_rejected():
    with pytest.raises(SystemExit):
        main(["backfill", "--region", "MARS"])


# --------------------------------------------------------------- status health
#
# `status` is the only thing standing between an unattended pipeline and the
# three-week silence its own schema comment warns about, so its health logic is
# tested rather than eyeballed.

from datetime import datetime, timedelta, timezone   # noqa: E402

from gridcast.cli import cmd_status                  # noqa: E402
from gridcast.config import STALE_AFTER_HOURS        # noqa: E402
from gridcast.models import Reading                  # noqa: E402
from gridcast.store import Store                     # noqa: E402


def _status(tmp_path, readings):
    store = Store(str(tmp_path / "s.db"))
    store.upsert(readings)
    args = argparse.Namespace(db=str(tmp_path / "s.db"), verbose=False)
    return cmd_status(args)


def _reading(area, hours_ago, value):
    return Reading(
        area=area, region="ALL",
        timestamp_utc=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        value=value,
    )


def test_status_exits_zero_when_every_series_is_current(tmp_path, capsys):
    code = _status(tmp_path, [
        _reading("wind", 1, 500.0),
        _reading("snsp", 1, 60.0),
    ])
    assert code == 0
    assert "Stale" not in capsys.readouterr().out


def test_status_exits_nonzero_when_a_series_has_stalled(tmp_path, capsys):
    """The failure this command exists to catch.

    A partial run leaves some series current and others stranded. Every other
    number on screen still looks healthy — row counts grow, the run log shows
    activity — so the stalled series has to be called out by name.
    """
    code = _status(tmp_path, [
        _reading("snsp", 1, 60.0),
        _reading("wind", STALE_AFTER_HOURS + 8, 500.0),
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "Stale" in out
    assert "wind" in out
    assert "snsp" not in out.split("Stale")[1]


def test_placeholder_rows_do_not_make_a_stalled_series_look_fresh(tmp_path, capsys):
    """The original bug, at the level the user actually sees.

    Wind holds the newest row in the database — a placeholder — and no real
    value for a day. Reporting `to` from the newest row of any kind showed it
    as the freshest series on screen.
    """
    code = _status(tmp_path, [
        _reading("snsp", 1, 60.0),
        _reading("wind", 24, 500.0),
        _reading("wind", -3, None),      # placeholder, three hours in the future
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "wind" in out.split("Stale")[1]


def test_a_run_that_never_finished_stops_claiming_to_be_running(tmp_path, capsys):
    """An orphaned ledger row is a killed process, not work in progress.

    Nothing takes a lock, so it blocks nothing — but a row that says "running"
    forever trains you to ignore the status column, which is the one thing it
    must never do.
    """
    db = str(tmp_path / "s.db")
    store = Store(db)
    store.upsert([_reading("wind", 1, 500.0)])
    run_id = store.start_run("update ALL")
    with store._connect() as connection:
        connection.execute(
            "UPDATE ingest_log SET started_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(hours=9))
             .strftime("%Y-%m-%dT%H:%M:%SZ"), run_id),
        )

    cmd_status(argparse.Namespace(db=db, verbose=False))
    out = capsys.readouterr().out

    assert "abandoned" in out

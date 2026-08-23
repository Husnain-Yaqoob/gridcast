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

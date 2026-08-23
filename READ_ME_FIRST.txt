Open this folder and copy what is INSIDE it over your project.
Do NOT drag the "gridcast-update" folder itself into gridcast\.

NEW
  gridcast\frame.py       readings -> regular time series
  gridcast\features.py    lags, ramps, calendar encodings
  gridcast\evaluate.py    baselines + walk-forward validation
  tests\test_features.py  18 tests, including the leakage guard

CHANGED
  gridcast\config.py      cool-down budget is per run, not per request
  gridcast\client.py      cool-downs report remaining time
  gridcast\ingest.py      stops the run when throttled out
  gridcast\cli.py         new 'baseline' command
  pyproject.toml          unchanged deps, listed for completeness
  README.md               documents the new design decisions

Your data\gridcast.db is NOT in here. Your 263,452 readings are safe.

  python -m pytest              (expect 84 passed)
  python -m gridcast baseline

Supported by EirGrid Group Data

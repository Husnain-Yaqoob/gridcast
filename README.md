# gridcast

Ingests live data from the Irish electricity grid and forecasts short-term wind
generation.

Ireland runs one of the highest shares of wind power of any grid in the world.
That makes two questions worth asking, and this project is built to answer
them with data rather than assertion:

- **How much of demand does wind actually cover, and when is the grid at its dirtiest?**
- **Can wind output be forecast a few hours ahead well enough to be worth anything?**

The second question is the harder one, and the honest answer to it is the point
of the project. In short-horizon forecasting, "output in an hour will be about
what it is now" — a persistence baseline — is unreasonably difficult to beat.
Any model here is reported against that baseline at every horizon, including
the horizons where it loses.

> **Status:** ingestion pipeline complete and tested. Forecasting is next.

---

## Data source

[EirGrid Smart Grid Dashboard](https://www.smartgriddashboard.com/), which
publishes the all-island system at roughly 15-minute resolution.

| Series | Unit | Why it is here |
|---|---|---|
| `demand` | MW | Total system demand |
| `wind` | MW | Metered wind generation — the forecast target |
| `generation` | MW | Total generation, all sources |
| `co2_intensity` | gCO2/kWh | Carbon intensity of what is being generated |
| `snsp` | % | System Non-Synchronous Penetration |
| `interconnection` | MW | Net interconnector flow, positive is import |

`snsp` earns its place. EirGrid operates to a ceiling on non-synchronous
penetration for stability reasons, which means wind is sometimes curtailed
*while it is windy*. A model that only sees wind and weather cannot explain
those hours; one that sees SNSP can.

### Licence and attribution

Data is used under the
[EirGrid Group Open Data Licence](https://www.smartgriddashboard.com/hr/open-data-license/),
which permits reuse, adaptation and republication, commercially and
non-commercially, subject to attribution.

**Supported by EirGrid Group Data**

The licence also reserves EirGrid's right to throttle access where they believe
excessive use is degrading the service. Every rate limit, chunk size and backoff
setting in `gridcast/config.py` exists because of that clause. The client is
synchronous and waits between requests on purpose.

Code in this repository is MIT licensed. The data is not mine to relicense.

---

## Quick start

```bash
git clone https://github.com/Husnain-Yaqoob/gridcast.git
cd gridcast

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[analysis,dev]"

python -m gridcast probe         # is the service answering?
python -m gridcast backfill --days 365
python -m gridcast status
```

`probe` first, always. The dashboard endpoint returns `503` when it is busy —
that happened during development — and it is worth knowing that before starting
a backfill rather than halfway through one.

### Commands

| Command | What it does |
|---|---|
| `probe` | Checks the service answers, writes nothing |
| `backfill --days N` | Loads N days of history for every series |
| `update` | Fetches whatever has appeared since the last run |
| `status` | What is held, how complete it is, how recent runs went |

Add `-v` to watch each request.

### Keeping it current

`update` is designed to be run on a schedule — every hour is ample for a source
that publishes every fifteen minutes.

**Linux/macOS** — `crontab -e`:

```
15 * * * * cd /path/to/gridcast && .venv/bin/python -m gridcast update
```

**Windows** — Task Scheduler, hourly, running `python -m gridcast update` with
the repository as the working directory.

---

## Design decisions

These are the choices that took thought. They are recorded because the
reasoning is more useful than the code.

### A null is not a zero

EirGrid publishes rows with a null value for periods that have not settled and
for gaps in metering. Those nulls survive parsing, storage and querying as
nulls.

Coercing them to zero would say the wind stopped blowing. On a grid where wind
routinely covers a third of demand, that is not a rounding error — it is a
fabricated blackout, and a model trained on it learns that output collapses to
nothing at predictable times. There is a test for every marker the service has
been seen to use, and a separate test asserting that a *genuine* measured zero
is preserved, because wind output really can be near zero on a still night.

### Everything is stored in UTC

The API speaks Irish local time. Ireland observes daylight saving, so one hour
in October occurs twice and one hour in March does not occur at all.

Lag features computed over a series stored in local time are wrong twice a
year, in a way that is nearly invisible in a chart and completely invisible in
a summary statistic. Conversion happens once, at the boundary, in `models.py`.

### The load is idempotent

The fact table's primary key is `(area, region, ts_utc)`, and writes are
`INSERT ... ON CONFLICT DO UPDATE`. Running the pipeline twice produces the
same database, not two copies of every reading.

This is enforced by the database rather than by remembering to be careful,
because a scheduled job *will* be run twice — retried after a timeout, or
kicked off by hand by someone who forgot cron had already done it.

A later fetch overwrites an earlier one deliberately: EirGrid revises
provisional figures as metering settles, so the newest answer is the most
correct one.

### The watermark ignores nulls

Incremental updates ask each series "what is the newest reading you hold *with
a value*?" — not simply the newest row.

If it counted the null placeholder rows, the pipeline would consider itself up
to date, resume from after the gap, and never go back for the real values once
they settled. That leaves a permanent hole that nothing reports.

### Each series carries its own watermark

Carbon intensity has been seen to lag wind by a settlement period. A single
global watermark would either re-fetch everything every run or silently skip
whichever series lags.

### Requests are chunked

Backfill is split into weekly requests. This is not an optimisation — a single
request for a year of 15-minute data asks the service to assemble tens of
thousands of rows, which is exactly the behaviour the fair-use clause exists to
prevent.

### 503 is retried, 404 is not

A `503` from this service means *busy*, not *gone*, so it is retried with
widening backoff. A `404` will still be a `404` in four seconds, so it fails
immediately.

That distinction was originally a bug: `raise_for_status()` throws `HTTPError`,
which is a subclass of `RequestException`, so the first version of the retry
loop caught it and cheerfully retried permanent failures four times. A test
caught it.

### SQLite, not a server

This pipeline runs on a laptop and collects a few thousand rows a day. A
dependency on a running database server would make "does it work on your
machine" a real question for anyone cloning it. The `Store` class is the only
thing that would change if it outgrew this.

---

## Layout

```
gridcast/
├── gridcast/
│   ├── config.py     endpoints, series definitions, rate limits
│   ├── models.py     Reading, and the parsing that produces it
│   ├── client.py     HTTP access, retries, chunking
│   ├── store.py      SQLite storage and idempotent upsert
│   ├── ingest.py     orchestration and run logging
│   └── cli.py        command line
├── tests/            43 tests, no network access required
└── data/             the database lives here (gitignored)
```

Tests never touch the network. A suite that needs the internet fails on a
train, and one that hits a public service on every run is precisely what the
fair-use clause is about.

```bash
python -m pytest
```

---

## Roadmap

- [x] Ingestion, storage, scheduling, run logging
- [ ] Feature engineering — lags, rolling means, ramp rates, hour and season
- [ ] Persistence baseline, evaluated with walk-forward validation
- [ ] Forecast models at +1h, +3h and +6h, reported against that baseline
- [ ] FastAPI service exposing predictions
- [ ] Dashboard: wind share of demand, carbon intensity by hour, forecast vs actual
- [ ] Docker

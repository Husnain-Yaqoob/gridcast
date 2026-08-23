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

> **Status:** a full year of all-island data collected (260k+ readings).
> Features and baselines built. Models next.

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
| `baseline` | Scores the naive forecasts, by horizon |
| `train` | Trains and validates a model per horizon against persistence |
| `importance` | Which features the model actually relies on |
| `forecast` | Predicts from the most recent data held |
| `serve` | Runs the forecast API |

Add `-v` to watch each request.

### The API

```bash
pip install -e ".[api]"
python -m gridcast serve
```

Interactive documentation at `http://127.0.0.1:8000/docs`.

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness, which models are loaded, cache age |
| `GET /horizons` | What can be forecast, and how each scored in validation |
| `GET /latest` | Most recent observed grid state |
| `GET /forecast?horizon=1` | A prediction, with its validated error and skill |

```json
{
  "made_at_utc": "2026-08-23T22:15:00Z",
  "valid_at_utc": "2026-08-23T23:15:00Z",
  "horizon_hours": 1.0,
  "predicted_wind_mw": 470.2,
  "current_wind_mw": 450.0,
  "expected_error_mw": 105.3,
  "persistence_error_mw": 120.8,
  "skill_vs_persistence_pct": 12.8,
  "attribution": "Supported by EirGrid Group Data"
}
```

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

### We got throttled, and the fix was to ask less

The first full backfill was rate-limited. `demand` completed fifteen weekly
windows and then started receiving `403`. `wind` managed six before the same
thing happened, `generation` one. Each series got fewer successes than the one
before it — the signature of a token bucket draining faster than it refills.

The licence reserves EirGrid's right to do exactly this, so the response was to
be less demanding rather than more persistent. Three changes:

**Bigger windows.** Four weeks per request instead of one. A month of
quarter-hourly data is about 2,900 rows — a comfortable response — and it cuts
a year-long backfill from 318 requests to 84. Fewer moderate requests is
gentler on a service than many small ones, and this is the change that mattered
most.

**A longer gap.** Five seconds between requests rather than 1.5.

**403 treated as throttling, not refusal.** A `403` here does not mean "you may
never have this" — the identical request succeeds a minute later. It means "you
have asked too often". Retrying it on a two-second backoff is both useless and
rude, so the client waits two minutes, then four, then six, and gives up on the
series after three cool-downs rather than hammering away.

### The fetch ledger, so a re-run resumes

`fetch_log` records every window successfully retrieved. A backfill skips
windows already in it.

Without this, being throttled is unrecoverable in practice: every re-run
re-requests the same early months, gets cut off at roughly the same point, and
never reaches the later ones no matter how many times it is run. With it, each
attempt makes progress and the backfill completes across as many runs as it
takes.

The ledger is deliberately separate from the readings themselves. "I asked for
October and October was genuinely empty" and "I never asked for October" are
different facts, and counting rows cannot distinguish them.

One exception: a window running up to today is never recorded as complete,
because the rest of today has not been published yet. Marking it done would
freeze the series at that moment and no later run would revisit it.

`--fresh` ignores the ledger and re-requests everything, for when the stored
data is suspect.

### Work is committed as it arrives

A backfill makes over three hundred requests across twenty minutes. Each
weekly chunk is written to the database the moment it lands, rather than
accumulating a year in memory and writing once at the end.

The first version did the latter, and the first real backfill proved why that
was wrong: it was interrupted partway through the third series, and every
request made up to that point was discarded. Twenty minutes of politely-rate-
limited fetching, an empty database, and nothing to show for it.

Stopping the job now keeps everything already fetched, and re-running continues
from where it stopped — which the idempotent upsert already made safe.

### Ctrl+C is handled, because it is not an Exception

`KeyboardInterrupt` inherits from `BaseException`, not `Exception`, so a broad
`except Exception` never sees it. The process died where it stood and left an
open row in the run log with no finish time, which reads forever afterwards as
"this job is still running".

It is now caught explicitly, recorded as `cancelled`, and exits with code 130.

### 503 is retried, 404 is not

A `503` from this service means *busy*, not *gone*, so it is retried with
widening backoff. A `404` will still be a `404` in four seconds, so it fails
immediately.

That distinction was originally a bug: `raise_for_status()` throws `HTTPError`,
which is a subclass of `RequestException`, so the first version of the retry
loop caught it and cheerfully retried permanent failures four times. A test
caught it.

### interconnection is published per jurisdiction

Every other series returns all-island rows. `interconnection` returns separate
ROI, NI and ALL rows whatever region is requested — East-West lands in ROI,
Moyle in NI.

The parser keeps whatever region each row declares, so the database holds all
three. That is more data, not wrong data, but any pivot must filter on region
or those timestamps triple and every join is quietly corrupted.

### SNSP is half-hourly, and filled by exactly one step

SNSP is published every 30 minutes where the rest are quarter-hourly, so on the
model's index every other SNSP slot is empty. It is forward-filled by one step
and no more.

One step asserts that the SNSP measured at 10:00 still applied at 10:15, which
is true of a quantity that moves slowly. An unlimited fill would assert that
the last reading before a three-day outage applied for three days — invention
dressed up as data.

### No feature may see the future

Everything in `features.py` is a fact about the present or the past. The target
is shifted backwards rather than the features forwards, so every row stays
anchored to the moment the forecast is made.

There is a test that corrupts all data after a cut point and asserts no feature
value at or before the cut moves. Leakage is the one bug that makes a model
look *better*, which is why it survives into so many finished projects.

### Time of day is encoded on a circle

Hour 23 and hour 0 are adjacent in reality. As plain integers they are
twenty-three apart, and a model is told midnight is the opposite of 11pm.
Sine/cosine pairs put them next to each other, where they belong.

### Validation walks forward, and always reports the baseline

Splits are expanding-window: train strictly on the past, test strictly on the
future. A random split lets a model learn from Thursday to predict Wednesday,
which raises the score and destroys the forecast.

Every horizon is reported against a persistence baseline — "output will not
change" — including the horizons where a model loses to it. At one hour ahead
persistence is genuinely hard to beat; by six hours it degrades enough that
even the long-run mean overtakes it. That crossover is a finding, and hiding it
would make the numbers meaningless.

### One model per horizon, not one model with a horizon input

Predicting one hour ahead and twelve hours ahead are different problems. At one
hour, current output dominates and little else matters. At twelve, current
output is nearly irrelevant and the model must lean on season, time of day and
recent variability. A single model forced to serve both learns a compromise
that is good at neither.

### The baseline is scored on exactly the same rows as the model

Within each fold, persistence is evaluated on that fold's test set — not on the
whole series, and not on a different sample. Comparing a model measured on one
set of rows to a baseline measured on another is not a comparison, however
favourable the numbers look.

### The headline MAE is weighted by rows

Folds differ in size, because the training window expands. Averaging the
per-fold MAEs would overweight the smallest fold; weighting by row count gives
the error an average prediction actually carries.

### Winning on average is reported separately from winning every time

A model that wins overall but loses in two folds out of five is not reliably
better — it caught a favourable stretch of weather. `train` marks that case, so
the headline number is never quoted without it.

### Missing values are passed to the model, not imputed

The gradient booster handles NaN natively. The auxiliary series have real gaps,
and filling them would invent readings the grid never published. Letting the
model learn "this was missing" is both more honest and more accurate.

### The model ships with a manifest

Every saved model writes a JSON file recording its horizon, the exact feature
columns it expects, how many rows it trained on, and its validation scores. A
`.joblib` on disk with none of that is an object nobody can safely use six
months later.

### The API returns the status code that describes the failure

A horizon with no trained model is a `404` — that resource genuinely does not
exist, and the body names the horizons that do. A database with no usable data
is a `503` — the service is temporarily unable to answer and the caller should
retry. A horizon of zero or 100 is a `422`, rejected by validation before any
work happens.

Plenty of APIs answer everything with `200` and hide the problem in the body.
That makes failure invisible to monitoring, to retry logic and to every
automated caller. "It returned a response" is not "it worked".

### Models load once, data is cached with a deadline

Deserialising a gradient booster takes a noticeable fraction of a second, so
doing it per request would turn a 5ms response into a 300ms one for a file that
has not changed.

The data frame is different: it *does* change, hourly. So it is cached for five
minutes rather than forever — fresher than the source can possibly be, without
re-reading 35,000 rows on every call. Caching it indefinitely would serve a
forecast anchored to yesterday while claiming to be current.

### Every forecast ships with its own error bar

The response carries the model's validated MAE, the persistence MAE it was
measured against, and the skill between them. A prediction without an error bar
invites false confidence, and one without its baseline cannot be judged at all.

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
│   ├── frame.py      readings -> a regular, modellable time series
│   ├── features.py   lags, ramps, calendar encodings
│   ├── evaluate.py   baselines and walk-forward validation
│   ├── model.py      training, validation, saving, forecasting
│   ├── api.py        FastAPI service
│   └── cli.py        command line
├── tests/            113 tests, no network access required
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

- [x] Ingestion, storage, scheduling, run logging (chunk-by-chunk commits, resumable)
- [ ] Feature engineering — lags, rolling means, ramp rates, hour and season
- [ ] Persistence baseline, evaluated with walk-forward validation
- [ ] Forecast models at +1h, +3h and +6h, reported against that baseline
- [ ] FastAPI service exposing predictions
- [ ] Dashboard: wind share of demand, carbon intensity by hour, forecast vs actual
- [ ] Docker

"""What we ask EirGrid for, and how politely we ask.

The Smart Grid Dashboard is a public service, and its open data licence
reserves EirGrid's right to "throttle or limit access to feeds" where they
believe excessive use is degrading it. Every rate limit and backoff setting
in this file exists because of that sentence. A portfolio project has no
business being the reason a national grid operator's public dashboard slows
down.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "https://www.smartgriddashboard.com/DashboardService.svc/data"

# EirGrid's API speaks Irish local time in its query parameters and echoes it
# back in EffectiveTime. Everything downstream of the client works in UTC.
GRID_TIMEZONE = "Europe/Dublin"

# The date format the service expects in datefrom / dateto.
API_DATE_FORMAT = "%d-%b-%Y %H:%M"


@dataclass(frozen=True)
class Area:
    """One measurement series the dashboard publishes."""

    key: str            # the 'area' query parameter
    label: str          # human name, used in the database and in charts
    unit: str
    description: str


# Only the series this project actually reasons about. The dashboard exposes
# more; adding one is a line here rather than a change anywhere else.
AREAS: tuple[Area, ...] = (
    Area("demandactual", "demand", "MW",
         "Total electricity demand on the system."),
    Area("windactual", "wind", "MW",
         "Metered wind generation. The quantity we are forecasting."),
    Area("generationactual", "generation", "MW",
         "Total generation from all sources."),
    Area("co2intensity", "co2_intensity", "gCO2/kWh",
         "Carbon intensity of electricity generated."),
    Area("SnspALL", "snsp", "%",
         "System Non-Synchronous Penetration: the share of demand met by "
         "non-synchronous sources, mostly wind and interconnectors. EirGrid "
         "operates to a ceiling on this, which is why wind is sometimes "
         "curtailed even when it is windy."),
    Area("interconnection", "interconnection", "MW",
         "Net flow across interconnectors. Positive is import."),
)

AREAS_BY_LABEL = {a.label: a for a in AREAS}
AREAS_BY_KEY = {a.key: a for a in AREAS}

# ROI, NI, or the all-island system. The all-island figure is the one that
# matches how the grid is actually operated.
REGIONS: tuple[str, ...] = ("ROI", "NI", "ALL")
DEFAULT_REGION = "ALL"

# --- politeness -----------------------------------------------------------
#
# These numbers are not guesses. The first full backfill was throttled: the
# service served roughly fifteen requests, then returned 403 for the rest, and
# each subsequent series got fewer successes than the one before it — the
# signature of a token bucket draining faster than it refills.
#
# Two things were wrong. Asking for a year in weekly slices meant 53 requests
# per series and 318 in total, which is a lot of asking however politely each
# one is phrased. And 1.5 seconds apart was too fast regardless.

# Four weeks per request rather than one. A month of quarter-hourly data is
# about 2,900 rows — a comfortable response — and it cuts the whole backfill
# from 318 requests to 84. Fewer, moderate requests is easier on the service
# than many small ones, and it is the change that matters most.
MAX_DAYS_PER_REQUEST = 28

# Seconds between consecutive requests. The dashboard publishes every fifteen
# minutes; there has never been anything to gain from going faster.
REQUEST_INTERVAL_SECONDS = 5.0

REQUEST_TIMEOUT_SECONDS = 45

# Transient server-side failures. Worth retrying quickly.
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0
RETRY_STATUS_CODES = frozenset({500, 502, 503, 504})

# Throttling, which is a different thing and needs a different answer.
#
# 403 here does not mean "you may never have this" — the same request succeeds
# fine a minute later. It means "you have asked too often". Retrying it on a
# two-second backoff is useless and rude; the client waits properly instead.
THROTTLE_STATUS_CODES = frozenset({403, 429})
THROTTLE_COOLDOWN_SECONDS = 90.0

# Total cool-downs allowed across a whole run, not per request.
#
# Per-request was the first version and it was badly wrong: 84 requests each
# permitted three escalating waits meant a persistently throttled backfill
# could sit there for hours, apparently frozen. Once the service has said "too
# often" this many times, the honest conclusion is that today's allowance is
# spent — stop, keep what was fetched, and come back later.
MAX_THROTTLE_WAITS = 4

# A long sleep that prints nothing is indistinguishable from a hang. Waits are
# broken into slices so the client can say how much longer it intends to wait.
COOLDOWN_TICK_SECONDS = 15.0

USER_AGENT = (
    "gridcast/0.1 (portfolio project; contact via github.com/Husnain-Yaqoob)"
)

# Attribution required by the EirGrid Group Open Data Licence. Kept in code so
# it travels with the data rather than living only in the README.
ATTRIBUTION = "Supported by EirGrid Group Data"

DEFAULT_DB_PATH = "data/gridcast.db"

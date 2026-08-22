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
# Seconds to wait between consecutive requests. The dashboard publishes at
# 15-minute resolution, so there is nothing to gain from going faster.
REQUEST_INTERVAL_SECONDS = 1.5

# A single request covering too long a window makes the service work hard and
# returns a response we then have to hold in memory. Backfill is chunked.
MAX_DAYS_PER_REQUEST = 7

REQUEST_TIMEOUT_SECONDS = 30

# Retry policy. The service returned a 503 during development, so transient
# failure is the expected case rather than the exceptional one.
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

USER_AGENT = (
    "gridcast/0.1 (portfolio project; contact via github.com/Husnain-Yaqoob)"
)

# Attribution required by the EirGrid Group Open Data Licence. Kept in code so
# it travels with the data rather than living only in the README.
ATTRIBUTION = "Supported by EirGrid Group Data"

DEFAULT_DB_PATH = "data/gridcast.db"

from datetime import datetime
import math
from typing import Any, Dict

from dateutil import tz
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from skyfield.api import load
from skyfield.framelib import ecliptic_frame

# -------------------------------------------------------------------
# FastAPI app metadata
# -------------------------------------------------------------------

app = FastAPI(
    title="Astro Brain Backend",
    version="0.3.0",
    description="Backend service that powers the custom GPT for astrology using Skyfield + DE421.",
)

# -------------------------------------------------------------------
# Skyfield setup: load timescale + ephemeris once at startup
# -------------------------------------------------------------------

ts = load.timescale()
eph = load("de421.bsp")  # downloaded once & cached in the container
earth = eph["earth"]

PLANET_KEYS: Dict[str, str] = {
    "sun": "sun",
    "moon": "moon",
    "mercury": "mercury",
    "venus": "venus",
    "mars": "mars",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
    "pluto": "pluto barycenter",
}

OBLIQUITY_DEG = 23.4392911  # mean obliquity of the ecliptic


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class ChartRequest(BaseModel):
    """
    Input for the /chart endpoint.
    Example:
    {
        "date": "2025-11-27",
        "time": "12:00",
        "timezone": "UTC",
        "lat": 51.5074,
        "lon": -0.1278
    }
    """
    date: str = Field(..., example="2025-11-27")  # YYYY-MM-DD
    time: str = Field(..., example="12:00")       # HH:MM (24h)
    timezone: str = Field(..., example="UTC")     # IANA timezone name
    lat: float = Field(..., example=51.5074)      # latitude in degrees
    lon: float = Field(..., example=-0.1278)      # longitude in degrees (east +, west -)


class ChartResponse(BaseModel):
    """
    Output for /chart.
    Flexible so we can evolve 'chart' structure over time.
    """
    engine: str
    input: Dict[str, Any]
    chart: Dict[str, Any]


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def parse_to_utc(date_str: str, time_str: str, timezone_str: str) -> datetime:
    """Combine date + time + timezone into a timezone-aware UTC datetime."""
    try:
        naive = datetime.fromisoformat(f"{date_str}T{time_str}")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date or time format. Expected date=YYYY-MM-DD and time=HH:MM.",
        )

    zone = tz.gettz(timezone_str)
    if zone is None:
        raise HTTPException(status_code=400, detail="Invalid timezone name.")

    local_dt = naive.replace(tzinfo=zone)
    return local_dt.astimezone(tz.UTC)


def compute_asc_mc(t, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """
    Compute Ascendant and Midheaven (MC) ecliptic longitudes in degrees.
    Based on local sidereal time, latitude, and mean obliquity.
    """
    gmst_hours = t.gmst
    lst_hours = gmst_hours + lon_deg / 15.0
    lst_deg = (lst_hours * 15.0) % 360.0

    eps = math.radians(OBLIQUITY_DEG)
    phi = math.radians(lat_deg)
    theta = math.radians(lst_deg)

    # Ascendant: λ = atan2( sinθ·cosε − tanφ·sinε , cosθ )
    y_asc = math.sin(theta) * math.cos(eps) - math.tan(phi) * math.sin(eps)
    x_asc = math.cos(theta)
    asc_rad = math.atan2(y_asc, x_asc)
    asc_deg = math.degrees(asc_rad) % 360.0

    # Midheaven (MC): tan L = sin(ARMC) / (cos(ARMC) · cosε)
    armc_deg = lst_deg
    armc_rad = math.radians(armc_deg)
    y_mc = math.sin(armc_rad)
    x_mc = math.cos(armc_rad) * math.cos(eps)
    mc_rad = math.atan2(y_mc, x_mc)
    mc_deg = math.degrees(mc_rad) % 360.0

    return asc_deg, mc_deg


def compute_planets(t) -> Dict[str, Dict[str, Any]]:
    """
    Compute geocentric ecliptic longitudes and basic retrograde flags for Sun–Pluto.
    Returns:
    {
      "sun": {"lon": 245.13, "retrograde": false},
      ...
    }
    """
    planets: Dict[str, Dict[str, Any]] = {}

    for name, key in PLANET_KEYS.items():
        target = eph[key]

        # Geocentric position
        astrometric = earth.at(t).observe(target)
        lat_angle, lon_angle, distance = astrometric.frame_latlon(ecliptic_frame)
        lon_deg = lon_angle.degrees % 360.0

        # Retrograde detection: compare longitudes 1 hour before and after
        dt_days = 1.0 / 24.0
        astrom_past = earth.at(t - dt_days).observe(target)
        astrom_future = earth.at(t + dt_days).observe(target)

        _, lon_past_angle, _ = astrom_past.frame_latlon(ecliptic_frame)
        _, lon_future_angle, _ = astrom_future.frame_latlon(ecliptic_frame)

        lon_past = lon_past_angle.degrees % 360.0
        lon_future = lon_future_angle.degrees % 360.0

        diff = (lon_future - lon_past + 540.0) % 360.0 - 180.0
        retrograde = diff < 0

        planets[name] = {"lon": lon_deg, "retrograde": retrograde}

    return planets


def compute_whole_sign_houses(asc_deg: float) -> Dict[str, float]:
    """
    Compute whole-sign house cusps.
    House 1 starts at the sign of the Ascendant, each house 30° further.
    Returns: { "1": deg, "2": deg, ..., "12": deg }
    """
    asc_sign = int(asc_deg // 30)  # 0..11
    houses: Dict[str, float] = {}

    for i in range(12):
        sign_index = (asc_sign + i) % 12
        cusp_deg = sign_index * 30.0
        houses[str(i + 1)] = cusp_deg

    return houses


def compute_mean_node_lon_deg(jd_tt: float) -> float:
    """
    Compute approximate mean lunar node longitude (degrees 0–360).
    Based on standard Meeus-style polynomial.
    """
    T = (jd_tt - 2451545.0) / 36525.0  # Julian centuries from J2000

    # Mean longitude of the ascending node of the Moon (omega)
    omega = (
        125.04452
        - 1934.136261 * T
        + 0.0020708 * T * T
        + (T * T * T) / 450000.0
    )

    omega = omega % 360.0
    # Convert to ecliptic longitude direction used in astrology
    lon = (360.0 - omega) % 360.0
    return lon


def compute_true_node(t) -> Dict[str, Any]:
    """
    For now we return the mean node, marked as retrograde=True.
    Later we can refine for a more exact 'true node' if needed.
    """
    jd_tt = t.tt  # Julian Date (Terrestrial Time)
    lon = compute_mean_node_lon_deg(jd_tt)
    return {"lon": lon, "retrograde": True}


# -------------------------------------------------------------------
# Basic endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Astro Brain Backend is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------------------------
# Main astrology endpoint: /chart
# -------------------------------------------------------------------

@app.post("/chart", response_model=ChartResponse)
def chart(payload: ChartRequest):
    """
    Compute a chart using Skyfield + DE421.
    Structure is designed to match what your custom GPT expects:
    {
      "engine": "skyfield_de421",
      "input": {...},
      "chart": {
        "asc": ...,
        "mc": ...,
        "planets": {...},
        "houses": {...},     # whole-sign houses
        "true_node": {...}   # mean node, retrograde
      }
    }
    """

    if not (-90.0 <= payload.lat <= 90.0):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90 degrees.")
    if not (-180.0 <= payload.lon <= 180.0):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180 degrees.")

    # Convert input date/time/timezone to UTC and then to Skyfield Time
    dt_utc = parse_to_utc(payload.date, payload.time, payload.timezone)
    t = ts.from_datetime(dt_utc)

    # Core computations
    asc_deg, mc_deg = compute_asc_mc(t, payload.lat, payload.lon)
    planets = compute_planets(t)
    houses = compute_whole_sign_houses(asc_deg)
    true_node = compute_true_node(t)

    chart_obj: Dict[str, Any] = {
        "asc": asc_deg,
        "mc": mc_deg,
        "planets": planets,
        "houses": houses,
        "true_node": true_node,
    }

    return ChartResponse(
        engine="skyfield_de421",
        input=payload.dict(),
        chart=chart_obj,
    )

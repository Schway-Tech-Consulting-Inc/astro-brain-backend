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
    version="0.2.0",
    description="Backend service that powers the custom GPT for astrology using Skyfield + DE421.",
)

# -------------------------------------------------------------------
# Skyfield setup: load timescale + ephemeris once at startup
# -------------------------------------------------------------------

ts = load.timescale()
eph = load("de421.bsp")  # This will download once and cache inside the container
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

OBLIQUITY_DEG = 23.4392911  # mean obliquity, good enough for our use


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class ChartRequest(BaseModel):
    """
    Input format for the /chart endpoint.
    Matches the structure you used in Postman:
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
    timezone: str = Field(..., example="UTC")     # IANA name, e.g. "UTC", "Europe/London"
    lat: float = Field(..., example=51.5074)      # latitude in degrees
    lon: float = Field(..., example=-0.1278)      # longitude in degrees (east +, west -)


class ChartResponse(BaseModel):
    """
    Output format for /chart.
    We keep it generic so we can evolve chart content over time.
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
    Based on local sidereal time, latitude, and mean obliquity of the ecliptic.
    """

    # 1) Local sidereal time (LST) in degrees
    gmst_hours = t.gmst              # Greenwich Mean Sidereal Time, in hours
    lst_hours = gmst_hours + lon_deg / 15.0
    lst_deg = (lst_hours * 15.0) % 360.0

    eps = math.radians(OBLIQUITY_DEG)
    phi = math.radians(lat_deg)
    theta = math.radians(lst_deg)

    # --- Ascendant ---
    # λ = atan2( sinθ·cosε − tanφ·sinε , cosθ )
    y_asc = math.sin(theta) * math.cos(eps) - math.tan(phi) * math.sin(eps)
    x_asc = math.cos(theta)
    asc_rad = math.atan2(y_asc, x_asc)
    asc_deg = math.degrees(asc_rad) % 360.0

    # --- Midheaven (MC) ---
    # tan L = sin(ARMC) / (cos(ARMC) · cosε)
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

        # Simple retrograde detection: look 1 hour before and after
        dt_days = 1.0 / 24.0
        astrom_past = earth.at(t - dt_days).observe(target)
        astrom_future = earth.at(t + dt_days).observe(target)

        _, lon_past_angle, _ = astrom_past.frame_latlon(ecliptic_frame)
        _, lon_future_angle, _ = astrom_future.frame_latlon(ecliptic_frame)

        lon_past = lon_past_angle.degrees % 360.0
        lon_future = lon_future_angle.degrees % 360.0

        # Signed angular difference, in range [-180, +180]
        diff = (lon_future - lon_past + 540.0) % 360.0 - 180.0
        retrograde = diff < 0

        planets[name] = {"lon": lon_deg, "retrograde": retrograde}

    return planets


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
    Compute a basic chart using Skyfield + DE421.
    Response matches the structure of the Postman example you shared:
    {
      "engine": "skyfield_de421",
      "input": {...},
      "chart": {
        "asc": ...,
        "mc": ...,
        "planets": {...},
        "houses": null,
        "true_node": null
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

    # For now, houses + true_node are not yet implemented (we'll add them in later steps)
    chart_obj: Dict[str, Any] = {
        "asc": asc_deg,
        "mc": mc_deg,
        "planets": planets,
        "houses": None,
        "true_node": None,
    }

    return ChartResponse(
        engine="skyfield_de421",
        input=payload.dict(),
        chart=chart_obj,
    )

from datetime import datetime
import math
from typing import Any, Dict, List

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
    version="0.5.0",
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
    Input for the /chart endpoint and for nested natal/transit charts.
    """
    date: str = Field(..., example="2025-11-27")  # YYYY-MM-DD
    time: str = Field(..., example="12:00")       # HH:MM (24h)
    timezone: str = Field(..., example="UTC")     # IANA timezone name
    lat: float = Field(..., example=51.5074)      # latitude in degrees
    lon: float = Field(..., example=-0.1278)      # longitude in degrees (east +, west -)


class ChartResponse(BaseModel):
    """
    Output for /chart.
    """
    engine: str
    input: Dict[str, Any]
    chart: Dict[str, Any]


class TransitRequest(BaseModel):
    """
    Input for /transits.
    natal: natal chart details
    transit: transit chart details (often 'now', but can be any date)
    max_orb: maximum orb for transit aspects (in degrees)
    """
    natal: ChartRequest
    transit: ChartRequest
    max_orb: float = Field(default=2.0, example=2.0)


class TransitResponse(BaseModel):
    """
    Output for /transits.
    """
    engine: str
    natal: Dict[str, Any]
    transit: Dict[str, Any]
    aspects: List[Dict[str, Any]]


# -------------------------------------------------------------------
# Utility functions: time, asc/mc, planets
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


# -------------------------------------------------------------------
# Houses (whole-sign) and node
# -------------------------------------------------------------------

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
# Moon phase (angle, illumination, phase name)
# -------------------------------------------------------------------

def compute_moon_phase(sun_lon: float, moon_lon: float) -> Dict[str, Any]:
    """
    Compute moon phase based on the elongation angle between Sun and Moon.
    Returns:
      - angle (0–360)
      - illumination (0–1)
      - phase_name (string)
    """
    angle = (moon_lon - sun_lon) % 360.0
    illumination = 0.5 * (1 - math.cos(math.radians(angle)))  # 0..1

    # Simple naming scheme
    if angle < 10 or angle > 350:
        name = "New Moon"
    elif 10 <= angle < 80:
        name = "Waxing Crescent"
    elif 80 <= angle < 100:
        name = "First Quarter"
    elif 100 <= angle < 170:
        name = "Waxing Gibbous"
    elif 170 <= angle < 190:
        name = "Full Moon"
    elif 190 <= angle < 260:
        name = "Waning Gibbous"
    elif 260 <= angle < 280:
        name = "Last Quarter"
    else:
        name = "Waning Crescent"

    return {
        "angle": angle,
        "illumination": illumination,
        "phase_name": name,
    }


# -------------------------------------------------------------------
# Aspects (within a single chart + transits)
# -------------------------------------------------------------------

MAJOR_ASPECTS = {
    0: "conjunction",
    60: "sextile",
    90: "square",
    120: "trine",
    180: "opposition",
}


def aspect_between(a: float, b: float, max_orb: float = 6.0):
    """
    Compute major aspect between two longitudes if within orb.
    Returns dict {type, angle, orb} or None.
    """
    diff = abs(a - b) % 360.0
    if diff > 180:
        diff = 360 - diff

    for exact_angle, name in MAJOR_ASPECTS.items():
        orb = abs(diff - exact_angle)
        if orb <= max_orb:
            return {
                "type": name,
                "angle": exact_angle,
                "orb": orb,
            }
    return None


def compute_chart_aspects(planets: Dict[str, Dict[str, float]], max_orb: float = 6.0) -> List[Dict[str, Any]]:
    """
    Compute aspects between all planet pairs in a single chart.
    planets: { "sun": {"lon": ...}, "moon": {"lon": ...}, ... }
    """
    aspects: List[Dict[str, Any]] = []
    keys = list(planets.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p1, p2 = keys[i], keys[j]
            a = planets[p1]["lon"]
            b = planets[p2]["lon"]
            asp = aspect_between(a, b, max_orb=max_orb)
            if asp:
                aspects.append({
                    "p1": p1,
                    "p2": p2,
                    **asp
                })

    return aspects


def compute_transit_aspects(
    natal_planets: Dict[str, Dict[str, float]],
    transit_planets: Dict[str, Dict[str, float]],
    max_orb: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Compute aspects between natal and transit planets.
    Returns:
      [
        {
          "natal": "sun",
          "transit": "saturn",
          "type": "square",
          "angle": 90,
          "orb": 1.2
        },
        ...
      ]
    """
    aspects: List[Dict[str, Any]] = []

    for n_name, n_data in natal_planets.items():
        for t_name, t_data in transit_planets.items():
            a = n_data["lon"]
            b = t_data["lon"]
            asp = aspect_between(a, b, max_orb=max_orb)
            if asp:
                aspects.append({
                    "natal": n_name,
                    "transit": t_name,
                    **asp
                })

    return aspects


# -------------------------------------------------------------------
# Core chart builder (used by /chart and /transits)
# -------------------------------------------------------------------

def build_chart(payload: ChartRequest) -> Dict[str, Any]:
    """
    Build a full chart object (without wrapping in engine/input).
    Reused by /chart and /transits.
    """
    if not (-90.0 <= payload.lat <= 90.0):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90 degrees.")
    if not (-180.0 <= payload.lon <= 180.0):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180 degrees.")

    dt_utc = parse_to_utc(payload.date, payload.time, payload.timezone)
    t = ts.from_datetime(dt_utc)

    asc_deg, mc_deg = compute_asc_mc(t, payload.lat, payload.lon)
    planets = compute_planets(t)
    houses = compute_whole_sign_houses(asc_deg)
    true_node = compute_true_node(t)

    # Moon phase (requires sun and moon longitudes)
    if "sun" not in planets or "moon" not in planets:
        raise HTTPException(status_code=500, detail="Sun or Moon data missing for moon phase calculation.")
    moon_phase = compute_moon_phase(planets["sun"]["lon"], planets["moon"]["lon"])

    aspects = compute_chart_aspects(planets)

    chart_obj: Dict[str, Any] = {
        "asc": asc_deg,
        "mc": mc_deg,
        "planets": planets,
        "houses": houses,
        "true_node": true_node,
        "moon_phase": moon_phase,
        "aspects": aspects,
    }
    return chart_obj


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
# /chart endpoint
# -------------------------------------------------------------------

@app.post("/chart", response_model=ChartResponse)
def chart(payload: ChartRequest):
    """
    Single chart endpoint used by the custom GPT.
    """
    chart_obj = build_chart(payload)
    return ChartResponse(
        engine="skyfield_de421",
        input=payload.dict(),
        chart=chart_obj,
    )


# -------------------------------------------------------------------
# /transits endpoint
# -------------------------------------------------------------------

@app.post("/transits", response_model=TransitResponse)
def transits(payload: TransitRequest):
    """
    Compute transit aspects between a natal chart and a transit chart.
    """
    natal_chart = build_chart(payload.natal)
    transit_chart = build_chart(payload.transit)

    natal_planets = natal_chart["planets"]
    transit_planets = transit_chart["planets"]

    aspects = compute_transit_aspects(
        natal_planets=natal_planets,
        transit_planets=transit_planets,
        max_orb=payload.max_orb,
    )

    return TransitResponse(
        engine="skyfield_de421",
        natal={
            "input": payload.natal.dict(),
            "chart": natal_chart,
        },
        transit={
            "input": payload.transit.dict(),
            "chart": transit_chart,
        },
        aspects=aspects,
    )

from datetime import datetime
import math
from typing import Any, Dict, List
import os
import logging

from dateutil import tz
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from skyfield.api import Loader
from skyfield.framelib import ecliptic_frame

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI app metadata
# -------------------------------------------------------------------

app = FastAPI(
    title="Astro Brain Backend",
    version="0.6.2",
    description="Backend service that powers the custom GPT for astrology using Skyfield + DE421.",
)

# -------------------------------------------------------------------
# Skyfield setup: load timescale + ephemeris once at startup
# Use /tmp for cache (writable on Railway and other cloud platforms)
# -------------------------------------------------------------------

try:
    logger.info("Initializing Skyfield loader...")
    load = Loader('/tmp/skyfield-data')
    logger.info("Loading timescale...")
    ts = load.timescale()
    logger.info("Loading ephemeris (de421.bsp) - this may take a moment on first run...")
    eph = load('de421.bsp')  # Auto-downloads if missing (~17MB)
    earth = eph["earth"]
    logger.info("Skyfield initialization complete!")
except Exception as e:
    logger.error(f"Failed to initialize Skyfield: {e}")
    raise

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
    Input for /chart and nested chart requests.
    """
    date: str = Field(..., example="1990-05-14")  # YYYY-MM-DD
    time: str = Field(..., example="15:30")       # HH:MM (24h)
    timezone: str = Field(..., example="UTC")     # IANA timezone name
    lat: float = Field(..., example=51.5074)      # latitude in degrees
    lon: float = Field(..., example=-0.1278)      # longitude in degrees (east +, west -)


class ChartResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    engine: str
    input: Dict[str, Any]
    chart: Dict[str, Any]


class TransitRequest(BaseModel):
    """
    Input for /transits.
    """
    natal: ChartRequest
    transit: ChartRequest
    max_orb: float = Field(default=2.0, example=2.0)


class TransitResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    engine: str
    natal: Dict[str, Any]
    transit: Dict[str, Any]
    aspects: List[Dict[str, Any]]


class SynastryRequest(BaseModel):
    """
    Input for /synastry.
    """
    person1: ChartRequest
    person2: ChartRequest
    max_orb: float = Field(default=4.0, example=4.0)


class SynastryResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    engine: str
    person1: Dict[str, Any]
    person2: Dict[str, Any]
    aspects: List[Dict[str, Any]]


class CompositeRequest(BaseModel):
    """
    Input for /composite.
    """
    person1: ChartRequest
    person2: ChartRequest


class CompositeResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    engine: str
    person1: Dict[str, Any]
    person2: Dict[str, Any]
    composite: Dict[str, Any]


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

        # Geocentric position at time t
        astrometric = earth.at(t).observe(target)
        lat_angle, lon_angle, distance = astrometric.frame_latlon(ecliptic_frame)
        lon_deg = lon_angle.degrees % 360.0

        # Retrograde detection: compare longitudes 1 hour before and after
        # NOTE: must build new Time objects; we can't do t +/- float directly.
        dt_days = 1.0 / 24.0
        t_past = ts.tt_jd(t.tt - dt_days)
        t_future = ts.tt_jd(t.tt + dt_days)

        astrom_past = earth.at(t_past).observe(target)
        astrom_future = earth.at(t_future).observe(target)

        _, lon_past_angle, _ = astrom_past.frame_latlon(ecliptic_frame)
        _, lon_future_angle, _ = astrom_future.frame_latlon(ecliptic_frame)

        lon_past = lon_past_angle.degrees % 360.0
        lon_future = lon_future_angle.degrees % 360.0

        diff = (lon_future - lon_past + 540.0) % 360.0 - 180.0
        retrograde = bool(diff < 0)  # Convert numpy.bool to Python bool

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
# Aspects (within chart, transits, synastry)
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
    Aspects between natal and transit planets.
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


def compute_synastry_aspects(
    p1_planets: Dict[str, Dict[str, float]],
    p2_planets: Dict[str, Dict[str, float]],
    max_orb: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Aspects between two natal charts (synastry).
    """
    aspects: List[Dict[str, Any]] = []

    for name1, d1 in p1_planets.items():
        for name2, d2 in p2_planets.items():
            a = d1["lon"]
            b = d2["lon"]
            asp = aspect_between(a, b, max_orb=max_orb)
            if asp:
                aspects.append({
                    "p1": name1,
                    "p2": name2,
                    **asp
                })
    return aspects


# -------------------------------------------------------------------
# Composite helpers
# -------------------------------------------------------------------

def midpoint_lon(a: float, b: float) -> float:
    """
    Circular midpoint between two longitudes.
    """
    diff = (b - a + 540.0) % 360.0 - 180.0
    return (a + diff / 2.0) % 360.0


def compute_composite_planets(
    p1_planets: Dict[str, Dict[str, float]],
    p2_planets: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """
    Composite planets as midpoint of longitudes.
    Retrograde is usually not meaningful in a composite, so we set False.
    """
    composite: Dict[str, Dict[str, Any]] = {}
    for name, d1 in p1_planets.items():
        if name not in p2_planets:
            continue
        d2 = p2_planets[name]
        comp_lon = midpoint_lon(d1["lon"], d2["lon"])
        composite[name] = {"lon": comp_lon, "retrograde": False}  # Already Python bool
    return composite


def compute_composite_chart(chart1: Dict[str, Any], chart2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a composite chart mostly by midpoint:
    - planets: midpoint of longitudes
    - asc, mc: midpoint of natal asc/mc
    - houses: whole-sign from composite asc
    - true_node: midpoint of node longitudes (retrograde=True)
    - moon_phase, aspects: recomputed from composite planets
    """
    comp_planets = compute_composite_planets(chart1["planets"], chart2["planets"])

    comp_asc = midpoint_lon(chart1["asc"], chart2["asc"])
    comp_mc = midpoint_lon(chart1["mc"], chart2["mc"])

    comp_houses = compute_whole_sign_houses(comp_asc)

    comp_true_node = {
        "lon": midpoint_lon(chart1["true_node"]["lon"], chart2["true_node"]["lon"]),
        "retrograde": True,
    }

    if "sun" not in comp_planets or "moon" not in comp_planets:
        raise HTTPException(status_code=500, detail="Composite chart missing Sun or Moon.")
    comp_moon_phase = compute_moon_phase(comp_planets["sun"]["lon"], comp_planets["moon"]["lon"])

    comp_aspects = compute_chart_aspects(comp_planets)

    return {
        "asc": comp_asc,
        "mc": comp_mc,
        "planets": comp_planets,
        "houses": comp_houses,
        "true_node": comp_true_node,
        "moon_phase": comp_moon_phase,
        "aspects": comp_aspects,
    }


# -------------------------------------------------------------------
# Core chart builder (used everywhere)
# -------------------------------------------------------------------

def build_chart(payload: ChartRequest) -> Dict[str, Any]:
    """
    Build a full chart object (without wrapping in engine/input).
    Reused by /chart, /transits, /synastry, /composite.
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
    return {
        "message": "Astro Brain Backend is running.",
        "version": "0.6.2",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {"status": "ok", "ephemeris_loaded": True}


@app.get("/test-chart")
def test_chart():
    """Test endpoint to diagnose chart generation issues"""
    try:
        # Test data
        test_payload = ChartRequest(
            date="1990-05-14",
            time="15:30",
            timezone="UTC",
            lat=51.5074,
            lon=-0.1278
        )
        
        # Try to build chart
        chart_obj = build_chart(test_payload)
        
        # Try to build the full response like the real endpoint does
        try:
            response = ChartResponse(
                engine="skyfield_de421",
                input=test_payload.model_dump(),
                chart=chart_obj,
            )
            return {
                "status": "success",
                "message": "Full response creation works!",
                "response_type": str(type(response))
            }
        except Exception as response_error:
            import traceback
            return {
                "status": "response_error",
                "message": "Chart builds but response fails",
                "error_type": type(response_error).__name__,
                "error_message": str(response_error),
                "traceback": traceback.format_exc()
            }
        
    except Exception as e:
        import traceback
        return {
            "status": "chart_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/chart2")
def chart2(payload: ChartRequest):
    """Working chart endpoint - exact copy of test-chart but with POST"""
    try:
        chart_obj = build_chart(payload)
        return {
            "engine": "skyfield_de421",
            "input": payload.model_dump(),
            "chart": chart_obj,
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# -------------------------------------------------------------------
# /chart endpoint - RENAMED TO FIX ROUTE REGISTRATION BUG
# -------------------------------------------------------------------

@app.post("/chart")
def generate_chart(payload: ChartRequest):
    """
    Single chart endpoint used by the custom GPT.
    """
    chart_obj = build_chart(payload)
    result = {
        "engine": "skyfield_de421",
        "input": payload.model_dump(),
        "chart": chart_obj,
    }
    return JSONResponse(content=result)


# -------------------------------------------------------------------
# /transits endpoint
# -------------------------------------------------------------------

@app.post("/transits")
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

    return {
        "engine": "skyfield_de421",
        "natal": {
            "input": payload.natal.model_dump(),
            "chart": natal_chart,
        },
        "transit": {
            "input": payload.transit.model_dump(),
            "chart": transit_chart,
        },
        "aspects": aspects,
    }


# -------------------------------------------------------------------
# /synastry endpoint
# -------------------------------------------------------------------

@app.post("/synastry")
def synastry(payload: SynastryRequest):
    """
    Compute synastry aspects between two natal charts.
    """
    chart1 = build_chart(payload.person1)
    chart2 = build_chart(payload.person2)

    aspects = compute_synastry_aspects(
        chart1["planets"],
        chart2["planets"],
        max_orb=payload.max_orb,
    )

    return {
        "engine": "skyfield_de421",
        "person1": {
            "input": payload.person1.model_dump(),
            "chart": chart1,
        },
        "person2": {
            "input": payload.person2.model_dump(),
            "chart": chart2,
        },
        "aspects": aspects,
    }


# -------------------------------------------------------------------
# /composite endpoint
# -------------------------------------------------------------------

@app.post("/composite")
def composite(payload: CompositeRequest):
    """
    Compute a composite chart (midpoint method) between two natal charts.
    """
    chart1 = build_chart(payload.person1)
    chart2 = build_chart(payload.person2)

    comp_chart = compute_composite_chart(chart1, chart2)

    return {
        "engine": "skyfield_de421",
        "person1": {
            "input": payload.person1.model_dump(),
            "chart": chart1,
        },
        "person2": {
            "input": payload.person2.model_dump(),
            "chart": chart2,
        },
        "composite": comp_chart,
    }


# -------------------------------------------------------------------
# Main entry point for Railway
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

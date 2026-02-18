# satgenpy/satgen/dynamic_state/aircraft_motion.py

import math
from satgen.distance_tools.distance_tools import geodetic2cartesian

# If your ns-3 simulation is short (e.g., 200s) but you want visible motion,
# this compresses real flight time into simulation time deterministically.
# Example: TIME_SCALE=60 => 200s sim ~ 3.33h of aircraft motion.
TIME_SCALE = 60.0

CRUISE_ALT_M = 10668.0  # 35,000 ft

AIRPORTS = {
    "LHR": (51.4700,  -0.4543,   25.0),
    "JFK": (40.6413, -73.7781,    4.0),
    "FRA": (50.0379,   8.5622,  111.0),
    "DXB": (25.2532,  55.3657,   19.0),
    "SIN": ( 1.3644, 103.9915,    7.0),
    "SYD": (-33.9399,151.1753,    6.0),
    "HND": (35.5494, 139.7798,    6.0),
    "NRT": (35.7719, 140.3929,   41.0),
    "LAX": (33.9416,-118.4085,   38.0),
    "GRU": (-23.4356,-46.4731,  750.0),
    "JNB": (-26.1337, 28.2420, 1694.0),
    "HNL": (21.3187,-157.9225,    4.0),
    "AKL": (-37.0082,174.7850,    7.0),
}

# (callsign, origin, dest, depart_time_s, duration_s)
FLIGHTS = [
    ("P0_LHR_JFK",  "LHR", "JFK",    0.0, 7.0*3600 + 20*60),
    ("P1_FRA_DXB",  "FRA", "DXB",  900.0, 6.0*3600 + 20*60),
    ("P2_DXB_SIN",  "DXB", "SIN", 1800.0, 7.0*3600 + 15*60),
    ("P3_SIN_SYD",  "SIN", "SYD", 2700.0, 7.0*3600 + 50*60),
    ("P4_HND_LAX",  "HND", "LAX", 1200.0, 9.0*3600 + 50*60),
    ("P5_LAX_JFK",  "LAX", "JFK", 3600.0, 5.0*3600 + 30*60),
    ("P6_GRU_JNB",  "GRU", "JNB",  600.0, 8.0*3600 + 35*60),
    ("P7_JNB_LHR",  "JNB", "LHR", 5400.0,11.0*3600 + 15*60),
    ("P8_LAX_HNL",  "LAX", "HNL",    0.0, 5.0*3600 + 40*60),
    ("P9_SYD_AKL",  "SYD", "AKL", 4200.0, 3.0*3600 +  0*60),
    ("P10_DXB_JNB", "DXB", "JNB", 7200.0, 8.0*3600 + 25*60),
    ("P11_SIN_NRT", "SIN", "NRT", 8100.0, 7.0*3600 + 10*60),
]

TURNAROUND_S = 45.0 * 60.0  # 45 minutes on the ground

def _deg2rad(d): return d * math.pi / 180.0
def _rad2deg(r): return r * 180.0 / math.pi

def _ll_to_unit(lat_deg, lon_deg):
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    return (x, y, z)

def _unit_to_ll(x, y, z):
    lat = math.atan2(z, math.sqrt(x*x + y*y))
    lon = math.atan2(y, x)
    return (_rad2deg(lat), _rad2deg(lon))

def _slerp_ll(lat1, lon1, lat2, lon2, f):
    # Great-circle interpolation using spherical linear interpolation (slerp)
    v1 = _ll_to_unit(lat1, lon1)
    v2 = _ll_to_unit(lat2, lon2)
    dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]))
    omega = math.acos(dot)

    if omega < 1e-12:
        return (lat1, lon1)

    so = math.sin(omega)
    a = math.sin((1.0 - f) * omega) / so
    b = math.sin(f * omega) / so
    x = a*v1[0] + b*v2[0]
    y = a*v1[1] + b*v2[1]
    z = a*v1[2] + b*v2[2]
    return _unit_to_ll(x, y, z)

def _flight_position(now_real_s, origin, dest, depart_s, duration_s):
    # Round-trip cycle: origin -> dest -> origin, with turnaround pauses
    arrive_s = depart_s + duration_s
    return_depart_s = arrive_s + TURNAROUND_S
    return_arrive_s = return_depart_s + duration_s
    cycle_s = return_arrive_s + TURNAROUND_S

    t = now_real_s % cycle_s

    o_lat, o_lon, _ = AIRPORTS[origin]
    d_lat, d_lon, _ = AIRPORTS[dest]

    if t < depart_s:
        return (o_lat, o_lon)
    if depart_s <= t < arrive_s:
        f = (t - depart_s) / duration_s
        return _slerp_ll(o_lat, o_lon, d_lat, d_lon, f)
    if arrive_s <= t < return_depart_s:
        return (d_lat, d_lon)
    if return_depart_s <= t < return_arrive_s:
        f = (t - return_depart_s) / duration_s
        return _slerp_ll(d_lat, d_lon, o_lat, o_lon, f)
    return (o_lat, o_lon)

def apply_aircraft_positions(ground_stations, time_since_epoch_ns):
    """
    Mutates entries in ground_stations in-place.
    Planes are recognized by name starting with 'Plane_'.
    """
    now_sim_s = time_since_epoch_ns / 1e9
    now_real_s = now_sim_s * TIME_SCALE

    # Build a stable mapping from plane index (0..len(FLIGHTS)-1) to gid
    # Assumption: your ground_stations list contains planes AFTER fixed GS,
    # and plane names are "Plane_<callsign>" in the same order as FLIGHTS.
    plane_defs = {f"P{i}_{FLIGHTS[i][0]}": FLIGHTS[i] for i in range(len(FLIGHTS))}

    # Match by exact name: "Plane_" + callsign
    for gs in ground_stations:
        if not gs["name"].startswith("Plane_"):
            continue

        callsign = gs["name"][len("Plane_"):].strip()
        # Find flight by callsign
        match = None
        for (cs, origin, dest, depart_s, duration_s) in FLIGHTS:
            if cs == callsign:
                match = (origin, dest, depart_s, duration_s)
                break
        if match is None:
            continue

        origin, dest, depart_s, duration_s = match
        lat_deg, lon_deg = _flight_position(now_real_s, origin, dest, depart_s, duration_s)

        # Update fields satgen uses for distance calculations
        gs["latitude_degrees_str"] = str(lat_deg)
        gs["longitude_degrees_str"] = str(lon_deg)
        gs["elevation_m_float"] = float(CRUISE_ALT_M)

        # Keep cartesian consistent (even if not always used)
        x, y, z = geodetic2cartesian(lat_deg, lon_deg, float(CRUISE_ALT_M))
        gs["cartesian_x"] = float(x)
        gs["cartesian_y"] = float(y)
        gs["cartesian_z"] = float(z)

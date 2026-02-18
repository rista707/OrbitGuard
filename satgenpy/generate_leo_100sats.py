#!/usr/bin/env python3
import os

from satgen.tles import generate_tles_from_scratch_with_sgp
from satgen.isls import generate_plus_grid_isls
from satgen.ground_stations import extend_ground_stations
from satgen.interfaces import generate_simple_gsl_interfaces_info
from satgen.description import generate_description
from satgen.dynamic_state import help_dynamic_state

# Where ns-3 run config expects networks
# Adjust if your ns3-sat-sim path differs.
OUTPUT_ROOT = "../ns3-sat-sim/satellite_network_state"

NAME = "leo_120sats_22endpoints"

# ---- Satellite constellation ----
N_ORBITS = 10
N_SATS_PER_ORBIT = 12
N_SATS = N_ORBITS * N_SATS_PER_ORBIT

# ---- Endpoints: 10 ground stations + 12 planes ----
# Planes are modeled as "ground stations" at 10 km elevation (static).
ENDPOINTS = [
    # --- Ground stations (10) ---
    ("London_UK",        51.5074,   -0.1278,     50.0),
    ("Frankfurt_DE",     50.1109,    8.6821,    112.0),
    ("Dubai_AE",         25.2048,   55.2708,      5.0),
    ("Singapore_SG",      1.3521,  103.8198,     15.0),
    ("Tokyo_JP",         35.6762,  139.6503,     40.0),
    ("Sydney_AU",       -33.8688,  151.2093,     58.0),
    ("NewYork_US",       40.7128,  -74.0060,     10.0),
    ("LosAngeles_US",    34.0522, -118.2437,     71.0),
    ("SaoPaulo_BR",     -23.5505,  -46.6333,    760.0),
    ("Kathmandu_NP",     27.7172,   85.3240,    1400.0),

    # --- Planes (12) : deterministic flight paths (Option A) ---
    # All start at their origin airport at t=0, then move deterministically.
    ("Plane_P0_LHR_JFK",   51.4700,   -0.4543, 10668.0),  # Heathrow
    ("Plane_P1_FRA_DXB",   50.0379,    8.5622, 10668.0),  # Frankfurt
    ("Plane_P2_DXB_SIN",   25.2532,   55.3657, 10668.0),  # Dubai
    ("Plane_P3_SIN_SYD",    1.3644,  103.9915, 10668.0),  # Changi
    ("Plane_P4_HND_LAX",   35.5494,  139.7798, 10668.0),  # Haneda
    ("Plane_P5_LAX_JFK",   33.9416, -118.4085, 10668.0),  # LAX
    ("Plane_P6_GRU_JNB",  -23.4356,  -46.4731, 10668.0),  # GRU
    ("Plane_P7_JNB_LHR",  -26.1337,   28.2420, 10668.0),  # JNB
    ("Plane_P8_LAX_HNL",   33.9416, -118.4085, 10668.0),  # LAX
    ("Plane_P9_SYD_AKL",  -33.9399,  151.1753, 10668.0),  # SYD
    ("Plane_P10_DXB_JNB",  25.2532,   55.3657, 10668.0),  # DXB
    ("Plane_P11_SIN_NRT",   1.3644,  103.9915, 10668.0),  # SIN
]
N_ENDPOINTS = len(ENDPOINTS)

# ---- Link ranges (meters) ----
MAX_GSL_M = 1_200_000.0   # ~1200 km slant range
MAX_ISL_M = 5700000.0   # ~5000 km

# ---- Dynamic state generation ----
TIME_STEP_MS = 100
DURATION_S = 200
DYN_ALGO = "algorithm_free_one_only_over_isls"  # simplest + scalable

def main():
    out_dir = os.path.join(OUTPUT_ROOT, NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Ground stations basic -> extended
    gs_basic = os.path.join(out_dir, "ground_stations_basic.txt")
    gs_ext = os.path.join(out_dir, "ground_stations.txt")
    with open(gs_basic, "w", encoding="utf-8") as f:
        for gid, (nm, lat, lon, elev) in enumerate(ENDPOINTS):
            # Format: gid,name,lat,lon,elevation
            f.write(f"{gid},{nm},{lat},{lon},{elev}\n")
    extend_ground_stations(gs_basic, gs_ext)

    # 2) TLEs
    tles_file = os.path.join(out_dir, "tles.txt")
    generate_tles_from_scratch_with_sgp(
        tles_file,
        "LEO",
        N_ORBITS,
        N_SATS_PER_ORBIT,
        True,       # phase_diff
        53.0,       # inclination_degree
        0.0001,     # eccentricity
        0.0,        # arg_of_perigee_degree
        15.19       # mean_motion_rev_per_day (LEO-ish)
    )

    # 3) ISLs: plus-grid (2 links per sat)
    isls_file = os.path.join(out_dir, "isls.txt")
    generate_plus_grid_isls(isls_file, N_ORBITS, N_SATS_PER_ORBIT, isl_shift=0)

    # 4) GSL interface info (1 per node)
    if_file = os.path.join(out_dir, "gsl_interfaces_info.txt")
    generate_simple_gsl_interfaces_info(
        if_file,
        N_SATS,
        N_ENDPOINTS,
        1,  # num_gsl_interfaces_per_satellite
        1,  # num_gsl_interfaces_per_ground_station
        10.0,  # agg_max_bandwidth_satellite
        10.0   # agg_max_bandwidth_ground_station
    )

    # 5) Description
    desc_file = os.path.join(out_dir, "description.txt")
    generate_description(desc_file, MAX_GSL_M, MAX_ISL_M)

    # 6) Dynamic state (routes/forwarding)
    threads = max(1, (os.cpu_count() or 4) // 2)
    help_dynamic_state(
        OUTPUT_ROOT,
        threads,
        NAME,
        TIME_STEP_MS,
        DURATION_S,
        MAX_GSL_M,
        MAX_ISL_M,
        DYN_ALGO,
        False  # print_logs
    )

    print("\nDONE.")
    print("Network dir:", out_dir)
    print("Dynamic state dir:",
          os.path.join(out_dir, f"dynamic_state_{TIME_STEP_MS}ms_for_{DURATION_S}s"))

if __name__ == "__main__":
    main()

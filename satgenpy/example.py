#!/usr/bin/env python3
import os
from satgen.tles.generate_tles_from_scratch import generate_tles_from_scratch
from satgen.isls.generate_plus_grid_isls import generate_plus_grid_isls
from satgen.ground_stations.extend_ground_stations import extend_ground_stations
from satgen.interfaces.generate_simple_gsl_interfaces_info import generate_simple_gsl_interfaces_info
from satgen.dynamic_state.generate_dynamic_state import generate_dynamic_state
from satgen.description.generate_description import generate_description

# === CONFIGURATION ===
output_dir = "../ns3-sat-sim/simulator/satellite_network_state_large"
os.makedirs(output_dir, exist_ok=True)

num_orbital_planes = 12        # planes
num_sat_per_plane = 10         # satellites per plane (→ 120 total)
altitude_km = 550
inclination_deg = 53
phase_diff = True

# Ground stations
num_gs = 10
base_gs_file = "tests/data_to_match/kuiper_630/ground_stations.txt"

# Simulation timeline
simulation_time_s = 200
state_interval_s = 0.1         # 100 ms steps

print("=== Generating large constellation ===")

# 1. Generate TLEs
generate_tles_from_scratch(
    output_dir,
    "constellation",
    num_orbital_planes,
    num_sat_per_plane,
    altitude_km,
    inclination_deg,
    phase_diff
)

# 2. Generate ISLs (plus-grid pattern)
generate_plus_grid_isls(
    output_dir,
    "constellation",
    num_orbital_planes,
    num_sat_per_plane
)

# 3. Add ground stations
extend_ground_stations(
    base_gs_file,
    num_gs,
    output_dir + "/ground_stations.txt"
)

# 4. Generate GSL interface info
generate_simple_gsl_interfaces_info(
    output_dir,
    "constellation",
    output_dir + "/ground_stations.txt",
    output_dir
)

# 5. Generate dynamic state (routing, snapshots, etc.)
generate_dynamic_state(
    output_dir,
    "constellation",
    simulation_time_s,
    state_interval_s,
    algorithm="algorithm_free_one_only_gs_relays"
)

# 6. Generate description (metadata)
generate_description(
    output_dir,
    "constellation",
    "ground_stations.txt",
    "isls.txt",
    "gsl_interfaces_info.txt"
)

print("✅ Large constellation generated in:", output_dir)


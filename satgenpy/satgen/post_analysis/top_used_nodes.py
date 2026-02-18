from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Tuple, List, Optional

from satgen.tles import read_tles
from satgen.ground_stations import read_ground_stations_extended


def _load_forwarding_updates(fstate_file: str, fwd: Dict[Tuple[int, int], int]) -> None:
    """
    fstate_<t>.txt lines are: current,destination,next_hop,my_if,next_hop_if
    BUT for post-analysis we only need next_hop.
    Files after t=0 are DELTAS (only changed entries), so we update fwd in-place.
    """
    with open(fstate_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spl = line.split(",")
            if len(spl) < 3:
                continue
            current = int(spl[0])
            destination = int(spl[1])
            next_hop = int(spl[2])
            fwd[(current, destination)] = next_hop


def _get_path(src: int, dst: int, fwd: Dict[Tuple[int, int], int], max_hops: int = 4096) -> Optional[List[int]]:
    """
    Reconstruct path using next-hop forwarding pointers:
      curr -> fwd[(curr, dst)] -> ... -> dst
    Returns None if unreachable or if loops occur.
    """
    if (src, dst) not in fwd:
        return None

    nh = fwd[(src, dst)]
    if nh == -1:
        return None

    path = [src]
    curr = src
    seen = {src}

    for _ in range(max_hops):
        nh = fwd.get((curr, dst), -1)
        if nh == -1:
            return None
        path.append(nh)
        if nh == dst:
            return path
        if nh in seen:
            # Loop => treat as invalid/unreachable
            return None
        seen.add(nh)
        curr = nh

    # Too long => likely loop/bug
    return None


def compute_top_used_satellites(
    satellite_network_dir: str,
    dynamic_state_update_interval_ms: int,
    simulation_end_time_s: int,
    top_k: int = 10,
    sample_every_n_steps: int = 1,
    include_endpoints_in_count: bool = False,
) -> List[Tuple[int, int]]:
    """
    Counts how often each SATELLITE node id appears on endpoint->endpoint paths over time.

    - Satellites are node ids: [0 .. num_sats-1]
    - Endpoints (GS + planes) are node ids: [num_sats .. num_sats + num_endpoints - 1]
    """
    tles = read_tles(os.path.join(satellite_network_dir, "tles.txt"))
    satellites = tles["satellites"]
    num_sats = len(satellites)

    endpoints = read_ground_stations_extended(os.path.join(satellite_network_dir, "ground_stations.txt"))
    num_endpoints = len(endpoints)

    dyn_dir = os.path.join(
        satellite_network_dir,
        f"dynamic_state_{dynamic_state_update_interval_ms}ms_for_{simulation_end_time_s}s"
    )
    if not os.path.isdir(dyn_dir):
        raise FileNotFoundError(f"Dynamic state directory not found: {dyn_dir}")

    step_ns = dynamic_state_update_interval_ms * 1_000_000
    end_ns = simulation_end_time_s * 1_000_000_000

    # Forwarding state (persistent, updated by deltas)
    fwd: Dict[Tuple[int, int], int] = {}

    # Counts for satellite node ids only
    sat_count: Counter[int] = Counter()

    endpoint_node_ids = list(range(num_sats, num_sats + num_endpoints))

    step_index = 0
    for t in range(0, end_ns, step_ns):
        fpath = os.path.join(dyn_dir, f"fstate_{t}.txt")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing fstate file: {fpath}")

        _load_forwarding_updates(fpath, fwd)

        # sampling knob (for speed if needed)
        if (step_index % sample_every_n_steps) != 0:
            step_index += 1
            continue

        # count usage across ALL ordered endpoint pairs
        for src in endpoint_node_ids:
            for dst in endpoint_node_ids:
                if src == dst:
                    continue
                path = _get_path(src, dst, fwd)
                if path is None:
                    continue

                for node in path:
                    if (not include_endpoints_in_count) and (node == src or node == dst):
                        continue
                    if 0 <= node < num_sats:
                        sat_count[node] += 1

        step_index += 1

    return sat_count.most_common(top_k)

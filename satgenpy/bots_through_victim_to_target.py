#!/usr/bin/env python3
import os
import sys
from collections import Counter, defaultdict

from satgen.tles import read_tles
from satgen.ground_stations import read_ground_stations_extended

# ---- Safe path reconstruction ----
def get_path_safe(src, dst, forward_state):
    """
    Reconstruct path using forward_state[(cur, dst)] = next_hop.
    Returns list of nodes [src, ..., dst] or None if unreachable / missing entries.
    """
    if (src, dst) not in forward_state:
        return None
    if forward_state[(src, dst)] == -1:
        return None

    curr = src
    path = [src]
    seen = set([src])

    # Guard against loops / corrupted state
    for _ in range(10000):
        if curr == dst:
            return path

        key = (curr, dst)
        if key not in forward_state:
            return None

        nh = forward_state[key]
        if nh == -1:
            return None

        path.append(nh)
        curr = nh

        if curr in seen:
            # loop detected
            return None
        seen.add(curr)

    # too long => treat as invalid
    return None


def load_fstate_at(fstate_file):
    fstate = {}
    with open(fstate_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spl = line.split(",")
            if len(spl) < 3:
                continue
            cur = int(spl[0])
            dst = int(spl[1])
            nh  = int(spl[2])
            fstate[(cur, dst)] = nh
    return fstate


def list_fstate_times_ns(dyn_dir):
    """
    Returns sorted list of t_ns for which fstate_<t>.txt exists.
    """
    times = []
    for fn in os.listdir(dyn_dir):
        if fn.startswith("fstate_") and fn.endswith(".txt"):
            # fstate_100000000.txt
            mid = fn[len("fstate_"):-len(".txt")]
            if mid.isdigit():
                times.append(int(mid))
    times.sort()
    return times


def ns_to_s(t_ns):
    return t_ns / 1_000_000_000.0


def main():
    if len(sys.argv) != 7:
        print("Usage:")
        print("  python3 bots_through_victim_to_target.py <sat_net_dir> <update_ms> <end_s> <victim_sat_id> <target_endpoint_node_id> <topK>")
        print("")
        print("Example:")
        print("  python3 bots_through_victim_to_target.py ../.../satellite_network_state 100 200 3 120 10")
        sys.exit(1)

    sat_net_dir = sys.argv[1].rstrip("/")
    update_ms   = int(sys.argv[2])
    end_s       = int(sys.argv[3])
    victim_sat  = int(sys.argv[4])
    target_node = int(sys.argv[5])
    topk        = int(sys.argv[6])

    # Read sizes
    tles = read_tles(os.path.join(sat_net_dir, "tles.txt"))
    n_sats = len(tles["satellites"])
    gss = read_ground_stations_extended(os.path.join(sat_net_dir, "ground_stations.txt"))
    n_endpoints = len(gss)

    if not (0 <= victim_sat < n_sats):
        raise ValueError(f"victim_sat_id must be in [0, {n_sats-1}]")

    if target_node < n_sats or target_node >= n_sats + n_endpoints:
        raise ValueError(f"target_endpoint_node_id must be in [{n_sats}, {n_sats+n_endpoints-1}]")

    endpoint_nodes = [n_sats + gid for gid in range(n_endpoints)]
    dyn_dir = os.path.join(sat_net_dir, f"dynamic_state_{update_ms}ms_for_{end_s}s")

    if not os.path.isdir(dyn_dir):
        raise FileNotFoundError(f"Dynamic state dir not found: {dyn_dir}")

    times_ns = list_fstate_times_ns(dyn_dir)
    if not times_ns:
        raise FileNotFoundError(f"No fstate_*.txt found in {dyn_dir}")

    # Optional: you can enforce end_s boundary; but using actual files is safer
    # Keep only those <= end_s
    max_ns = end_s * 1_000_000_000
    times_ns = [t for t in times_ns if t < max_ns]

    # Stats
    reachable = Counter()  # src -> count
    includes  = Counter()  # src -> count

    # Store a few example times for debugging / scheduling
    reachable_times = defaultdict(list)  # src -> [t_ns...]
    include_times   = defaultdict(list)  # src -> [t_ns...]

    missing_files = 0
    processed_steps = 0

    for t_ns in times_ns:
        fpath = os.path.join(dyn_dir, f"fstate_{t_ns}.txt")
        if not os.path.isfile(fpath):
            missing_files += 1
            continue

        fstate = load_fstate_at(fpath)
        processed_steps += 1

        for src in endpoint_nodes:
            if src == target_node:
                continue

            path = get_path_safe(src, target_node, fstate)
            if path is None:
                continue

            reachable[src] += 1
            if len(reachable_times[src]) < 10:
                reachable_times[src].append(t_ns)

            if victim_sat in path:
                includes[src] += 1
                if len(include_times[src]) < 10:
                    include_times[src].append(t_ns)

    print(f"Victim sat V = {victim_sat}")
    print(f"Target endpoint T = {target_node} (endpoint node ids are {n_sats}..{n_sats+n_endpoints-1})")
    print(f"Dynamic state dir: {dyn_dir}")
    print("")
    print(f"Processed fstate steps: {processed_steps} (missing: {missing_files})")
    print("")

    # ---- 1) Top by reachability ----
    by_reach = [(reachable[src], src) for src in endpoint_nodes if src != target_node and reachable[src] > 0]
    by_reach.sort(reverse=True)

    print(f"Top {topk} bot endpoints by REACHABILITY to T:")
    if not by_reach:
        print("  (none reachable at any timestep)")
    else:
        for r, src in by_reach[:topk]:
            ex = reachable_times[src]
            ex_s = [f"{ns_to_s(x):.1f}s" for x in ex]
            print(f"  bot {src:3d} : reachable_steps={r:4d}  example_times={ex_s[:5]}")

    print("")

    # ---- 2) Top by include fraction (but show counts) ----
    scored = []
    for src in endpoint_nodes:
        if src == target_node:
            continue
        r = reachable[src]
        if r == 0:
            continue
        inc = includes[src]
        frac = inc / r
        scored.append((frac, inc, r, src))
    scored.sort(reverse=True)

    print(f"Top {topk} bot endpoints whose route to T includes V most often:")
    if not scored:
        print("  (none reachable at any timestep)")
    else:
        for frac, inc, r, src in scored[:topk]:
            ex_inc = include_times[src]
            ex_inc_s = [f"{ns_to_s(x):.1f}s" for x in ex_inc]
            print(f"  bot {src:3d} : include_frac={frac:.3f}  (includes {inc}/{r} reachable timesteps)  include_times={ex_inc_s[:5]}")

    print("")

    # ---- 3) Extra: how many endpoints ever reach target at all ----
    ever_reach = sum(1 for src in endpoint_nodes if src != target_node and reachable[src] > 0)
    print(f"Endpoints that can reach T at least once: {ever_reach}/{len(endpoint_nodes)-1}")


if __name__ == "__main__":
    main()

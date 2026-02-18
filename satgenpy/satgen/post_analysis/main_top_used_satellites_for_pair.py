import sys
from collections import Counter, defaultdict

from satgen.tles import read_tles
from satgen.ground_stations import read_ground_stations_extended
from satgen.post_analysis.graph_tools import get_path


def node_label(n: int, num_sats: int, gs_list):
    """
    Pretty label for a node id.
    - satellites: SAT_<id>
    - endpoints:  EP_<gid>(<name>)
    """
    if 0 <= n < num_sats:
        return f"SAT_{n}"
    gid = n - num_sats
    if 0 <= gid < len(gs_list):
        nm = gs_list[gid].get("name", f"GS{gid}")
        return f"EP_{gid}({nm})"
    return f"NODE_{n}"


def path_to_string(path, num_sats, gs_list):
    return " -> ".join(node_label(x, num_sats, gs_list) for x in path)


def main():
    args = sys.argv[1:]
    if len(args) != 6:
        print(
            "Usage:\n"
            "  python3 -m satgen.post_analysis.main_top_used_satellites_for_pair "
            "<satellite_network_dir> <update_ms> <end_time_s> <src_node_id> <dst_node_id> <topk>\n\n"
            "Example:\n"
            "  python3 -m satgen.post_analysis.main_top_used_satellites_for_pair "
            "../ns3-sat-sim/satellite_network_state/leo_120sats_22endpoints 100 200 120 121 10\n"
        )
        sys.exit(1)

    satellite_network_dir = args[0].rstrip("/")
    update_ms = int(args[1])
    end_time_s = int(args[2])
    src = int(args[3])
    dst = int(args[4])
    topk = int(args[5])

    # Load topology sizes
    tles = read_tles(satellite_network_dir + "/tles.txt")
    satellites = tles["satellites"]
    gs_list = read_ground_stations_extended(satellite_network_dir + "/ground_stations.txt")

    num_sats = len(satellites)
    num_eps = len(gs_list)
    num_nodes = num_sats + num_eps

    if not (0 <= src < num_nodes) or not (0 <= dst < num_nodes):
        raise SystemExit(f"src/dst out of range: valid node ids are 0..{num_nodes-1}")

    dyn_dir = f"{satellite_network_dir}/dynamic_state_{update_ms}ms_for_{end_time_s}s"

    step_ns = update_ms * 1_000_000
    end_ns = end_time_s * 1_000_000_000

    # forwarding state (deltas applied over time)
    fstate = {}

    # satellite usage count across reachable timesteps
    sat_counts = Counter()

    # unique path counts
    path_counts = Counter()

    total_steps = 0
    reachable_steps = 0

    last_path = None

    print("\n=== PATH TRACE (prints only when path changes) ===")
    print(f"Pair: {node_label(src, num_sats, gs_list)}  ->  {node_label(dst, num_sats, gs_list)}")
    print(f"Dynamic state dir: {dyn_dir}\n")

    for t in range(0, end_ns, step_ns):
        total_steps += 1
        ffile = f"{dyn_dir}/fstate_{t}.txt"

        # Apply delta updates for this timestep
        with open(ffile, "r") as f:
            for line in f:
                spl = line.strip().split(",")
                if len(spl) < 3:
                    continue
                cur = int(spl[0])
                dst_id = int(spl[1])
                nxt = int(spl[2])
                fstate[(cur, dst_id)] = nxt

        # Compute path for THIS pair at time t
        if (src, dst) not in fstate:
            continue

        path = get_path(src, dst, fstate)
        if path is None:
            continue

        reachable_steps += 1

        # Count satellites on the path (exclude endpoints)
        for node in path[1:-1]:
            if 0 <= node < num_sats:
                sat_counts[node] += 1

        # Count this exact path
        path_key = tuple(path)
        path_counts[path_key] += 1

        # Print only if changed
        if last_path != path_key:
            time_s = t / 1_000_000_000
            print(f"[t={time_s:8.3f}s] {path_to_string(path, num_sats, gs_list)}")
            last_path = path_key

    print("\n=== SUMMARY ===")
    print(f"Timesteps total: {total_steps}")
    print(f"Timesteps reachable: {reachable_steps} "
          f"({(reachable_steps/total_steps*100.0 if total_steps else 0.0):.2f}%)")

    if reachable_steps == 0:
        print("Never reachable for this pair (no path).")
        return

    print("\n=== Top satellites on the path over time ===")
    for rank, (sid, c) in enumerate(sat_counts.most_common(topk), start=1):
        per_step = c / float(reachable_steps)
        print(f"{rank:>2}. SAT_{sid:<4}  count={c:<10}  avg_uses_per_reachable_step={per_step:.6f}")

    print("\n=== Unique paths (top 10 by frequency) ===")
    for rank, (pkey, c) in enumerate(path_counts.most_common(10), start=1):
        pct = (c / float(reachable_steps)) * 100.0
        print(f"{rank:>2}. occurrences={c} ({pct:.2f}%)")
        print(f"    {path_to_string(list(pkey), num_sats, gs_list)}")


if __name__ == "__main__":
    main()

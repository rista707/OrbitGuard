#!/usr/bin/env python3
"""
DDoS planner for Hypatia / ns3-sat-sim dynamic_state routes.

What it does:
- Loads forwarding-state snapshots from a dynamic_state_* directory
- Reconstructs paths for flows (bots -> target)
- Counts per-satellite usage across all paths and times
- Picks a victim satellite (top-1 by default)
- Computes per-bot "coverage": fraction of timesteps where path(bot->target) contains victim sat

Assumptions (matches your setup):
- Satellites have node IDs: [0 .. n_sats-1]
- Endpoints (ground stations + planes) have node IDs: [n_sats .. n_sats+n_endpoints-1]
- Forwarding state files are text with 3 integers per line:
    src_id  dst_id  next_hop_id
  (comma or whitespace separated)
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, Tuple, List, Optional


Triple = Tuple[int, int, int]  # (src, dst, next_hop)
NextHopMap = Dict[Tuple[int, int], int]


def _is_comment_or_empty(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#") or s.startswith("//")


def parse_forwarding_state_file(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if _is_comment_or_empty(line):
                continue
            # split by comma or whitespace
            parts = re.split(r"[,\s]+", line.strip())
            if len(parts) < 3:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
                c = int(parts[2])
            except ValueError:
                continue
            triples.append((a, b, c))
    return triples


def triples_to_nexthop(triples: List[Triple]) -> NextHopMap:
    nh: NextHopMap = {}
    for src, dst, nxt in triples:
        nh[(src, dst)] = nxt
    return nh


def reconstruct_path(src: int, dst: int, nh: NextHopMap, max_hops: int) -> Optional[List[int]]:
    """
    Follow next-hops until reaching dst, or bail out on loop / missing entry.
    Returns list of node IDs including src and dst, or None if unreachable.
    """
    if src == dst:
        return [src]

    path = [src]
    cur = src
    seen = set([src])

    for _ in range(max_hops):
        key = (cur, dst)
        if key not in nh:
            return None
        nxt = nh[key]
        path.append(nxt)
        if nxt == dst:
            return path
        if nxt in seen:
            return None
        seen.add(nxt)
        cur = nxt

    return None


def discover_route_files(routes_dir: str) -> List[Tuple[float, str]]:
    """
    Recursively find forwarding-state snapshot files and attach a timestamp.

    We support filenames containing a number (e.g., fstate_100000000.txt or 100.txt).
    We infer whether that number is in ns or in seconds:
      - If max timestamp > 1e6, treat as ns and convert to seconds.
      - Else treat as seconds.

    Returns: list of (t_seconds, filepath) sorted by time.
    """
    candidates: List[Tuple[int, str]] = []
    num_re = re.compile(r"(\d+)")

    for root, _, files in os.walk(routes_dir):
        for fn in files:
            # Heuristic: ignore non-text
            if not (fn.endswith(".txt") or fn.endswith(".csv") or fn.endswith(".dat")):
                continue
            m = num_re.findall(fn)
            if not m:
                continue
            # take the LAST number in filename (usually the timestamp)
            ts_int = int(m[-1])
            candidates.append((ts_int, os.path.join(root, fn)))

    if not candidates:
        raise FileNotFoundError(f"No route snapshot files found under: {routes_dir}")

    candidates.sort(key=lambda x: x[0])
    max_ts = candidates[-1][0]

    # infer unit
    if max_ts > 1_000_000:  # likely ns
        out = [(ts / 1e9, p) for ts, p in candidates]
    else:
        out = [(float(ts), p) for ts, p in candidates]

    out.sort(key=lambda x: x[0])
    return out


def filter_times(route_files: List[Tuple[float, str]], t0: float, t1: float) -> List[Tuple[float, str]]:
    return [(t, p) for (t, p) in route_files if (t0 <= t <= t1)]


def plan_ddos(
    routes_dir: str,
    n_sats: int,
    n_endpoints: int,
    target_endpoint_id: int,
    t0: float,
    t1: float,
    top_sats: int,
    top_bots: int,
    victim_sat_id: Optional[int] = None,
    bots_subset: Optional[List[int]] = None,
) -> None:
    n_nodes = n_sats + n_endpoints
    endpoints = list(range(n_sats, n_nodes))

    if target_endpoint_id < n_sats or target_endpoint_id >= n_nodes:
        raise ValueError(f"target_endpoint_id={target_endpoint_id} must be in [{n_sats}, {n_nodes-1}]")

    # choose bot candidates
    if bots_subset is None:
        bot_candidates = [e for e in endpoints if e != target_endpoint_id]
    else:
        # keep only valid endpoints
        bot_candidates = [b for b in bots_subset if n_sats <= b < n_nodes and b != target_endpoint_id]
        if not bot_candidates:
            raise ValueError("bots_subset produced empty bot_candidates after filtering.")

    route_files_all = discover_route_files(routes_dir)
    route_files = filter_times(route_files_all, t0, t1)
    if not route_files:
        raise ValueError(f"No route snapshots in time window [{t0}, {t1}] seconds.")

    # Usage counts
    sat_use = defaultdict(int)          # sat_id -> count occurrences on paths
    sat_use_as_transit = defaultdict(int)  # exclude endpoints; also exclude src/dst if they are satellites (not here)
    total_paths = 0
    total_unreachable = 0

    # For coverage later
    victim_hits = defaultdict(int)  # bot_id -> timesteps where victim sat appears
    victim_total = defaultdict(int)  # bot_id -> timesteps considered (where path exists)
    timesteps_considered = 0

    max_hops = 2 * n_nodes

    # First pass: count satellite usage across all bot->target paths
    for (t_sec, fp) in route_files:
        triples = parse_forwarding_state_file(fp)
        nh = triples_to_nexthop(triples)

        timesteps_considered += 1

        for b in bot_candidates:
            path = reconstruct_path(b, target_endpoint_id, nh, max_hops=max_hops)
            if path is None:
                total_unreachable += 1
                continue

            total_paths += 1
            victim_total[b] += 1

            # Count satellites on this path
            for node in path:
                if 0 <= node < n_sats:
                    sat_use[node] += 1

            # Transit-only: exclude endpoints (already excluded), but also exclude if ever sat endpoints (not your case)
            # Here bots and target are endpoints, so all satellites in path are transit.
            for node in path[1:-1]:
                if 0 <= node < n_sats:
                    sat_use_as_transit[node] += 1

    if total_paths == 0:
        print("No reachable paths found in the specified window.")
        print(f"Unreachable count: {total_unreachable}")
        return

    # Choose victim satellite
    if victim_sat_id is None:
        if not sat_use_as_transit:
            # fallback to sat_use
            victim_sat_id = max(sat_use.items(), key=lambda kv: kv[1])[0]
        else:
            victim_sat_id = max(sat_use_as_transit.items(), key=lambda kv: kv[1])[0]

    # Second pass: compute coverage(b) for victim sat
    # (We could have done this during the first pass, but we needed victim sat first.)
    for (t_sec, fp) in route_files:
        triples = parse_forwarding_state_file(fp)
        nh = triples_to_nexthop(triples)

        for b in bot_candidates:
            path = reconstruct_path(b, target_endpoint_id, nh, max_hops=max_hops)
            if path is None:
                continue

            if victim_sat_id in path:
                victim_hits[b] += 1

    # Print results
    print("\n=== DDoS Planner Results ===")
    print(f"Routes dir: {routes_dir}")
    print(f"Time window: [{t0}, {t1}] s  (snapshots considered: {len(route_files)})")
    print(f"n_sats={n_sats}, n_endpoints={n_endpoints}, n_nodes={n_nodes}")
    print(f"Target endpoint id: {target_endpoint_id}")
    print(f"Bot candidates: {len(bot_candidates)} endpoints")
    print(f"Total reachable bot->target paths counted: {total_paths}")
    print(f"Total unreachable bot->target attempts: {total_unreachable}")

    # Top satellites by transit usage
    print("\nTop satellites by TRANSIT usage (most meaningful for 'critical gateway/satellite'):")
    top_list = sorted(sat_use_as_transit.items(), key=lambda kv: kv[1], reverse=True)[:top_sats]
    for rank, (sid, cnt) in enumerate(top_list, start=1):
        print(f"  {rank:2d}) sat {sid:4d}  transit_count={cnt}")

    print(f"\nChosen victim satellite S* = {victim_sat_id}")

    # Top bots by coverage
    # coverage = victim_hits[b] / victim_total[b]
    bot_cov = []
    for b in bot_candidates:
        denom = victim_total[b]
        if denom == 0:
            continue
        cov = victim_hits[b] / denom
        bot_cov.append((cov, b, victim_hits[b], denom))

    bot_cov.sort(reverse=True, key=lambda x: x[0])

    print("\nTop bots by coverage(b) = fraction of timesteps where path(bot->target) contains S*:")
    for rank, (cov, b, hit, denom) in enumerate(bot_cov[:top_bots], start=1):
        print(f"  {rank:2d}) bot {b:4d}  coverage={cov:.3f}  (hits={hit}, steps={denom})")

    # Optional: print bots with zero coverage (useful debugging)
    zeros = [b for (cov, b, _, denom) in bot_cov if denom > 0 and cov == 0.0]
    if zeros:
        print(f"\nBots with ZERO coverage w.r.t S* (count={len(zeros)}). Example first 10: {zeros[:10]}")


def parse_args():
    ap = argparse.ArgumentParser(description="Compute top-used satellites and best bots for victim traversal.")
    ap.add_argument("routes_dir", help="Path to dynamic_state_* directory (or its parent).")
    ap.add_argument("--n_sats", type=int, required=True, help="Number of satellites (e.g., 120).")
    ap.add_argument("--n_endpoints", type=int, required=True, help="Number of endpoints (ground+planes) (e.g., 22).")
    ap.add_argument("--target", type=int, required=True, help="Target ENDPOINT node id (e.g., 120 for first endpoint).")
    ap.add_argument("--t0", type=float, default=0.0, help="Start time in seconds (inclusive).")
    ap.add_argument("--t1", type=float, required=True, help="End time in seconds (inclusive).")
    ap.add_argument("--top_sats", type=int, default=10, help="How many top satellites to print.")
    ap.add_argument("--top_bots", type=int, default=10, help="How many top bots to print.")
    ap.add_argument("--victim_sat", type=int, default=None, help="Pin victim satellite id (optional).")
    ap.add_argument("--bots", type=str, default=None,
                    help="Optional comma-separated bot endpoint IDs to restrict to (e.g., '121,122,130').")
    return ap.parse_args()


def main():
    args = parse_args()
    bots_subset = None
    if args.bots:
        bots_subset = [int(x.strip()) for x in args.bots.split(",") if x.strip()]

    plan_ddos(
        routes_dir=args.routes_dir,
        n_sats=args.n_sats,
        n_endpoints=args.n_endpoints,
        target_endpoint_id=args.target,
        t0=args.t0,
        t1=args.t1,
        top_sats=args.top_sats,
        top_bots=args.top_bots,
        victim_sat_id=args.victim_sat,
        bots_subset=bots_subset,
    )


if __name__ == "__main__":
    main()

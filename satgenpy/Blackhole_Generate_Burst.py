#!/usr/bin/env python3
from __future__ import annotations
import sys
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# from satgen.post_analysis.graph_tools import get_path  # uses fstate[(cur,dst)] = next_hop

from satgen.post_analysis.graph_tools import get_path
# =========================
# CONFIG
# =========================

# Output location
OUT_ROOT = Path("../ns3-sat-sim/simulator/test_data/end_to_end/attacks/Blackhole")

# Where your dynamic routing states live (must contain dynamic_state_100ms_for_200s/)
SATNET_ROOT = Path("../ns3-sat-sim/simulator/test_data/end_to_end/satellite_network_state")

N_SATS = 120
N_ENDPOINTS = 22  # endpoints are node IDs 120..141 when N_SATS=120
ENDPOINTS = list(range(N_SATS, N_SATS + N_ENDPOINTS))  # 120..141

UPDATE_MS = 100
DURATION_S = 200

SIM_END_NS = 200_000_000_000

ATTACK_START_S = 50.0
ATTACK_END_S = 200.0

# We will generate 512 bursts per target satellite
N_BURSTS = 512

# You asked for 10 targets. Default list (same style you used earlier)
REQUESTED_TARGET_SATS = [14, 26, 38, 50, 61, 72, 84, 97, 103, 109]

# If any requested target has ZERO impacted candidates, automatically substitute
AUTO_SUBSTITUTE_IF_EMPTY = True

# Traffic characteristics (keep benign-ish; blackhole effect comes from loss)
RATES_MBPS = [0.5, 1.0, 2.0, 5.0]
DUR_S_MIN = 2.0
DUR_S_MAX = 20.0

# Prime seeds (widely spread)
MASTER_SEED = 20260208
MIN_SEED_GAP = 50_000_000
SEED_LO = 100_000_000
SEED_HI = 2_147_483_647


# =========================
# PRIME SEEDS (widely spread)
# =========================

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Miller-Rabin (enough for this range)
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    def modpow(a: int, e: int, mod: int) -> int:
        r = 1
        a %= mod
        while e:
            if e & 1:
                r = (r * a) % mod
            a = (a * a) % mod
            e >>= 1
        return r

    for a in (2, 3, 5, 7, 11):
        x = modpow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _next_prime(x: int, hi: int) -> int:
    if x <= 2:
        return 2
    if x % 2 == 0:
        x += 1
    while x <= hi and not _is_prime(x):
        x += 2
    if x > hi:
        raise RuntimeError("prime search overflow")
    return x

def spread_primes(n: int, master_seed: int) -> List[int]:
    rng = random.Random(master_seed)
    seeds: List[int] = []
    tries = 0
    while len(seeds) < n:
        tries += 1
        if tries > 200_000:
            raise RuntimeError("Could not find enough spaced primes; reduce MIN_SEED_GAP or widen range.")
        p = _next_prime(rng.randrange(SEED_LO, SEED_HI), SEED_HI)
        if all(abs(p - s) >= MIN_SEED_GAP for s in seeds):
            seeds.append(p)
    rng.shuffle(seeds)
    return seeds


# =========================
# UTIL
# =========================

def s_to_ns(s: float) -> int:
    return int(round(s * 1_000_000_000))

def write_threat_model(run_dir: Path, sat_id: int) -> None:
    (run_dir / "threat_model.csv").write_text(
f"""# attack_name,attack_type,target_type,target_ids,start_time_s,end_time_s,params
blackhole_sat{sat_id},node_blackhole,node,{sat_id},{ATTACK_START_S},{ATTACK_END_S},
"""
    )

def write_config(run_dir: Path, seed: int) -> None:
    log_ids = ",".join(str(i) for i in range(N_BURSTS))
    (run_dir / "config_ns3.properties").write_text(
f"""simulation_end_time_ns={SIM_END_NS}
simulation_seed={seed}

satellite_network_dir="../../../satellite_network_state"
satellite_network_routes_dir="../../../satellite_network_state/dynamic_state_100ms_for_200s"
dynamic_state_update_interval_ns=100000000

isl_data_rate_megabit_per_s=10.0
gsl_data_rate_megabit_per_s=10.0
isl_max_queue_size_pkts=100
gsl_max_queue_size_pkts=100

enable_isl_utilization_tracking=true
isl_utilization_tracking_interval_ns=1000000000

enable_udp_burst_scheduler=true
udp_burst_schedule_filename="udp_burst_schedule.csv"
udp_burst_enable_logging_for_udp_burst_ids=set(0)

tcp_socket_type=TcpNewReno

threat_model_filename="threat_model.csv"
"""
    )

def load_candidates_by_satellite() -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Build a mapping:
      sat_id -> list of (t_ns_step, src_endpoint, dst_endpoint)
    where sat_id is on the src->dst path at that time.

    We scan fstate from t=0 to end, apply deltas, and only record within [ATTACK_START, ATTACK_END).
    """
    dyn_dir = SATNET_ROOT / f"dynamic_state_{UPDATE_MS}ms_for_{DURATION_S}s"
    if not dyn_dir.exists():
        raise SystemExit(f"Dynamic state dir not found: {dyn_dir}")

    step_ns = UPDATE_MS * 1_000_000
    end_ns = DURATION_S * 1_000_000_000

    attack_start_ns = s_to_ns(ATTACK_START_S)
    attack_end_ns = s_to_ns(min(ATTACK_END_S, float(DURATION_S)))

    sat_to = defaultdict(list)  # sat -> [(t_ns, src, dst), ...]

    fstate: Dict[Tuple[int, int], int] = {}

    for t in range(0, end_ns, step_ns):
        ffile = dyn_dir / f"fstate_{t}.txt"
        if not ffile.exists():
            # if your dyn dir has all files, this won't happen; skip otherwise
            continue

        # apply delta updates for this timestep
        with ffile.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                spl = line.split(",")
                if len(spl) < 3:
                    continue
                cur = int(spl[0])
                dst = int(spl[1])
                nxt = int(spl[2])
                fstate[(cur, dst)] = nxt

        # record only inside attack window
        if not (attack_start_ns <= t < attack_end_ns):
            continue

        # evaluate all endpoint pairs (462 pairs)
        for src in ENDPOINTS:
            for dst in ENDPOINTS:
                if src == dst:
                    continue
                if (src, dst) not in fstate:
                    continue
                path = get_path(src, dst, fstate)
                if path is None:
                    continue

                # count satellites on the path
                for node in path[1:-1]:
                    if 0 <= node < N_SATS:
                        sat_to[node].append((t, src, dst))

    return sat_to


def make_schedule_for_target(sat_id: int, seed: int, candidates: List[Tuple[int, int, int]]) -> List[str]:
    """
    Generate N_BURSTS bursts by sampling from (t_step, src, dst) candidates
    where sat_id is on-path at that time step.

    Output is sorted by start_time_ns ascending, and burst_id follows that order.
    """
    rng = random.Random(seed)

    step_ns = UPDATE_MS * 1_000_000
    bursts = []  # (start_ns, end_ns, src, dst, rate)

    for _ in range(N_BURSTS):
        t_step_ns, src, dst = candidates[rng.randrange(0, len(candidates))]

        # jitter within step so we get nicer spread but same routing state
        jitter = rng.randrange(0, step_ns)
        start_ns = t_step_ns + jitter

        dur_s = rng.uniform(DUR_S_MIN, DUR_S_MAX)
        end_ns = min(start_ns + s_to_ns(dur_s), SIM_END_NS)

        rate = rng.choice(RATES_MBPS)
        bursts.append((start_ns, end_ns, src, dst, rate))

    # sort by time ascending (and stable tie-breakers)
    bursts.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    # assign burst ids in sorted order
    lines: List[str] = []
    for bid, (start_ns, end_ns, src, dst, rate) in enumerate(bursts):
        lines.append(f"{bid},{src},{dst},{rate},{start_ns},{end_ns},,\n")

    return lines


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # sanity: targets must be satellites
    for s in REQUESTED_TARGET_SATS:
        if not (0 <= s < N_SATS):
            raise SystemExit(f"Requested target {s} is not a satellite id (0..{N_SATS-1}).")

    print("[+] Scanning dynamic fstate to find impacted (src,dst,time) candidates per satellite...")
    sat_to_candidates = load_candidates_by_satellite()

    # rank satellites by how many candidates they appear in
    ranked = sorted(((len(v), s) for s, v in sat_to_candidates.items()), reverse=True)
    top_available = [s for (c, s) in ranked if c > 0]

    # build final target list (10 sats)
    targets = list(REQUESTED_TARGET_SATS)
    if AUTO_SUBSTITUTE_IF_EMPTY:
        used = set(targets)
        for i, s in enumerate(targets):
            if len(sat_to_candidates.get(s, [])) == 0:
                # substitute with a strong, frequently-used satellite not already used
                repl = next((x for x in top_available if x not in used), None)
                if repl is None:
                    raise SystemExit("Could not find enough satellites with non-empty candidate sets.")
                print(f"[!] Target SAT_{s} has 0 candidates in attack window; substituting with SAT_{repl}")
                used.discard(s)
                targets[i] = repl
                used.add(repl)

    # ensure we have 10
    if len(targets) != 10:
        raise SystemExit(f"Need 10 targets, got {len(targets)}")

    # seeds (one per target)
    seeds = spread_primes(len(targets), MASTER_SEED)

    for sat_id, seed in zip(targets, seeds):
        candidates = sat_to_candidates.get(sat_id, [])
        if not candidates:
            raise SystemExit(f"SAT_{sat_id} has no impacted candidates (even after substitution).")

        run_dir = OUT_ROOT / f"run_blackhole_sat{sat_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[+] Creating {run_dir} (seed={seed})  candidates={len(candidates)}")

        write_threat_model(run_dir, sat_id)

        schedule_lines = make_schedule_for_target(sat_id, seed, candidates)
        (run_dir / "udp_burst_schedule.csv").write_text("".join(schedule_lines), encoding="utf-8")

        write_config(run_dir, seed)

        # quick check count
        n_lines = sum(1 for _ in (run_dir / "udp_burst_schedule.csv").open("r", encoding="utf-8"))
        if n_lines != N_BURSTS:
            raise RuntimeError(f"Expected {N_BURSTS} bursts, got {n_lines} for SAT_{sat_id}")

        # also confirm sorted
        # (lightweight check: parse starts and ensure nondecreasing)
        starts = []
        with (run_dir / "udp_burst_schedule.csv").open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                starts.append(int(parts[4]))
        if any(starts[i] > starts[i+1] for i in range(len(starts)-1)):
            raise RuntimeError(f"Schedule is not sorted by start time for SAT_{sat_id}")

        print(f"    - wrote {n_lines} bursts sorted by time (impacted by SAT_{sat_id})")

    print("[✓] Done. Generated 10 blackhole runs with 512 impacted bursts each.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple


def s_to_ns(s: float) -> int:
    return int(round(s * 1_000_000_000))

def hits_sbad_when_forwarding(src: int, dest: int, s_bad: int, fstate, max_hops: int = 300) -> bool:
    curr = src
    seen = {src}
    for _ in range(max_hops):
        if curr == s_bad:
            return True
        if curr == dest:
            return False
        nh = fstate.get((curr, dest), -1)
        if nh == -1 or nh in seen:
            return False
        seen.add(nh)
        curr = nh
    return False




def get_path_safe(src: int, dst: int, fstate: Dict[Tuple[int, int], int], max_hops: int = 300):
    """
    Loop-guarded reconstruction using fstate[(cur,dst)] = next_hop.
    Returns [src,...,dst] or None if unreachable / loop / missing.
    """
    key = (src, dst)
    if key not in fstate:
        return None
    if fstate[key] == -1:
        return None

    curr = src
    path = [src]
    seen = {src}

    for _ in range(max_hops):
        if curr == dst:
            return path

        nh = fstate.get((curr, dst), -1)
        if nh == -1:
            return None

        # loop guard
        if nh in seen:
            return None

        path.append(nh)
        seen.add(nh)
        curr = nh

    return None  # too long => treat as invalid


def load_fstate_deltas_and_collect_candidates(
    routes_dir: Path,
    n_sats: int,
    n_endpoints: int,
    update_ns: int,
    duration_ns: int,
    t_start_ns: int,
    t_end_ns: int,
    s_bad: int,
    dest: int,
    allow_sat_sources: bool,
    candidate_cap: int,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Returns:
      candidates: list of (t_step_ns, src) such that path(src->dest) includes s_bad at that timestep
      processed_steps: number of timesteps processed
    """
    endpoints = list(range(n_sats, n_sats + n_endpoints))
    sats = list(range(0, n_sats))

    if allow_sat_sources:
        sources = sats + [e for e in endpoints if e != dest]
    else:
        sources = [e for e in endpoints if e != dest]

    fstate: Dict[Tuple[int, int], int] = {}
    candidates: List[Tuple[int, int]] = []

    processed = 0
    for t in range(0, duration_ns, update_ns):
        ffile = routes_dir / f"fstate_{t}.txt"
        if not ffile.exists():
            continue

        # apply delta updates
        with ffile.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                spl = line.split(",")
                if len(spl) < 3:
                    continue
                cur = int(spl[0]); dst = int(spl[1]); nxt = int(spl[2])
                fstate[(cur, dst)] = nxt

        processed += 1

        if not (t_start_ns <= t < t_end_ns):
            continue

        # only src->dest paths
        for src in sources:
            if (src, dest) not in fstate:
                continue

            if hits_sbad_when_forwarding(src, dest, s_bad, fstate):
                candidates.append((t, src))

    return candidates, processed


def build_schedule(
    candidates: List[Tuple[int, int]],
    seed: int,
    n_bursts: int,
    dest: int,
    update_ns: int,
    sim_end_ns: int,
    dur_s_min: float,
    dur_s_max: float,
    rates_mbps: List[float],
) -> List[str]:
    rng = random.Random(seed)

    bursts = []  # (start_ns, end_ns, src, rate)
    for _ in range(n_bursts):
        t_step_ns, src = candidates[rng.randrange(0, len(candidates))]
        jitter = rng.randrange(0, update_ns)
        start_ns = t_step_ns

        dur_s = rng.uniform(dur_s_min, dur_s_max)
        end_ns = min(start_ns + s_to_ns(dur_s), sim_end_ns)

        rate = rng.choice(rates_mbps)
        bursts.append((start_ns, end_ns, src, rate))

    # sort ascending by time
    bursts.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    lines: List[str] = []
    for bid, (start_ns, end_ns, src, rate) in enumerate(bursts):
        lines.append(f"{bid},{src},{dest},{rate},{start_ns},{end_ns},,\n")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes_dir", required=True, type=Path)
    ap.add_argument("--n_sats", type=int, required=True)
    ap.add_argument("--n_endpoints", type=int, required=True)
    ap.add_argument("--dest", type=int, required=True)
    ap.add_argument("--s_bad", type=int, required=True)

    ap.add_argument("--update_ns", type=int, default=100_000_000)
    ap.add_argument("--duration_s", type=int, default=200)

    ap.add_argument("--t_start_s", type=float, default=40.0)
    ap.add_argument("--t_end_s", type=float, default=140.0)

    ap.add_argument("--n_bursts", type=int, default=512)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--allow_sat_sources", action="store_true")

    ap.add_argument("--dur_s_min", type=float, default=2.0)
    ap.add_argument("--dur_s_max", type=float, default=20.0)
    ap.add_argument("--rates_mbps", type=str, default="0.5,1.0,2.0,5.0")

    ap.add_argument("--sim_end_s", type=float, default=200.0)
    ap.add_argument("--out", type=Path, default=Path("udp_burst_schedule.csv"))

    # safety cap so candidates list never explodes
    ap.add_argument("--candidate_cap", type=int, default=200000)

    args = ap.parse_args()

    duration_ns = args.duration_s * 1_000_000_000
    t_start_ns = s_to_ns(args.t_start_s)
    t_end_ns = s_to_ns(args.t_end_s)
    sim_end_ns = s_to_ns(args.sim_end_s)

    rates = [float(x.strip()) for x in args.rates_mbps.split(",") if x.strip()]

    candidates, processed = load_fstate_deltas_and_collect_candidates(
        routes_dir=args.routes_dir,
        n_sats=args.n_sats,
        n_endpoints=args.n_endpoints,
        update_ns=args.update_ns,
        duration_ns=duration_ns,
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
        s_bad=args.s_bad,
        dest=args.dest,
        allow_sat_sources=args.allow_sat_sources,
        candidate_cap=args.candidate_cap,
    )

    if not candidates:
        raise SystemExit(
            f"No impacted candidates found: s_bad={args.s_bad} dest={args.dest} "
            f"in window [{args.t_start_s},{args.t_end_s}) using {args.routes_dir}"
        )

    lines = build_schedule(
        candidates=candidates,
        seed=args.seed,
        n_bursts=args.n_bursts,
        dest=args.dest,
        update_ns=args.update_ns,
        sim_end_ns=sim_end_ns,
        dur_s_min=args.dur_s_min,
        dur_s_max=args.dur_s_max,
        rates_mbps=rates,
    )

    args.out.write_text("".join(lines), encoding="utf-8")

    print(f"[+] processed_steps={processed}")
    print(f"[+] candidates={len(candidates)} (each includes s_bad on path to dest)")
    print(f"[+] wrote {args.out} with {len(lines)} bursts (sorted by start time)")


if __name__ == "__main__":
    main()

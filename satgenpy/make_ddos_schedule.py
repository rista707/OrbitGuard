#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Output line format used by ns3-sat-sim UDP burst scheduler:
# burst_id,from_node,to_node,rate,start_time_ns,end_time_ns,,
#
# Example:
# 0,120,121,1.0,0,200000000000,,
#
# NOTE: In your setup "from" is the sender (bot), "to" is the victim (target).


@dataclass
class Segment:
    t_start_s: float
    t_end_s: float
    sats: List[int]      # transit satellites on the path in that segment
    reachable: bool


@dataclass
class TraceResult:
    segments: List[Segment]
    reachable_steps: int
    total_steps: int


TIME_RE = re.compile(r"\[t=\s*([0-9]+(?:\.[0-9]+)?)s\]\s*(.*)$")
SAT_RE = re.compile(r"\bSAT_(\d+)\b")
TOTAL_RE = re.compile(r"Timesteps total:\s*(\d+)")
REACH_RE = re.compile(r"Timesteps reachable:\s*(\d+)")



def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def run_pair_trace(root_dir: str,
                   update_ms: int,
                   duration_s: int,
                   src: int,
                   dst: int,
                   topk_print: int = 10) -> TraceResult:
    """
    Calls:
      python3 -m satgen.post_analysis.main_top_used_satellites_for_pair <root_dir> <update_ms> <duration_s> <src> <dst> <topk>

    And parses the "[t= ...] ..." path-change lines into segments.
    """
    cmd = [
        "python3", "-m", "satgen.post_analysis.main_top_used_satellites_for_pair",
        root_dir, str(update_ms), str(duration_s), str(src), str(dst), str(topk_print)
    ]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(
            f"Pair trace failed for src={src} dst={dst}\nCMD: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        )

    # Extract reachability summary
    m_total = TOTAL_RE.search(out)

    # Old style might be "Timesteps reachable: X/Y (...)" or new style "Timesteps reachable: X (...)"
    m_reach_slash = re.search(r"Timesteps reachable:\s*(\d+)\s*/\s*(\d+)", out)
    m_reach = REACH_RE.search(out)

    if m_reach_slash:
        reachable_steps = int(m_reach_slash.group(1))
        total_steps = int(m_reach_slash.group(2))
    else:
        if not (m_total and m_reach):
            raise RuntimeError(
                f"Could not parse reachability summary for src={src} dst={dst}.\nOutput:\n{out}"
            )
        total_steps = int(m_total.group(1))
        reachable_steps = int(m_reach.group(1))


    # Parse path-change lines
    # Each "[t= ...]" line defines the path used starting at that time until the next change.
    changes: List[Tuple[float, List[int], bool]] = []
    for line in out.splitlines():
        mm = TIME_RE.match(line.strip())
        if not mm:
            continue
        t_s = float(mm.group(1))
        rest = mm.group(2)

        # If the tool prints unreachable lines, detect it.
        # (We handle both "UNREACHABLE" and "unreachable" robustly.)
        if "UNREACH" in rest.upper():
            changes.append((t_s, [], False))
            continue

        sats = [int(x) for x in SAT_RE.findall(rest)]
        # If it's reachable, sats may be empty for direct EP->EP (rare) but still reachable.
        changes.append((t_s, sats, True))

    # If there were no [t=] lines, treat as fully unreachable
    if not changes:
        # Build a single segment covering entire duration
        return TraceResult(
            segments=[Segment(0.0, float(duration_s), [], reachable_steps > 0)],
            reachable_steps=reachable_steps,
            total_steps=total_steps
        )

    # Build segments from changes
    changes.sort(key=lambda x: x[0])
    segments: List[Segment] = []
    for i, (t0, sats, ok) in enumerate(changes):
        t1 = float(duration_s) if i == len(changes) - 1 else changes[i + 1][0]
        # clamp
        t0c = max(0.0, min(float(duration_s), t0))
        t1c = max(0.0, min(float(duration_s), t1))
        if t1c > t0c:
            segments.append(Segment(t0c, t1c, sats, ok))

    # Ensure coverage starts at 0.0 (some traces may start later)
    if segments and segments[0].t_start_s > 0.0:
        # If first change happens after 0, assume the first known path applies from 0.
        first = segments[0]
        segments.insert(0, Segment(0.0, first.t_start_s, first.sats, first.reachable))

    return TraceResult(segments=segments, reachable_steps=reachable_steps, total_steps=total_steps)


def seconds_to_ns(s: float) -> int:
    return int(round(s * 1_000_000_000))


def merge_intervals(intervals: List[Tuple[float, float]], eps: float = 1e-9) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + eps:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def intersect_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return (lo, hi) if hi > lo else None


def intervals_where_path_contains_any(segments: List[Segment],
                                     sats_set: set,
                                     attack_window: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Return merged intervals within attack_window where the path is reachable and contains any satellite in sats_set.
    """
    raw: List[Tuple[float, float]] = []
    for seg in segments:
        if not seg.reachable:
            continue
        if sats_set and not (sats_set.intersection(seg.sats)):
            continue
        inter = intersect_interval((seg.t_start_s, seg.t_end_s), attack_window)
        if inter:
            raw.append(inter)
    return merge_intervals(raw)


def time_in_intervals(intervals: List[Tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)


def compute_transit_usage_all_bots(traces: Dict[int, TraceResult],
                                  update_ms: int) -> Dict[int, int]:
    """
    Transit usage count (in timesteps) for each satellite across all bot->target traces.
    We count one per timestep per satellite when the satellite is on the path.
    """
    dt = update_ms / 1000.0
    usage: Dict[int, int] = {}
    for bot, tr in traces.items():
        for seg in tr.segments:
            if not seg.reachable:
                continue
            steps = int(round((seg.t_end_s - seg.t_start_s) / dt))
            if steps <= 0:
                continue
            for s in seg.sats:
                usage[s] = usage.get(s, 0) + steps
    return usage


def write_udp_burst_schedule(path: str,
                             flows: List[Tuple[int, int, float, float, float]]):
    """
    flows: list of (src, dst, rate, start_s, end_s)
    writes: burst_id,src,dst,rate,start_ns,end_ns,,
    """
    with open(path, "w", encoding="utf-8") as f:
        for bid, (src, dst, rate, t0, t1) in enumerate(flows):
            f.write(f"{bid},{src},{dst},{rate},{seconds_to_ns(t0)},{seconds_to_ns(t1)},,\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", help="Path to satellite_network_state/ directory (the one containing dynamic_state_... folder)")
    ap.add_argument("--n_sats", type=int, required=True)
    ap.add_argument("--n_endpoints", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)

    ap.add_argument("--update_ms", type=int, required=True)
    ap.add_argument("--duration_s", type=int, required=True)

    ap.add_argument("--model", type=int, choices=[1, 2], required=True)
    ap.add_argument("--top_bots", type=int, default=10)

    # NEW: top-K choke satellites for model 2 (Model 2a if K=1, Model 2b if K>1)
    ap.add_argument("--top_sats", type=int, default=1,
                    help="Model 2: number of choke satellites K. K=1 => Model 2a, K>1 => Model 2b (union over top-K transit sats).")

    ap.add_argument("--attack_start", type=float, required=True)
    ap.add_argument("--attack_end", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True)

    ap.add_argument("--out", type=str, default="udp_burst_schedule.csv")
    ap.add_argument("--topk_print", type=int, default=10,
                    help="How many satellites to request tool to print; used only for debugging/trace parsing robustness.")

    args = ap.parse_args()

    n_nodes = args.n_sats + args.n_endpoints
    if not (0 <= args.target < n_nodes):
        raise SystemExit(f"Target {args.target} out of range [0, {n_nodes-1}]")

    # Endpoint IDs are [n_sats .. n_sats+n_endpoints-1]
    endpoints = list(range(args.n_sats, args.n_sats + args.n_endpoints))
    bot_candidates = [e for e in endpoints if e != args.target]

    attack_window = (max(0.0, args.attack_start), min(float(args.duration_s), args.attack_end))
    if attack_window[1] <= attack_window[0]:
        raise SystemExit("Invalid attack window (attack_end must be > attack_start and within duration)")

    # Precompute traces bot->target for all bots (only 21 max, ok)
    traces: Dict[int, TraceResult] = {}
    for bot in bot_candidates:
        traces[bot] = run_pair_trace(args.root_dir, args.update_ms, args.duration_s, bot, args.target, args.topk_print)

    if args.model == 1:
        print("\n=== MODEL 1: Direct DDoS of target endpoint ===")
        print(f"Target node id: {args.target}")

        ranked = sorted(
            bot_candidates,
            key=lambda b: traces[b].reachable_steps,
            reverse=True
        )[:args.top_bots]

        print("Top bots by reachability:")
        for b in ranked:
            tr = traces[b]
            pct = 100.0 * tr.reachable_steps / max(1, tr.total_steps)
            print(f"  bot {b}: reachable_steps={tr.reachable_steps}/{tr.total_steps} ({pct:.2f}%)")

        flows: List[Tuple[int, int, float, float, float]] = []
        for b in ranked:
            # In model 1, bots attack throughout the attack window regardless of path details.
            flows.append((b, args.target, args.rate, attack_window[0], attack_window[1]))

        write_udp_burst_schedule(args.out, flows)
        print(f"\nWrote {args.out} with {len(flows)} flows.")
        return

    # MODEL 2: Choke-point DDoS via transit satellite set (Model 2a/2b)
    print("\n=== MODEL 2: Choke-point DDoS via transit satellite set ===")
    print(f"Target node id: {args.target}")

    # Compute transit usage across all bot->target paths, pick top-K satellites
    usage = compute_transit_usage_all_bots(traces, args.update_ms)
    if not usage:
        raise SystemExit("No reachable bot->target paths found; cannot compute transit satellites for Model 2.")

    top_sorted = sorted(usage.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, args.top_sats)
    choke_list = [s for s, _ in top_sorted[:k]]
    choke_set = set(choke_list)

    if k == 1:
        print(f"Chosen S* (transit sat): {choke_list[0]}")
    else:
        print(f"Chosen S* set (top-{k} transit sats): {choke_list}")

    # Rank bots by total time where path contains ANY satellite in choke_set (within attack_window)
    bot_scores: List[Tuple[int, float, int, int]] = []
    # tuple: (bot, time_on_set_s, reachable_steps, total_steps)
    bot_intervals: Dict[int, List[Tuple[float, float]]] = {}

    for b in bot_candidates:
        tr = traces[b]
        intervals = intervals_where_path_contains_any(tr.segments, choke_set, attack_window)
        bot_intervals[b] = intervals
        bot_scores.append((b, time_in_intervals(intervals), tr.reachable_steps, tr.total_steps))

    bot_scores.sort(key=lambda x: x[1], reverse=True)
    chosen = [b for (b, t_on, rs, ts) in bot_scores if t_on > 0.0][:args.top_bots]

    if k == 1:
        print("Top bots by time-on-S*:")
    else:
        print("Top bots by time-on-S* set (union coverage):")

    for b in chosen:
        t_on = time_in_intervals(bot_intervals[b])
        tr = traces[b]
        pct = 100.0 * tr.reachable_steps / max(1, tr.total_steps)
        print(f"  bot {b}: time_on_S*={t_on:.3f}s, reachable={tr.reachable_steps}/{tr.total_steps} ({pct:.2f}%)")

    # Build flows:
    # One flow per interval to ensure traffic only runs when it actually traverses choke_set.
    flows: List[Tuple[int, int, float, float, float]] = []
    for b in chosen:
        for (t0, t1) in bot_intervals[b]:
            flows.append((b, args.target, args.rate, t0, t1))

    write_udp_burst_schedule(args.out, flows)
    print(f"\nWrote {args.out} with {len(flows)} flows (bot-interval bursts).")


if __name__ == "__main__":
    main()

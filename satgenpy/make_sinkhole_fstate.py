#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Tuple, Dict

def parse_time_from_fname(fname: str) -> int:
    m = re.match(r"fstate_(\d+)\.txt$", os.path.basename(fname))
    return int(m.group(1)) if m else -1

def find_last_nonempty_fstate(routes_dir: str, t_ns: int) -> str:
    best_t = -1
    best_path = None
    for fn in os.listdir(routes_dir):
        if not fn.startswith("fstate_") or not fn.endswith(".txt"):
            continue
        tt = parse_time_from_fname(fn)
        if tt < 0 or tt > t_ns:
            continue
        path = os.path.join(routes_dir, fn)
        if os.path.getsize(path) > 0 and tt > best_t:
            best_t = tt
            best_path = path
    if best_path is None:
        raise RuntimeError(f"No non-empty fstate_*.txt found in {routes_dir} at or before t={t_ns}")
    return best_path

def load_fstate(path: str) -> Tuple[List[List[int]], Dict[Tuple[int,int], int]]:
    rows: List[List[int]] = []
    idx: Dict[Tuple[int,int], int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            vals = [int(x) for x in parts]
            if len(vals) < 3:
                raise RuntimeError(f"Bad row (too few columns) in {path} line {line_no+1}: {line}")
            src, dst = vals[0], vals[1]
            idx[(src, dst)] = len(rows)
            rows.append(vals)
    return rows, idx

def write_fstate(path: str, rows: List[List[int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes_dir", required=True)
    ap.add_argument("--update_ns", type=int, required=True)
    ap.add_argument("--t_start_s", type=float, required=True)
    ap.add_argument("--t_end_s", type=float, required=True)
    ap.add_argument("--s_bad", type=int, required=True, help="malicious satellite node id (0..n_sats-1)")
    ap.add_argument("--dest", type=int, required=True, help="victim destination node id (endpoint or satellite)")
    ap.add_argument("--mode", choices=["mitm", "drop"], default="mitm")
    args = ap.parse_args()

    t_start_ns = int(round(args.t_start_s * 1e9))
    t_end_ns   = int(round(args.t_end_s   * 1e9))

    if t_start_ns % args.update_ns != 0:
        raise RuntimeError(f"t_start_ns={t_start_ns} not multiple of update_ns={args.update_ns}")
    if t_end_ns % args.update_ns != 0:
        raise RuntimeError(f"t_end_ns={t_end_ns} not multiple of update_ns={args.update_ns}")
    if t_end_ns <= t_start_ns:
        raise RuntimeError("t_end must be > t_start")

    # Base state: last non-empty fstate at or before t_start
    base_path = find_last_nonempty_fstate(args.routes_dir, t_start_ns)
    base_t = parse_time_from_fname(base_path)
    print(f"[+] Using base fstate: {base_path} (t={base_t} ns)")

    base_rows, base_idx = load_fstate(base_path)

    # Make a modified copy for sinkhole start
    mod_rows = [r[:] for r in base_rows]

    # For each src that has (src->s_bad), copy forwarding action into (src->dest)
    changed = 0
    skipped_no_route = 0
    skipped_missing = 0

    for (src, dst), i in base_idx.items():
        # iterate unique src values by scanning entries to dest and s_bad via dict later;
        # easiest: just loop over all src values present in file
        pass

    # Build list of src nodes present
    src_nodes = sorted({src for (src, _) in base_idx.keys()})

    for src in src_nodes:
        key_to_sbad = (src, args.s_bad)
        key_to_dest = (src, args.dest)

        if key_to_sbad not in base_idx or key_to_dest not in base_idx:
            skipped_missing += 1
            continue

        row_sbad = base_rows[base_idx[key_to_sbad]]
        row_dest = mod_rows[base_idx[key_to_dest]]

        # row format: src,dst,next,if,extra... ; next == -1 => unreachable
        if row_sbad[2] == -1:
            skipped_no_route += 1
            continue

        # Copy forwarding action columns [2:]
        row_dest[2:] = row_sbad[2:]
        changed += 1

    # Optionally drop at s_bad for dest
    if args.mode == "drop":
        k = (args.s_bad, args.dest)
        if k in base_idx:
            r = mod_rows[base_idx[k]]
            # set next hop + all remaining cols to -1
            for j in range(2, len(r)):
                r[j] = -1
            print(f"[+] DROP mode: set ({args.s_bad}->{args.dest}) to unreachable")
        else:
            print(f"[!] DROP mode: no entry found for ({args.s_bad},{args.dest}) in base table")

    out_start = os.path.join(args.routes_dir, f"fstate_{t_start_ns}.txt")
    out_end   = os.path.join(args.routes_dir, f"fstate_{t_end_ns}.txt")

    # Write sinkhole ON at start
    write_fstate(out_start, mod_rows)
    print(f"[+] Wrote sinkhole-ON fstate: {out_start}")

    # Write restore at end (restore to base state)
    write_fstate(out_end, base_rows)
    print(f"[+] Wrote restore fstate:     {out_end}")

    print("\n=== Summary ===")
    print(f"src nodes seen: {len(src_nodes)}")
    print(f"entries changed (src->dest rewritten): {changed}")
    print(f"skipped (no route to s_bad): {skipped_no_route}")
    print(f"skipped (missing entries):   {skipped_missing}")
    print(f"Attack window: [{args.t_start_s}s, {args.t_end_s}s]")
    print(f"Victim dest: {args.dest}, S_bad: {args.s_bad}, mode: {args.mode}")

if __name__ == "__main__":
    main()

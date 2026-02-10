#!/usr/bin/env python3
import argparse, os, re
from collections import defaultdict, deque

def parse_line(line):
    parts = line.strip().split(',')
    if len(parts) < 3:
        return None
    # keep everything as strings except src/dst/nh as int
    src = int(parts[0]); dst = int(parts[1]); nh = int(parts[2])
    rest = parts[3:]  # interface + extras (keep as raw strings)
    return src, dst, nh, rest

def load_patch_file(path):
    """Return dict (src,dst)->(nh,rest) from one fstate file."""
    d = {}
    if not os.path.isfile(path):
        return d
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pl = parse_line(line)
            if pl is None:
                continue
            src, dst, nh, rest = pl
            d[(src, dst)] = (nh, rest)
    return d

def list_fstate_files(routes_dir):
    files = []
    for fn in os.listdir(routes_dir):
        m = re.match(r"fstate_(\d+)\.txt$", fn)
        if m:
            t = int(m.group(1))
            files.append((t, os.path.join(routes_dir, fn)))
    files.sort(key=lambda x: x[0])
    return files

def reconstruct_state_at(routes_dir, t_ns):
    """
    Starting from fstate_0, apply all patches with timestamp <= t_ns.
    Result is dict (src,dst)->(nh,rest).
    """
    files = list_fstate_files(routes_dir)
    state = {}

    # must have fstate_0
    base = os.path.join(routes_dir, "fstate_0.txt")
    if not os.path.isfile(base):
        raise RuntimeError(f"Missing baseline: {base}")
    state.update(load_patch_file(base))

    for ts, path in files:
        if ts == 0:
            continue
        if ts <= t_ns:
            patch = load_patch_file(path)
            if patch:
                state.update(patch)
        else:
            break
    return state

def build_neighbor_iface_map(state, n_sats):
    """
    From existing routes, infer mapping src -> next_hop -> rest_fields
    (rest_fields contain out_if and possibly other needed columns).
    """
    neigh = [dict() for _ in range(n_sats)]
    for (src, dst), (nh, rest) in state.items():
        if 0 <= src < n_sats and nh >= 0:
            # store first seen mapping; should be consistent
            if nh not in neigh[src]:
                neigh[src][nh] = rest
    return neigh

def parse_isls(isls_path):
    """
    Parse isls.txt lines like:
      a,b
    or
      a b
    Return adjacency list.
    """
    adj = defaultdict(set)
    with open(isls_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # split on comma or whitespace
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            a = int(parts[0]); b = int(parts[1])
            adj[a].add(b); adj[b].add(a)
    return adj

def next_hop_toward(adj, src, target):
    """
    BFS shortest path from src to target in static ISL graph.
    Return the first step neighbor (next hop) or None.
    """
    if src == target:
        return None
    q = deque([src])
    prev = {src: None}
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                q.append(v)
    if target not in prev:
        return None
    # backtrack from target to src, find neighbor after src
    cur = target
    while prev[cur] is not None and prev[cur] != src:
        cur = prev[cur]
    # now prev[cur] == src
    return cur

def write_merged_patch(path, original_patch_dict, new_entries_dict):
    """
    Merge: new_entries overwrite original_patch entries for same (src,dst).
    Then write out as CSV lines.
    """
    merged = dict(original_patch_dict)
    merged.update(new_entries_dict)

    # write sorted for stability: by src then dst
    items = sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    with open(path, 'w', encoding='utf-8') as f:
        for (src, dst), (nh, rest) in items:
            f.write(",".join([str(src), str(dst), str(nh)] + rest) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes_dir", required=True)
    ap.add_argument("--isls", required=True)
    ap.add_argument("--n_sats", type=int, required=True)
    ap.add_argument("--dest", type=int, required=True)
    ap.add_argument("--s_bad", type=int, required=True)

    # NEW: gateway sat (the satellite that directly serves the endpoint dest)
    ap.add_argument("--gateway_sat", type=int, default=None,
                    help="Satellite that directly serves the endpoint dest (e.g., 14 for dest=120). "
                         "We do NOT rewrite this src to avoid loops.")

    ap.add_argument("--t_start_s", type=float, required=True)
    ap.add_argument("--t_end_s", type=float, required=True)
    ap.add_argument("--update_ns", type=int, required=True)
    args = ap.parse_args()

    t_start_ns = int(args.t_start_s * 1e9)
    t_end_ns   = int(args.t_end_s * 1e9)

    # reconstruct state right before attack start (t_start - 1ns)
    state_before = reconstruct_state_at(args.routes_dir, max(0, t_start_ns - 1))
    # reconstruct state at end time for restore correctness
    state_restore = reconstruct_state_at(args.routes_dir, t_end_ns)

    neigh_iface = build_neighbor_iface_map(state_before, args.n_sats)
    adj = parse_isls(args.isls)

    # Build attack patch: rewrite (src,dest) to go toward s_bad where possible
    attack_patch = {}
    changed = 0
    skipped_no_entry = 0
    skipped_no_path = 0
    skipped_no_iface = 0
    skipped_protected_src = 0

    for src in range(args.n_sats):  # focus on satellites

        # ---- CRITICAL LOOP SAFETY ----
        # Never rewrite the malicious satellite itself
        if src == args.s_bad:
            skipped_protected_src += 1
            continue

        # Never rewrite the gateway satellite (sat directly connected to endpoint dest)
        # because that creates 2-node loops (gateway <-> s_bad) very easily.
        if args.gateway_sat is not None and src == args.gateway_sat:
            skipped_protected_src += 1
            continue
        # ------------------------------

        key = (src, args.dest)
        if key not in state_before:
            skipped_no_entry += 1
            continue

        nh0, rest0 = state_before[key]
        if nh0 < 0:
            # unreachable already
            continue

        # Optional cleanup: if already going into s_bad, don't rewrite
        if nh0 == args.s_bad:
            continue

        hop = next_hop_toward(adj, src, args.s_bad)
        if hop is None:
            skipped_no_path += 1
            continue

        if hop not in neigh_iface[src]:
            # we don't know which interface corresponds to that hop => unsafe to rewrite
            skipped_no_iface += 1
            continue

        rest_for_hop = neigh_iface[src][hop]
        if nh0 != hop or rest0 != rest_for_hop:
            attack_patch[key] = (hop, rest_for_hop)
            changed += 1

    # Build restore patch: put back the original value at t_end for those we changed
    restore_patch = {}
    for (src, dst) in attack_patch.keys():
        k = (src, dst)
        if k in state_restore:
            restore_patch[k] = state_restore[k]

    # Merge into existing patch files at exact timestamps
    start_file = os.path.join(args.routes_dir, f"fstate_{t_start_ns}.txt")
    end_file   = os.path.join(args.routes_dir, f"fstate_{t_end_ns}.txt")

    orig_start = load_patch_file(start_file) if os.path.isfile(start_file) else {}
    orig_end   = load_patch_file(end_file) if os.path.isfile(end_file) else {}

    write_merged_patch(start_file, orig_start, attack_patch)
    write_merged_patch(end_file, orig_end, restore_patch)

    print("\n=== Sinkhole v2 summary ===")
    print(f"routes_dir: {args.routes_dir}")
    print(f"dest: {args.dest}, s_bad: {args.s_bad}, gateway_sat: {args.gateway_sat}")
    print(f"attack window: [{args.t_start_s}s, {args.t_end_s}s]")
    print(f"attack patch entries written: {len(attack_patch)} (changed={changed})")
    print(f"restore patch entries written: {len(restore_patch)}")
    print(f"skipped: protected_src={skipped_protected_src}, no (src,dest) entry={skipped_no_entry}, "
          f"no ISL path={skipped_no_path}, no iface map={skipped_no_iface}")
    print(f"Wrote/merged: {start_file}")
    print(f"Wrote/merged: {end_file}")

if __name__ == "__main__":
    main()

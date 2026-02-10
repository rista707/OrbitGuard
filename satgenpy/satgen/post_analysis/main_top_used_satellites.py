import sys
from satgen.post_analysis.top_used_nodes import compute_top_used_satellites

def main():
    if len(sys.argv) != 5:
        print("Usage:")
        print("  python3 -m satgen.post_analysis.main_top_used_satellites "
              "<satellite_network_dir> <dyn_ms> <end_time_s> <top_k>")
        sys.exit(1)

    sat_net_dir = sys.argv[1]
    dyn_ms = int(sys.argv[2])
    end_s = int(sys.argv[3])
    top_k = int(sys.argv[4])

    top = compute_top_used_satellites(
        satellite_network_dir=sat_net_dir,
        dynamic_state_update_interval_ms=dyn_ms,
        simulation_end_time_s=end_s,
        top_k=top_k,
        sample_every_n_steps=1,
        include_endpoints_in_count=False,
    )

    print("\nTop used satellite nodes:")
    for rank, (sid, cnt) in enumerate(top, start=1):
        print(f"{rank:2d}) sat_id={sid:3d}  count={cnt}")

if __name__ == "__main__":
    main()


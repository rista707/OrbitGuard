#!/usr/bin/env bash
set -euo pipefail

# Try all sinkhole satellites and all endpoint destinations
mapfile -t BAD_SATS < <(seq 1 119)
mapfile -t DESTS    < <(seq 120 141)

BASE_STATE="../ns3-sat-sim/simulator/test_data/end_to_end/satellite_network_state/dynamic_state_100ms_for_200s"
SATS_NET_DIR="../ns3-sat-sim/simulator/test_data/end_to_end/satellite_network_state"
ISLS="../ns3-sat-sim/simulator/test_data/end_to_end/satellite_network_state/isls.txt"

OUT_ROOT="../ns3-sat-sim/simulator/test_data/end_to_end/attacks/Sinkhole"

T_START_S=40
T_END_S=140

N_BURSTS=512
LOG_IDS="$(seq -s, 0 $((N_BURSTS-1)))"

NEED_RUNS=10
MAX_DEST_TRIES_PER_SAT=8   # limits copies per sat; bump if you want

mkdir -p "$OUT_ROOT"

# Generate plenty of prime seeds (newline-separated)
mapfile -t SEEDS < <(
python3 - <<'PY'
import random
N=2000
MASTER=20260208
LO,HI=100_000_000,2_147_483_647
rng=random.Random(MASTER)

def is_p(n):
  if n<2: return False
  for p in (2,3,5,7,11,13,17,19,23,29,31,37):
    if n==p: return True
    if n%p==0: return False
  d=n-1; s=0
  while d%2==0: s+=1; d//=2
  def mp(a,e,m):
    r=1; a%=m
    while e:
      if e&1: r=(r*a)%m
      a=(a*a)%m; e//=2
    return r
  for a in (2,3,5,7,11):
    x=mp(a,d,n)
    if x in (1,n-1): continue
    for _ in range(s-1):
      x=(x*x)%n
      if x==n-1: break
    else: return False
  return True

def next_p(x):
  x = x if x%2 else x+1
  while x<=HI and not is_p(x):
    x += 2
  return x

out=[]
tries=0
while len(out)<N:
  tries += 1
  if tries > 2_000_000:
    raise SystemExit("could not find enough primes")
  p = next_p(rng.randrange(LO,HI))
  out.append(p)

# shuffle for variety
rng.shuffle(out)
print("\n".join(map(str,out)))
PY
)

made=0
seed_i=0

# Shuffle satellites so you don't bias low IDs
mapfile -t S_SHUF < <(printf "%s\n" "${BAD_SATS[@]}" | shuf)

for S in "${S_SHUF[@]}"; do
  (( made >= NEED_RUNS )) && break

  # shuffle destinations for this satellite
  mapfile -t D_SHUF < <(printf "%s\n" "${DESTS[@]}" | shuf)

  tries=0
  for DEST in "${D_SHUF[@]}"; do
    (( made >= NEED_RUNS )) && break
    (( tries >= MAX_DEST_TRIES_PER_SAT )) && break
    tries=$((tries+1))

    SEED="${SEEDS[$seed_i]}"
    seed_i=$((seed_i+1))

    RUN_DIR="${OUT_ROOT}/run_sinkhole_sat${S}_dest${DEST}"
    STATE_DIR="${SATS_NET_DIR}/dynamic_state_100ms_for_200s_sinkhole${S}_dest${DEST}"

    echo "[+] TRY sinkhole sat=${S} dest=${DEST} seed=${SEED}"
    echo "    state: ${STATE_DIR}"
    echo "    run:   ${RUN_DIR}"

    rm -rf "$STATE_DIR"
    cp -a "$BASE_STATE" "$STATE_DIR"

    ~/hypatia/hypatia/satgenpy/make_sinkhole_fstate_v2.py \
      --routes_dir "$STATE_DIR" \
      --isls "$ISLS" \
      --n_sats 120 \
      --dest "$DEST" \
      --s_bad "$S" \
      --t_start_s "$T_START_S" \
      --t_end_s "$T_END_S" \
      --update_ns 100000000

    mkdir -p "$RUN_DIR"

    # IMPORTANT: this must return nonzero when no candidates; we skip those
    if python3 make_sinkhole_impacted_schedule.py \
        --routes_dir "$STATE_DIR" \
        --n_sats 120 --n_endpoints 22 \
        --dest "$DEST" --s_bad "$S" \
        --t_start_s "$T_START_S" --t_end_s "$T_END_S" \
        --duration_s 200 --update_ns 100000000 \
        --n_bursts "$N_BURSTS" \
        --seed "$SEED" \
        --out "$RUN_DIR/udp_burst_schedule.csv"
    then
      cat > "$RUN_DIR/config_ns3.properties" <<EOF
simulation_end_time_ns=200000000000
simulation_seed=${SEED}

satellite_network_dir="../../../satellite_network_state"
satellite_network_routes_dir="../../../satellite_network_state/dynamic_state_100ms_for_200s_sinkhole${S}_dest${DEST}"
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
EOF

      echo "[✓] SUCCESS sat=${S} dest=${DEST} -> ${RUN_DIR}"
      made=$((made+1))
      break
    else
      echo "[!] No impacted candidates for sat=${S} dest=${DEST} -> skipping"
      rm -rf "$RUN_DIR"
      rm -rf "$STATE_DIR"
    fi
  done
done

echo "[✓] Created ${made}/${NEED_RUNS} sinkhole runs"
if (( made < NEED_RUNS )); then
  echo "[!] Could not find ${NEED_RUNS} working pairs. Increase MAX_DEST_TRIES_PER_SAT or widen the attack window."
  exit 1
fi

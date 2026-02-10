from satgen.tles import read_tles
from satgen.isls import read_isls
from satgen.distance_tools import distance_m_between_satellites
from astropy import units as u

NETDIR = "../ns3-sat-sim/satellite_network_state/leo_120sats_22endpoints"  # <-- change
TIME_STEP_NS = 100_000_000   # 100 ms
END_NS = 200_000_000_000     # 200 s

tles = read_tles(f"{NETDIR}/tles.txt")
sats = tles["satellites"]
epoch = tles["epoch"]
isls = read_isls(f"{NETDIR}/isls.txt", len(sats))

max_d = 0.0
max_pair = None
max_t = None

for t in range(0, END_NS, TIME_STEP_NS):
    time = epoch + t * u.ns
    for (a, b) in isls:
        d = distance_m_between_satellites(sats[a], sats[b], str(epoch), str(time))
        if d > max_d:
            max_d = d
            max_pair = (a, b)
            max_t = t

print("MAX ISL distance:", max_d, "m")
print("Worst pair:", max_pair, "at t_ns:", max_t)
print("Set MAX_ISL_M to at least:", max_d * 1.05, "m (5% margin)")

/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef ORBIT_AWARE_THREAT_MODEL_H
#define ORBIT_AWARE_THREAT_MODEL_H

#include "ns3/object.h"
#include "ns3/nstime.h"

#include <vector>
#include <string>
#include <utility>
#include <cstdint>

namespace ns3 {

/**
 * Orbit-aware threat model shared by satellite-network components.
 *
 * This class loads a CSV describing attacks (for now: node-level
 * blackhole / grayhole) and exposes query functions used by the
 * routing and link layers.
 */
class OrbitAwareThreatModel : public Object
{
public:
  static TypeId GetTypeId (void);
  OrbitAwareThreatModel ();
  virtual ~OrbitAwareThreatModel ();

  struct AttackSpec
  {
    enum AttackType
    {
      ATTACK_NODE_BLACKHOLE,
      ATTACK_NODE_GRAYHOLE
    };

    enum TargetType
    {
      TARGET_NODE,
      TARGET_LINK,
      TARGET_REGION
    };

    std::string name;
    AttackType  type;
    TargetType  targetType;

    // Node-level targets
    std::vector<int32_t> nodeIds;

    // Link-level targets (srcNodeId, dstNodeId) – unused for now
    std::vector<std::pair<int32_t,int32_t> > links;

    // Region-level targets – unused for now
    std::string regionId;

    // Active time window in simulation seconds
    double startTimeS;
    double endTimeS;

    // Parameters
    double dropProb;  // probability of dropping a packet [0,1]
    double lossProb;  // probability of loss on a link [0,1]

    AttackSpec ()
      : type (ATTACK_NODE_BLACKHOLE),
        targetType (TARGET_NODE),
        startTimeS (0.0),
        endTimeS (0.0),
        dropProb (0.0),
        lossProb (0.0)
    {}
  };

  /**
   * Configure where to load the CSV from.
   *
   * @param runDir Directory of the current ns-3 run
   *               (BasicSimulation::GetRunDir()).
   * @param threatModelFilename File name of the CSV, relative to runDir.
   */
  void SetConfig (const std::string &runDir,
                  const std::string &threatModelFilename);

  /**
   * Load the configured CSV file.
   * Safe to call multiple times: internal state is cleared before re-loading.
   */
  void Load ();

  /**
   * Query whether a node is currently under a blackhole/grayhole attack.
   *
   * @param nodeId Node identifier used by the satellite-network module.
   * @param now Current simulation time.
   * @param dropProb [out] Set to the drop probability if true is returned.
   * @return true if at least one active blackhole/grayhole attack applies.
   */
  bool IsNodeBlackhole (int32_t nodeId,
                        Time now,
                        double &dropProb) const;

private:
  std::string m_runDir;
  std::string m_threatModelFilename;
  std::vector<AttackSpec> m_attacks;

  void LoadFromFile (const std::string &absoluteFilename);
  void ParseCsvLine (const std::string &line);
};

/**
 * Global threat model pointer used by satellite-network code.
 *
 * It is created and initialized by TopologySatelliteNetwork.
 */
extern Ptr<OrbitAwareThreatModel> g_orbitAwareThreatModel;

} // namespace ns3

#endif /* ORBIT_AWARE_THREAT_MODEL_H */

/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "orbit-aware-threat-model.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OrbitAwareThreatModel");

// Global instance pointer (defined here, declared in the header).
Ptr<OrbitAwareThreatModel> g_orbitAwareThreatModel;

NS_OBJECT_ENSURE_REGISTERED (OrbitAwareThreatModel);

TypeId
OrbitAwareThreatModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OrbitAwareThreatModel")
    .SetParent<Object> ()
    .SetGroupName ("SatelliteNetwork");
  return tid;
}

OrbitAwareThreatModel::OrbitAwareThreatModel ()
{
}

OrbitAwareThreatModel::~OrbitAwareThreatModel ()
{
}

void
OrbitAwareThreatModel::SetConfig (const std::string &runDir,
                                  const std::string &threatModelFilename)
{
  m_runDir = runDir;
  m_threatModelFilename = threatModelFilename;
}

static std::string
TrimString (const std::string &s)
{
  const char *ws = " \t\r\n";
  std::string::size_type start = s.find_first_not_of (ws);
  if (start == std::string::npos)
    {
      return "";
    }
  std::string::size_type end = s.find_last_not_of (ws);
  return s.substr (start, end - start + 1);
}

void
OrbitAwareThreatModel::Load ()
{
  if (m_threatModelFilename.empty ())
    {
      NS_LOG_INFO ("OrbitAwareThreatModel: no threat_model_filename configured, no attacks active");
      return;
    }

  std::string fullPath = m_runDir + "/" + m_threatModelFilename;
  LoadFromFile (fullPath);
}

void
OrbitAwareThreatModel::LoadFromFile (const std::string &absoluteFilename)
{
  NS_LOG_INFO ("OrbitAwareThreatModel: loading threat model from " << absoluteFilename);

  m_attacks.clear ();

  std::ifstream in (absoluteFilename.c_str (), std::ios::in);
  if (!in.is_open ())
    {
      NS_LOG_WARN ("OrbitAwareThreatModel: could not open file " << absoluteFilename
                   << " – no attacks will be active");
      return;
    }

  std::string line;
  while (std::getline (in, line))
    {
      line = TrimString (line);
      if (line.empty ())
        {
          continue;
        }
      if (line[0] == '#')
        {
          continue;
        }
      ParseCsvLine (line);
    }

  NS_LOG_INFO ("OrbitAwareThreatModel: loaded " << m_attacks.size () << " attack entries");
}

void
OrbitAwareThreatModel::ParseCsvLine (const std::string &line)
{
  // Expected format:
  // attack_name,attack_type,target_type,target_ids,start_time_s,end_time_s,params
  //
  // Example:
  // blackhole_sat10,node_blackhole,node,10,50,250,
  // grayhole_sat11,node_grayhole,node,11,100,300,drop_prob=0.4
  //
  std::vector<std::string> fields;
  std::string field;
  std::stringstream ss (line);
  while (std::getline (ss, field, ','))
    {
      fields.push_back (TrimString (field));
    }

  if (fields.size () < 6)
    {
      NS_LOG_WARN ("OrbitAwareThreatModel: ignoring malformed line (need >= 6 fields): " << line);
      return;
    }

  AttackSpec spec;

  spec.name = fields[0];

  const std::string &attackTypeStr = fields[1];
  const std::string &targetTypeStr = fields[2];
  const std::string &targetIdsStr  = fields[3];
  const std::string &startStr      = fields[4];
  const std::string &endStr        = fields[5];
  const std::string paramsStr      = (fields.size () >= 7 ? fields[6] : "");

  // Attack type
  if (attackTypeStr == "node_blackhole")
    {
      spec.type = AttackSpec::ATTACK_NODE_BLACKHOLE;
      spec.dropProb = 1.0;
    }
  else if (attackTypeStr == "node_grayhole")
    {
      spec.type = AttackSpec::ATTACK_NODE_GRAYHOLE;
      spec.dropProb = 0.5; // default, can be overridden by params
    }
  else
    {
      NS_LOG_WARN ("OrbitAwareThreatModel: unsupported attack_type '" << attackTypeStr
                   << "', line: " << line);
      return;
    }

  // Target type
  if (targetTypeStr == "node")
    {
      spec.targetType = AttackSpec::TARGET_NODE;
    }
  else if (targetTypeStr == "link")
    {
      spec.targetType = AttackSpec::TARGET_LINK;
    }
  else if (targetTypeStr == "region")
    {
      spec.targetType = AttackSpec::TARGET_REGION;
    }
  else
    {
      NS_LOG_WARN ("OrbitAwareThreatModel: unsupported target_type '" << targetTypeStr
                   << "', line: " << line);
      return;
    }

  // Timing
  spec.startTimeS = std::atof (startStr.c_str ());
  spec.endTimeS   = std::atof (endStr.c_str ());

  if (spec.endTimeS <= spec.startTimeS)
    {
      NS_LOG_WARN ("OrbitAwareThreatModel: end_time_s <= start_time_s, ignoring line: " << line);
      return;
    }

  // Node targets: "10" or "10|11|12"
  if (spec.targetType == AttackSpec::TARGET_NODE)
    {
      std::stringstream ssIds (targetIdsStr);
      std::string idToken;
      while (std::getline (ssIds, idToken, '|'))
        {
          idToken = TrimString (idToken);
          if (idToken.empty ())
            {
              continue;
            }
          int32_t id = static_cast<int32_t> (std::strtol (idToken.c_str (), nullptr, 10));
          spec.nodeIds.push_back (id);
        }

      if (spec.nodeIds.empty ())
        {
          NS_LOG_WARN ("OrbitAwareThreatModel: no nodeIds parsed from '" << targetIdsStr
                       << "', line: " << line);
          return;
        }
    }

  // Params: key=value;key2=value2
  if (!paramsStr.empty ())
    {
      std::stringstream ssParams (paramsStr);
      std::string paramToken;

      while (std::getline (ssParams, paramToken, ';'))
        {
          paramToken = TrimString (paramToken);
          if (paramToken.empty ())
            {
              continue;
            }
          std::size_t eqPos = paramToken.find ('=');
          if (eqPos == std::string::npos)
            {
              continue;
            }
          std::string key = TrimString (paramToken.substr (0, eqPos));
          std::string value = TrimString (paramToken.substr (eqPos + 1));

          if (key == "drop_prob")
            {
              spec.dropProb = std::atof (value.c_str ());
            }
          else if (key == "loss_prob")
            {
              spec.lossProb = std::atof (value.c_str ());
            }
        }
    }

  m_attacks.push_back (spec);
}

bool
OrbitAwareThreatModel::IsNodeBlackhole (int32_t nodeId,
                                        Time now,
                                        double &dropProb) const
{
  double t = now.GetSeconds ();
  bool found = false;
  double maxDrop = 0.0;

  for (const auto &a : m_attacks)
    {
      if (a.targetType != AttackSpec::TARGET_NODE)
        {
          continue;
        }
      if (a.type != AttackSpec::ATTACK_NODE_BLACKHOLE &&
          a.type != AttackSpec::ATTACK_NODE_GRAYHOLE)
        {
          continue;
        }
      if (t < a.startTimeS || t >= a.endTimeS)
        {
          continue;
        }

      if (std::find (a.nodeIds.begin (), a.nodeIds.end (), nodeId) == a.nodeIds.end ())
        {
          continue;
        }

      found = true;
      if (a.dropProb > maxDrop)
        {
          maxDrop = a.dropProb;
        }
    }

  if (found)
    {
    dropProb = maxDrop;
    return true;
    }

  dropProb = 0.0;
  return false;
}

} // namespace ns3

#pragma once
#include "ns3/header.h"
#include "ns3/uinteger.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE_DECLARE("IdentityClaimHeader");

class IdentityClaimHeader : public Header
{
public:
  IdentityClaimHeader()
    : m_magic(MAGIC), m_claimedId(0), m_burstId(0), m_seq(0) {}

  IdentityClaimHeader(uint32_t claimedId, uint32_t burstId, uint64_t seq)
    : m_magic(MAGIC), m_claimedId(claimedId), m_burstId(burstId), m_seq(seq) {}

  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("ns3::IdentityClaimHeader")
      .SetParent<Header>()
      .SetGroupName("Applications")
      .AddConstructor<IdentityClaimHeader>()
      .AddAttribute("ClaimedId", "The claimed identity",
                    UintegerValue(0),
                    MakeUintegerAccessor(&IdentityClaimHeader::m_claimedId),
                    MakeUintegerChecker<uint32_t>())
      .AddAttribute("BurstId", "The burst identifier",
                    UintegerValue(0),
                    MakeUintegerAccessor(&IdentityClaimHeader::m_burstId),
                    MakeUintegerChecker<uint32_t>())
      .AddAttribute("Seq", "The sequence number",
                    UintegerValue(0),
                    MakeUintegerAccessor(&IdentityClaimHeader::m_seq),
                    MakeUintegerChecker<uint64_t>());
    return tid;
  }

  TypeId GetInstanceTypeId() const override
  {
    return GetTypeId();
  }

  void SetClaimedId(uint32_t id) { m_claimedId = id; }
  void SetBurstId(uint32_t id) { m_burstId = id; }
  void SetSeq(uint64_t s) { m_seq = s; }

  uint32_t GetClaimedId() const { return m_claimedId; }
  uint32_t GetBurstId() const { return m_burstId; }
  uint64_t GetSeq() const { return m_seq; }
  
  // Magic number support
  static constexpr uint32_t MAGIC = 0xC0FFEE42;
  uint32_t GetMagic() const { return m_magic; }

  uint32_t GetSerializedSize() const override
  {
    return 4 + 4 + 4 + 8; // magic + claimedId + burstId + seq
  }

  void Serialize(Buffer::Iterator i) const override
  {
    i.WriteHtonU32(m_magic);
    i.WriteHtonU32(m_claimedId);
    i.WriteHtonU32(m_burstId);
    i.WriteHtonU64(m_seq);
  }

  uint32_t Deserialize(Buffer::Iterator i) override
  {
    m_magic = i.ReadNtohU32();
    m_claimedId = i.ReadNtohU32();
    m_burstId = i.ReadNtohU32();
    m_seq = i.ReadNtohU64();
    return GetSerializedSize();
  }

  void Print(std::ostream &os) const override
  {
    os << "magic=" << std::hex << m_magic << std::dec
       << " claimedId=" << m_claimedId
       << " burstId=" << m_burstId
       << " seq=" << m_seq;
  }

private:
  uint32_t m_magic;      // Magic number to identify this header
  uint32_t m_claimedId;
  uint32_t m_burstId;
  uint64_t m_seq;
};

NS_LOG_COMPONENT_DEFINE("IdentityClaimHeader");

} // namespace ns3
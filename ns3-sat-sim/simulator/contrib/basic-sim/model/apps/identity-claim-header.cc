#include "identity-claim-header.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("IdentityClaimHeader");
NS_OBJECT_ENSURE_REGISTERED(IdentityClaimHeader);

IdentityClaimHeader::IdentityClaimHeader()
  : m_magic(MAGIC), m_claimedId(0), m_burstId(0), m_seq(0) {}

IdentityClaimHeader::IdentityClaimHeader(uint32_t claimedId, uint32_t burstId, uint64_t seq)
  : m_magic(MAGIC), m_claimedId(claimedId), m_burstId(burstId), m_seq(seq) {}

TypeId
IdentityClaimHeader::GetTypeId()
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

TypeId
IdentityClaimHeader::GetInstanceTypeId() const
{
  return GetTypeId();
}

void IdentityClaimHeader::SetClaimedId(uint32_t id) { m_claimedId = id; }
void IdentityClaimHeader::SetBurstId(uint32_t id) { m_burstId = id; }
void IdentityClaimHeader::SetSeq(uint64_t s) { m_seq = s; }

uint32_t IdentityClaimHeader::GetClaimedId() const { return m_claimedId; }
uint32_t IdentityClaimHeader::GetBurstId() const { return m_burstId; }
uint64_t IdentityClaimHeader::GetSeq() const { return m_seq; }

uint32_t
IdentityClaimHeader::GetSerializedSize() const
{
  return 4 + 4 + 4 + 8; // magic + claimedId + burstId + seq
}

void
IdentityClaimHeader::Serialize(Buffer::Iterator i) const
{
  i.WriteHtonU32(m_magic);
  i.WriteHtonU32(m_claimedId);
  i.WriteHtonU32(m_burstId);
  i.WriteHtonU64(m_seq);
}

uint32_t
IdentityClaimHeader::Deserialize(Buffer::Iterator i)
{
  m_magic = i.ReadNtohU32();
  m_claimedId = i.ReadNtohU32();
  m_burstId = i.ReadNtohU32();
  m_seq = i.ReadNtohU64();
  return GetSerializedSize();
}

void
IdentityClaimHeader::Print(std::ostream &os) const
{
  os << "magic=" << std::hex << m_magic << std::dec
     << " claimedId=" << m_claimedId
     << " burstId=" << m_burstId
     << " seq=" << m_seq;
}

} // namespace ns3
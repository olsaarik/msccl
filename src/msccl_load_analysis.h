#ifndef MSCCL_GENERATED_ANALYSIS_H_
#define MSCCL_GENERATED_ANALYSIS_H_

#include "nccl.h"

#include <vector>

namespace msccl_load_analysis {

ncclResult_t run(int algorithm_index, size_t rank, size_t ranks);
std::vector<int> recvPeersForChannel(int channel);
std::vector<int> sendPeersForChannel(int channel);
int numChannels();
int numRecvOpsForChannelAndPeer(int channel, int peer);
int numSendOpsForChannelAndPeer(int channel, int peer);

} // namespace msccl_load_analysis

#endif // MSCCL_GENERATED_ANALYSIS_H_

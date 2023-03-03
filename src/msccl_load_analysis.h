#ifndef MSCCL_GENERATED_ANALYSIS_H_
#define MSCCL_GENERATED_ANALYSIS_H_

#include "nccl.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace msccl_load_analysis_detail {
  struct pair_hash
  {
      template <class T1, class T2>
      std::size_t operator() (const std::pair<T1, T2> &pair) const {
        std::size_t hash = std::hash<T1>()(pair.first);
        // This is the boost::hash_combine implementation
        hash ^= std::hash<T2>()(pair.second) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
      }
  };
} // namespace msccl_load_analysis_detail

struct MSCCLLoadAnalysisResults {
  int numThreadblocks;
  std::unordered_map<size_t, std::unordered_set<int>> recvPeers;
  std::unordered_map<size_t, std::unordered_set<int>> sendPeers;
  size_t numChannels;
  std::unordered_map<std::pair<size_t, int>, int, msccl_load_analysis_detail::pair_hash> recvOps;
  std::unordered_map<std::pair<size_t, int>, int, msccl_load_analysis_detail::pair_hash> sendOps;

  std::vector<int> recvPeersForChannel(int channel) {
    auto it = recvPeers.find(channel);
    if (it == recvPeers.end()) return {};
    return std::vector<int>(it->second.begin(), it->second.end());
  }

  std::vector<int> sendPeersForChannel(int channel) {
    auto it = sendPeers.find(channel);
    if (it == sendPeers.end()) return {};
    return std::vector<int>(it->second.begin(), it->second.end());
  }

  int numRecvOpsForChannelAndPeer(int channel, int peer) {
    auto it = recvOps.find({channel, peer});
    if (it == recvOps.end()) return 0;
    return it->second;
  }

  int numSendOpsForChannelAndPeer(int channel, int peer) {
    auto it = sendOps.find({channel, peer});
    if (it == sendOps.end()) return 0;
    return it->second;
  }
};

ncclResult_t runMSCCLLoadAnalysis(MSCCLLoadAnalysisResults* results, int algorithm_index, size_t rank, size_t ranks);

#endif // MSCCL_GENERATED_ANALYSIS_H_

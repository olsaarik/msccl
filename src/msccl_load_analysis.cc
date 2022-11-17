#include "msccl_load_analysis.h"
#include "msccl_generated.h"
#include "devcomm.h"
#include "debug.h"

#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <limits>
#include <cassert>

namespace msccl_load_analysis {

namespace detail {
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

  // Define the analysis runtime types and functions
  int rank;
  int ranks;
  int threadblock;
  std::unordered_map<size_t, std::unordered_set<int>> recvPeers;
  std::unordered_map<size_t, std::unordered_set<int>> sendPeers;
  std::unordered_map<std::pair<size_t, int>, int, pair_hash> recvOps;
  std::unordered_map<std::pair<size_t, int>, int, pair_hash> sendOps;
  size_t numChannels;

  template <class T> struct MemRef {
    size_t size;
  };

  struct Channel {
    size_t recvPeer;
    size_t sendPeer;
    size_t port;
  };

  template <class U> bool arith_cmpi_eq(U lhs, U rhs) { return lhs == rhs; }

  template <class U> U arith_addi(U lhs, U rhs) { return lhs + rhs; }

  template <class U> U arith_subi(U lhs, U rhs) { return lhs - rhs; }

  template <class U> U arith_remui(U lhs, U rhs) { return lhs % rhs; }

  template <class T> void send(Channel &chan, MemRef<T> &chunk) {
    sendOps[{chan.port, chan.sendPeer}]++;
  }

  template <class T> void recv(Channel &chan, MemRef<T> &chunk) {
    recvOps[{chan.port, chan.recvPeer}]++;
  }

  template <class T> void recv_reduce(Channel &chan, MemRef<T> &chunk) {
    recvOps[{chan.port, chan.recvPeer}]++;
  }

  size_t proc_id() {
    return rank;
  }

  size_t proc_dim() {
    return ranks;
  }

  size_t local_thread_id() {
      return threadblock;
  }

  std::tuple<size_t, size_t> chunk_vol(size_t total_size, size_t chunks, size_t chunk, size_t count) {
    size_t remainder = total_size % chunks;
    size_t small_chunk_size = total_size / chunks;
    size_t large_chunk_size = small_chunk_size + 1;
    size_t large_chunk_count = chunk < remainder ? remainder - chunk : 0;
    size_t small_chunk_count = count - large_chunk_count;
    size_t offset = (remainder - large_chunk_count) * large_chunk_size +
                    (chunk > remainder ? chunk - remainder : 0) * small_chunk_size;
    return std::make_tuple(offset, large_chunk_count * large_chunk_size + small_chunk_count * small_chunk_size);
  }

  size_t required_tiles(size_t size, size_t chunks) {
    return 1; // TODO: figure out how to calculate this
  }

  template <class T> size_t memref_dim(MemRef<T> &source, size_t index) {
    assert(index == 0);
    return source.size;
  }

  template <class T> MemRef<T> memref_subview(MemRef<T> &source, size_t offset, size_t size) {
    return MemRef<T>{size};
  }

  template <class T> void memref_copy(MemRef<T> &source, MemRef<T> &target) {}

  inline Channel create_channel(size_t peer, size_t port) {
    numChannels = std::max(numChannels, port + 1);
    recvPeers[port].insert(peer);
    sendPeers[port].insert(peer);
    return Channel{peer, peer, port};
  }

  inline Channel create_relay_channel(size_t recv_peer, size_t send_peer, size_t port) {
    numChannels = std::max(numChannels, port + 1);
    recvPeers[port].insert(recv_peer);
    sendPeers[port].insert(send_peer);
    return Channel{recv_peer, send_peer, port};
  }

  inline void debug_print(const char *str) {}

  // Include the algorithms in the detail namespace, so that they
  // will use the analysis runtime types and functions.
  using T = void;
  #define MSCCL_GEN_FUNC_ATTRIBUTES
  #include "msccl_generated.cpp.inc"
} // namespace detail

ncclResult_t run(int algorithm_index, size_t rank, size_t ranks) {
  detail::rank = rank;
  detail::ranks = ranks;
  detail::recvPeers.clear();
  detail::sendPeers.clear();
  detail::recvOps.clear();
  detail::sendOps.clear();
  detail::numChannels = 0;

  // Call each generated algorithm
  AlgorithmInfo info;
  detail::MemRef<void> input = {1};
  detail::MemRef<void> output = {1};
#define X(name, index) if (algorithm_index == index) { \
    info = name##_info(); \
    for (int tb = 0; tb < info.num_threads; ++tb) { detail::threadblock = tb; detail::name(input, output); } \
  }
MSCCL_GENERATED_ALGORITHMS_LIST
#undef X

  if (detail::numChannels >= MAXCHANNELS) {
    WARN("MSCCL generated algorithm's number of channels %zu is larger than supported", detail::numChannels);
    return ncclInternalError;
  }

  return ncclSuccess;
}

std::vector<int> recvPeersForChannel(int channel) {
  auto it = detail::recvPeers.find(channel);
  if (it == detail::recvPeers.end()) return {};
  return std::vector<int>(it->second.begin(), it->second.end());
}

std::vector<int> sendPeersForChannel(int channel) {
  auto it = detail::sendPeers.find(channel);
  if (it == detail::sendPeers.end()) return {};
  return std::vector<int>(it->second.begin(), it->second.end());
}

int numChannels() {
    return detail::numChannels;
}

int numRecvOpsForChannelAndPeer(int channel, int peer) {
  auto it = detail::recvOps.find({channel, peer});
  if (it == detail::recvOps.end()) return 0;
  return it->second;
}

int numSendOpsForChannelAndPeer(int channel, int peer) {
  auto it = detail::sendOps.find({channel, peer});
  if (it == detail::sendOps.end()) return 0;
  return it->second;
}

} // namespace msccl_load_analysis

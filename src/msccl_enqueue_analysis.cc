#include "msccl_enqueue_analysis.h"
#include "msccl_generated.h"
#include "devcomm.h"
#include "debug.h"

#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <limits>
#include <cassert>

struct EnqueueAnalysisRuntime {
  size_t rank;
  size_t ranks;
  size_t threadblock;
  size_t datatypeSize;
  size_t threadblockScratchSize;
  MSCCLEnqueueAnalysisResults* results;

  template <class T> struct MemRef {
    size_t size;
  };

  template <class T> struct Buffer {
    size_t size;

    operator MemRef<T>() {
      return MemRef<T>{size};
    }
  };

  struct Channel {};

  struct Barrier {};

  template <class U> bool arith_cmpi_eq(U lhs, U rhs) { return lhs == rhs; }

  template <class U> bool arith_cmpi_ne(U lhs, U rhs) { return lhs != rhs; }

  template <class U> U arith_addi(U lhs, U rhs) { return lhs + rhs; }

  template <class U> U arith_subi(U lhs, U rhs) { return lhs - rhs; }

  template <class U> U arith_muli(U lhs, U rhs) { return lhs * rhs; }

  template <class U> U arith_remui(U lhs, U rhs) { return lhs % rhs; }

  template <class T> void send(Channel &chan, MemRef<T> &chunk) {}

  template <class T> void recv(Channel &chan, MemRef<T> &chunk) {}

  template <class T> void recv_reduce(Channel &chan, MemRef<T> &chunk) {}

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
    return {};
  }

  inline Channel create_relay_channel(size_t recv_peer, size_t send_peer, size_t port) {
    return {};
  }

  inline void barrier_init(Barrier &barrier, size_t expected) {}

  inline void barrier_wait(Barrier &barrier) {}

  template <class T> inline void buffer_init(Buffer<T> &buffer, size_t size) {
    buffer.size = size;
    threadblockScratchSize += size * datatypeSize;
  }

  inline void debug_print(const char *str) {}

  // This is the one user function currently supported on the NCCL platform
  template<class T> void reduce_pointwise(MemRef<T> v1, MemRef<T> v2) {}
};

namespace algorithms {
  // Include the algorithms in the detail namespace, so that they
  // will use the analysis runtime types and functions.
  using RT = EnqueueAnalysisRuntime;
  using T = void;
  #define MSCCL_FUNC_ATTRIBUTES
  #define GET_MSCCL_ALGORITHMS
  #include "msccl_generated.h.inc"
} // namespace algorithms

void initRuntime(EnqueueAnalysisRuntime &runtime, size_t rank, size_t ranks, size_t datatypeSize, MSCCLEnqueueAnalysisResults* results) {
  runtime.rank = rank;
  runtime.ranks = ranks;
  runtime.results = results;
  runtime.datatypeSize= datatypeSize;
}

void initThreadblock(EnqueueAnalysisRuntime &runtime, size_t threadblock) {
  runtime.threadblock = threadblock;
  runtime.threadblockScratchSize = 0;
}

void updateResults(EnqueueAnalysisRuntime &runtime) {
  runtime.results->scratchSize = std::max(runtime.results->scratchSize, runtime.threadblockScratchSize);
}

ncclResult_t runMSCCLEnqueueAnalysis(MSCCLEnqueueAnalysisResults* results, int algorithm_index, size_t rank, size_t ranks, size_t count, size_t datatypeSize) {
  results->numThreadblocks = -1;
  results->scratchSize = 0;

  EnqueueAnalysisRuntime::MemRef<void> input = {count};
  EnqueueAnalysisRuntime::MemRef<void> output = {count};
#define X(name, index) if (algorithm_index == index) { \
    algorithms::name algo; \
    initRuntime(algo, rank, ranks, datatypeSize, results); \
    results->numThreadblocks = algo.init(input, output); \
    for (int tb = 0; tb < results->numThreadblocks; ++tb) { \
      initThreadblock(algo, tb); \
      algo.run(input, output); \
      updateResults(algo); \
    } \
  }
MSCCL_ALGORITHMS_LIST
#undef X

  if (results->numThreadblocks == -1) {
    WARN("MSCCL algorithm %d not found", algorithm_index);
    return ncclInternalError;
  }

  return ncclSuccess;
}

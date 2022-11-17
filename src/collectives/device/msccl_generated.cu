#include "common.h"
#include "collectives.h"
#include "primitives.h"
#include "msccl_generated.h"
#include <stdio.h>

namespace msccl {

#include "std_tuple_shim.h"

template<int Proto>
struct ProtoHandler {};

template<>
struct ProtoHandler<NCCL_PROTO_LL> {
  using type = ProtoLL;
  static constexpr int SlicePerChunk = 1;
  static constexpr int StepPerSlice = 1;
};

template<>
struct ProtoHandler<NCCL_PROTO_LL128> {
  using type = ProtoLL128;
  static constexpr int SlicePerChunk = 1;
  static constexpr int StepPerSlice = 1;
};

template<>
struct ProtoHandler<NCCL_PROTO_SIMPLE> {
  using type = ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>; // TODO: figure out these constants
  static constexpr int SlicePerChunk = type::SlicePerChunk;
  static constexpr int StepPerSlice = type::StepPerSlice;
};

template<typename T, typename RedOp, int Proto>
struct mscclGenDevImpl {
  struct ncclDevCommAndChannels *commAndChans;
  int nextShmemChannelIndex;
  int nthreads;
  uint64_t redOpArg;

  // Implement runtime API

  template<class U> struct MemRef {
    size_t size;
    U *data;
  };

  using ProtoType = typename ProtoHandler<Proto>::type;
  using Channel = Primitives<T, RedOp, FanAsymmetric<1,1>, 0, ProtoType, 0>;

  template <class U> __device__ bool arith_cmpi_eq(U lhs, U rhs) { return lhs == rhs; }

  template <class U> __device__ U arith_addi(U lhs, U rhs) { return lhs + rhs; }

  template <class U> __device__ U arith_subi(U lhs, U rhs) { return lhs - rhs; }

  template <class U> __device__ U arith_remui(U lhs, U rhs) { return lhs % rhs; }

  __device__ void send(Channel &chan, MemRef<T> chunk) {
    chan.setDataPtrs(chunk.data, nullptr);
    chan.send(0, chunk.size);
  }

  __device__ void recv(Channel &chan, MemRef<T> chunk) {
    chan.setDataPtrs(nullptr, chunk.data);
    chan.recv(0, chunk.size);
  }

  __device__ void recv_reduce(Channel &chan, MemRef<T> chunk) {
    chan.setDataPtrs(chunk.data, chunk.data);
    chan.recvReduceCopy(0, 0, chunk.size);
  }

  __device__ size_t proc_id() {
    return ncclShmem.comm.rank;
  }

  __device__ size_t proc_dim() {
    return ncclShmem.comm.nRanks;
  }

  __device__ size_t local_thread_id() {
    return blockIdx.x;
  }

  __device__ std::tuple<size_t, size_t> chunk_vol(size_t total_size, size_t chunks, size_t chunk, size_t count) {
    size_t remainder = total_size % chunks;
    size_t small_chunk_size = total_size / chunks;
    size_t large_chunk_size = small_chunk_size + 1;
    size_t large_chunk_count = chunk < remainder ? remainder - chunk : 0;
    size_t small_chunk_count = count - large_chunk_count;
    size_t offset = (remainder - large_chunk_count) * large_chunk_size +
                    (chunk > remainder ? chunk - remainder : 0) * small_chunk_size;
    return std::make_tuple(offset, large_chunk_count * large_chunk_size + small_chunk_count * small_chunk_size);
  }

  __device__ size_t required_tiles(size_t size, size_t chunks) {
    // Calculate the maximum size of a chunk that a call to the primitives can handle
    size_t maxChunkSize = ProtoType::calcBytePerStep()/sizeof(T) * ProtoHandler<Proto>::StepPerSlice * ProtoHandler<Proto>::SlicePerChunk;
    // Now given that we know the size of the whole buffer and how many chunks the algorithm is guaranteed to divide it to,
    // we can calculate the number of tiles the buffer needs to be divided into to ensure maxChunkSize is never exceeded.
    // Formally, the following inequality must hold: ceil(ceil(size / requiredTiles) / chunks) <= maxChunkSize
    return ((size + chunks - 1) / chunks + maxChunkSize - 1) / maxChunkSize;
  }

  __device__ size_t memref_dim(MemRef<T> &source, size_t index) {
    // assert(index == 0);
    return source.size;
  }

  __device__ MemRef<T> memref_subview(MemRef<T> &source, size_t offset, size_t size) {
    return MemRef<T>{size, source.data + offset};
  }

  __device__ void memref_copy(MemRef<T> &source, MemRef<T> &target) {
    // TODO: make this more efficient
    for (int i = threadIdx.x; i < source.size; i += nthreads)
      target.data[i] = source.data[i];
    __syncthreads();
  }

  __device__ inline Channel create_channel(size_t peer, size_t port) {
    return create_relay_channel(peer, peer, port);
  }

  __device__ inline Channel create_relay_channel(size_t recv_peer, size_t send_peer, size_t port) {
    struct ncclChannel *channel = &commAndChans->channels[port];
    simpleCopy(&ncclShmem.channels[nextShmemChannelIndex], channel, threadIdx.x, nthreads);
    __syncthreads();
    nextShmemChannelIndex += 1;
    int recvPeerInt = recv_peer;
    int sendPeerInt = send_peer;
    return Channel(threadIdx.x, nthreads, &recvPeerInt, &sendPeerInt, nullptr, nullptr, redOpArg);
  }

  // End runtime API implementation

  // Include the generated code inside this struct so that they use the types and functions defined above
  #define MSCCL_GEN_FUNC_ATTRIBUTES __device__
  #include "../../msccl_generated.cpp.inc"

  template<void (mscclGenDevImpl::*algorithm)(MemRef<T>,MemRef<T>), int algorithmIndex>
  __device__ void run(struct ncclDevComm* comm, struct ncclWorkElem work) {
    nthreads = work.header.nWarps*WARP_SIZE;
    redOpArg = work.redOpArg;

    simpleCopy(&ncclShmem.comm, comm, threadIdx.x, nthreads);
    __syncthreads();

    commAndChans = (ncclDevCommAndChannels*)comm;
    nextShmemChannelIndex = 0;

    MemRef<T> input;
    input.data = (T*)work.sendbuff;
    input.size = work.count;

    MemRef<T> output;
    output.data = (T*)work.recvbuff;
    output.size = work.count; // TODO: fix these counts. Relative sizes in algorithm signature oslt?

    (this->*algorithm)(input, output);
  }

  #define X(name, index) \
    __device__ void _run_wrapper_##name(struct ncclDevComm* comm, struct ncclWorkElem work) { \
      run<&mscclGenDevImpl::name, index>(comm, work); \
    }
  MSCCL_GENERATED_ALGORITHMS_LIST
  #undef X
};

} // namespace msccl

// For generated algorithms, always generate a separate kernel
#define IMPL_COLL_KERN_GEN(name, proto, devredop, type) \
__global__ void NCCL_KERN_NAME_GEN(name, proto, devredop, type)(struct ncclDevComm* comm, struct ncclWorkElem work) { \
  msccl::mscclGenDevImpl<type, Func##devredop<type>, NCCL_PROTO_##proto>()._run_wrapper_##name(comm, work); \
}

#define X(name, index) IMPL_COLL_GEN(name);
MSCCL_GENERATED_ALGORITHMS_LIST
#undef X

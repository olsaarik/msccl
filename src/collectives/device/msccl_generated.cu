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

  struct CUDARuntime {
    struct ncclDevCommAndChannels *commAndChans;
    int nextShmemChannelIndex;
    int nthreads;
    uint64_t redOpArg;
    size_t scratchAllocatedBytes;

    // Implement runtime API

    template<class U> struct MemRef {
      size_t size;
      U *data;
    };

    template <class U> struct Buffer {
      size_t size;
      U *data;

      __device__ operator MemRef<U>() {
        return MemRef<U>{size, data};
      }
    };

    using ProtoType = typename ProtoHandler<Proto>::type;
    using PrimitivesType = Primitives<T, RedOp, FanAsymmetric<1,1>, 0, ProtoType, 0>;
    struct ChannelBuilder {
      CUDARuntime* runtime;
      size_t recv_peer;
      size_t send_peer;
      size_t port;
    };
    struct Channel {
      union {
        PrimitivesType primitives;
      };
      bool initialized;
      __device__ Channel() : initialized(false) {}
      __device__ void operator=(ChannelBuilder builder) {
        CUDARuntime* runtime = builder.runtime;
        struct ncclChannel *channel = &runtime->commAndChans->channels[builder.port];
        simpleCopy(&ncclShmem.channels[runtime->nextShmemChannelIndex], channel, threadIdx.x, runtime->nthreads);
        __syncthreads();
        runtime->nextShmemChannelIndex += 1;
        int recvPeerInt = builder.recv_peer;
        int sendPeerInt = builder.send_peer;
        new (&primitives) PrimitivesType(threadIdx.x, runtime->nthreads, &recvPeerInt, &sendPeerInt, nullptr, nullptr, runtime->redOpArg);
        initialized = true;
      }
      __device__ ~Channel() {
        if (initialized) primitives.~PrimitivesType();
      }
    };
    // TODO: Implement barrier
    struct Barrier {};

    template <class U> __device__ bool arith_cmpi_eq(U lhs, U rhs) { return lhs == rhs; }

    template <class U> __device__ bool arith_cmpi_ne(U lhs, U rhs) { return lhs != rhs; }

    template <class U> __device__ U arith_addi(U lhs, U rhs) { return lhs + rhs; }

    template <class U> __device__ U arith_subi(U lhs, U rhs) { return lhs - rhs; }

    template <class U> __device__ U arith_muli(U lhs, U rhs) { return lhs * rhs; }

    template <class U> __device__ U arith_remui(U lhs, U rhs) { return lhs % rhs; }

    __device__ void send(Channel &chan, MemRef<T> chunk) {
      chan.primitives.setDataPtrs(chunk.data, nullptr);
      chan.primitives.send(0, chunk.size);
    }

    __device__ void recv(Channel &chan, MemRef<T> chunk) {
      chan.primitives.setDataPtrs(nullptr, chunk.data);
      chan.primitives.recv(0, chunk.size);
    }

    __device__ void recv_reduce(Channel &chan, MemRef<T> chunk) {
      chan.primitives.setDataPtrs(chunk.data, chunk.data);
      chan.primitives.recvReduceCopy(0, 0, chunk.size);
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

    __device__ inline ChannelBuilder create_channel(size_t peer, size_t port) {
      return create_relay_channel(peer, peer, port);
    }

    __device__ inline ChannelBuilder create_relay_channel(size_t recv_peer, size_t send_peer, size_t port) {
      return ChannelBuilder{this, recv_peer, send_peer, port};
    }

    // TODO: Implement barrier
    __device__ inline void barrier_init(Barrier &barrier, size_t expected) {}
    __device__ inline void barrier_wait(Barrier &barrier) {}
    
    template <class U> __device__ inline void buffer_init(Buffer<U> &buffer, size_t size) {
      // TODO: do we need to worry about alignment?
      buffer.size = size;
      buffer.data = (U*)((char*) ncclShmem.mscclShmem.scratchBuffer + scratchAllocatedBytes);
      scratchAllocatedBytes += size * sizeof(U);
    }

    // This is the one user function currently supported on the NCCL platform
    template <class U> __device__ void reduce_pointwise(MemRef<U> v1, MemRef<U> v2) {}

    __device__ void debug_print(const char *str) {
        printf("rank %d, threadblock %d: %s\n", ncclShmem.comm.rank, blockIdx.x, str);
    }

  }; // struct CUDARuntime

  struct algorithms {
    using RT = CUDARuntime;
    #define MSCCL_FUNC_ATTRIBUTES __device__
    #define GET_MSCCL_ALGORITHMS
    #include "msccl_generated.h.inc"
  }; // struct algorithms

  template<class Algorithm, int algorithmIndex>
  __device__ static void run(struct ncclDevComm* comm, struct ncclWorkElem work) {
    Algorithm algo;
    algo.nthreads = work.header.nWarps*WARP_SIZE;
    algo.redOpArg = work.redOpArg;

    simpleCopy(&ncclShmem.comm, comm, threadIdx.x, algo.nthreads);
    __syncthreads();

    algo.commAndChans = (ncclDevCommAndChannels*)comm;
    algo.nextShmemChannelIndex = 0;

    typename Algorithm::MemRef<T> input;
    input.data = (T*)work.sendbuff;
    input.size = work.count;

    typename Algorithm::MemRef<T> output;
    output.data = (T*)work.recvbuff;
    output.size = work.count; // TODO: fix these counts. Relative sizes in algorithm signature oslt?

    algo.init(input, output);
    algo.run(input, output);
  }

  #define X(name, index) \
    __device__ static void _run_wrapper_##name(struct ncclDevComm* comm, struct ncclWorkElem work) { \
      run<algorithms::name, index>(comm, work); \
    }
  MSCCL_ALGORITHMS_LIST
  #undef X
};

} // namespace msccl

// For generated algorithms, always generate a separate kernel
#define IMPL_COLL_KERN_GEN(name, proto, devredop, type) \
__global__ void NCCL_KERN_NAME_GEN(name, proto, devredop, type)(struct ncclDevComm* comm, struct ncclWorkElem work) { \
  msccl::mscclGenDevImpl<type, Func##devredop<type>, NCCL_PROTO_##proto>::_run_wrapper_##name(comm, work); \
}

#define X(name, index) IMPL_COLL_GEN(name);
MSCCL_ALGORITHMS_LIST
#undef X

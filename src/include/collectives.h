/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "msccl_generated.h"

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMax, ncclDevMin,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

#define FUNC_INDEX_P2P 0
#define FUNC_INDEX(func, devredop, ncclType, al, pr) (1+ncclNumTypes+(((((func)*ncclNumDevRedOps + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))
#define FUNC_INDEX_GEN(index, devredop, ncclType, pr) (FUNC_INDEX(NCCL_NUM_FUNCTIONS-1, ncclNumDevRedOps-1, ncclNumTypes-1, NCCL_NUM_ALGORITHMS-1, NCCL_NUM_PROTOCOLS-1)+1+ \
  ((((index)*MSCCL_NUM_GENERATED_ALGOS + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_PROTOCOLS+(pr))

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ncclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_ONERANK_REDUCE_NAME(devredop, type) \
  ncclFunction_OneRankReduce_##devredop##_##type

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  ncclKernel_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_KERN_NAME_GEN(name, proto, devredop, type) \
  ncclKernel_##name##_##proto##_##devredop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  nccl##func##algo##proto

/* Declare all collective operations */
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclDevComm* comm, struct ncclWorkElem c); \

#define DECL5_GEN(func, proto, devredop, type) \
  extern __global__ void NCCL_KERN_NAME_GEN(func, proto, devredop, type)(struct ncclDevComm* comm, struct ncclWorkElem c); \

#define CONCAT(a,b) a##b
#define MACRO_IF(cond, t, f) CONCAT(MACRO_IF_, cond)(t, f)
#define MACRO_IF_0(t, f) f
#define MACRO_IF_1(t, f) t

#define DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL128,  devredop, type))

#define DECL4_GEN(name, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5_GEN(name, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5_GEN(name, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5_GEN(name, LL128,  devredop, type))

#define DECL3(func, devredop, type, undef) \
  DECL4(func, RING,    devredop, type, undef) \
  DECL4(func, TREE,    devredop, type, undef) \
  DECL4(func, MSCCL,    devredop, type, undef) \
  DECL4(func, COLLNET, devredop, type, undef)

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __nv_bfloat16, /*undef=*/undefForFloat)
#define DECL2_GEN(name, devredop, undefForFloat) \
  DECL4_GEN(name, devredop, int8_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, uint8_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, int32_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, uint32_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, int64_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, uint64_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, half, /*undef=*/undefForFloat) \
  DECL4_GEN(name, devredop, float, /*undef=*/undefForFloat) \
  DECL4_GEN(name, devredop, double, /*undef=*/undefForFloat) \
  DECL4_GEN(name, devredop, __nv_bfloat16, /*undef=*/undefForFloat)
#else
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat)
#define DECL2_GEN(name, devredop, undefForFloat) \
  DECL4_GEN(name, devredop, int8_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, uint8_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, int32_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, uint32_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, int64_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, uint64_t, /*undef=*/0) \
  DECL4_GEN(name, devredop, half, /*undef=*/undefForFloat) \
  DECL4_GEN(name, devredop, float, /*undef=*/undefForFloat) \
  DECL4_GEN(name, devredop, double, /*undef=*/undefForFloat)
#endif

#define DECL(func) \
  DECL2(func, Sum, /*undefForFloat=*/0) \
  DECL2(func, Prod, /*undefForFloat=*/0) \
  DECL2(func, Min, /*undefForFloat=*/0) \
  DECL2(func, Max, /*undefForFloat=*/0) \
  DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  DECL2(func, SumPostDiv, /*undefForFloat=*/1)
#define DECL_GEN(name) \
  DECL2_GEN(name, Sum, /*undefForFloat=*/0) \
  DECL2_GEN(name, Prod, /*undefForFloat=*/0) \
  DECL2_GEN(name, Min, /*undefForFloat=*/0) \
  DECL2_GEN(name, Max, /*undefForFloat=*/0) \
  DECL2_GEN(name, PreMulSum, /*undefForFloat=*/0) \
  DECL2_GEN(name, SumPostDiv, /*undefForFloat=*/1)

DECL2(Broadcast, Sum, /*undefForFloat=*/0)
DECL(Reduce)
DECL2(AllGather, Sum, /*undefForFloat=*/0)
DECL(ReduceScatter)
DECL(AllReduce)
DECL2(AllToAll, Sum, /*undefForFloat=*/0)
DECL(CustomCollective)
#define X(name, index) DECL_GEN(name)
MSCCL_GENERATED_ALGORITHMS_LIST
#undef X
DECL5(SendRecv, RING, SIMPLE, Sum, int8_t)

extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, half)();
#if defined(__CUDA_BF16_TYPES_EXIST__)
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, __nv_bfloat16)();
#endif
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, float)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, double)();

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SENDRECV_SLICEFACTOR 4
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above

#endif

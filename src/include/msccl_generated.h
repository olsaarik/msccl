#ifndef MSCCL_GENERATED_H_
#define MSCCL_GENERATED_H_

struct AlgorithmInfo {
    int num_threads;
};

#include "msccl_generated.h.inc"

const int MSCCL_NUM_GENERATED_ALGOS = 0
#define X(name, index) + 1
MSCCL_GENERATED_ALGORITHMS_LIST
#undef X
    ;

inline AlgorithmInfo mscclGeneratedAlgorithmInfoByIndex(int index) {
  switch (index) {
#define X(name, algo_index) \
    case algo_index: \
        return name##_info();
MSCCL_GENERATED_ALGORITHMS_LIST
#undef X
    default:
      return AlgorithmInfo();
  }
}

#endif // MSCCL_GENERATED_H_

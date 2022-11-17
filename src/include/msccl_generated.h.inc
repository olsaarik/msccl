#define MSCCL_GENERATED_ALGORITHMS_LIST \
  X(ring_allreduce, 0)

inline AlgorithmInfo ring_allreduce_info() {
  AlgorithmInfo info;
  info.num_threads = 1;
  return info;
}

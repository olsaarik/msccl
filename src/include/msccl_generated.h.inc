#ifdef GET_MSCCL_ALGORITHMS
#undef GET_MSCCL_ALGORITHMS
struct allreduce_allpairs : public RT {
  typename RT::Barrier barrier;

  typename RT::template Buffer<T> scratch;

  MSCCL_FUNC_ATTRIBUTES size_t init(typename RT::template MemRef<T> v1, typename RT::template MemRef<T> unused) {
    RT::debug_print("allreduce_allpairs: init");
    size_t v2 = RT::proc_id();
    size_t v3 = RT::proc_dim();
    size_t v4 = 0;
    size_t v5 = RT::memref_dim(v1, v4);
    size_t v6 = 1;
    size_t v7;
    size_t v8;
    std::tie(v7, v8) = RT::chunk_vol(v5, v3, v2, v6);
    typename RT::template MemRef<T> v9 = RT::memref_subview(v1, v7, v8);
    size_t v10 = RT::arith_muli(v8, v3);
    typename RT::template Buffer<T>& v11 = this->scratch;
    RT::buffer_init(v11, v10);
    typename RT::Barrier v12 = this->barrier;
    RT::barrier_init(v12, v3);
    return v3;
  }

  MSCCL_FUNC_ATTRIBUTES void run(typename RT::template MemRef<T> v1, typename RT::template MemRef<T> unused) {
    RT::debug_print("allreduce_allpairs: run");
    size_t v2 = RT::proc_id();
    size_t v3 = RT::proc_dim();
    size_t v4 = RT::local_thread_id();
    typename RT::template MemRef<T> v5 = this->scratch;
    bool v6 = RT::arith_cmpi_ne(v4, v2);
    typename RT::Channel v8;
    RT::debug_print("allreduce_allpairs: first communication");
    if (v6) {
      size_t v7 = 0;
      v8 = RT::create_channel(v4, v7);
      size_t v9 = RT::memref_dim(v1, v7);
      size_t v10 = 1;
      size_t v11;
      size_t v12;
      std::tie(v11, v12) = RT::chunk_vol(v9, v3, v4, v10);
      typename RT::template MemRef<T> v13 = RT::memref_subview(v1, v11, v12);
      RT::send(v8, v13);
      size_t v14 = RT::memref_dim(v5, v7);
      size_t v15;
      size_t v16;
      std::tie(v15, v16) = RT::chunk_vol(v14, v3, v4, v10);
      typename RT::template MemRef<T> v17 = RT::memref_subview(v5, v15, v16);
      RT::recv(v8, v17);
      ;
    }
    typename RT::Barrier v18 = this->barrier;
    RT::barrier_wait(v18);
    RT::debug_print("allreduce_allpairs: local reduction");
    size_t v19 = 0;
    size_t v20 = RT::memref_dim(v1, v19);
    size_t v21 = 1;
    size_t v22;
    size_t v23;
    std::tie(v22, v23) = RT::chunk_vol(v20, v3, v2, v21);
    typename RT::template MemRef<T> v24 = RT::memref_subview(v1, v22, v23);
    for (size_t v25 = v19; v25 < v3; v25 += v21) {
      bool v26 = RT::arith_cmpi_ne(v25, v2);
      if (v26) {
        size_t v27 = RT::memref_dim(v5, v19);
        size_t v28;
        size_t v29;
        std::tie(v28, v29) = RT::chunk_vol(v27, v3, v25, v21);
        typename RT::template MemRef<T> v30 = RT::memref_subview(v5, v28, v29);
        size_t v31;
        size_t v32;
        std::tie(v31, v32) = RT::chunk_vol(v23, v3, v4, v21);
        typename RT::template MemRef<T> v33 = RT::memref_subview(v24, v31, v32);
        size_t v34;
        size_t v35;
        std::tie(v34, v35) = RT::chunk_vol(v29, v3, v4, v21);
        typename RT::template MemRef<T> v36 = RT::memref_subview(v30, v34, v35);
        reduce_pointwise(v33, v36);
        ;
      };
    }
    RT::barrier_wait(v18);
    RT::debug_print("allreduce_allpairs: second communication");
    if (v6) {
      // typename RT::Channel v37 = RT::create_channel(v4, v19);
      size_t v38;
      size_t v39;
      std::tie(v38, v39) = RT::chunk_vol(v20, v3, v2, v21);
      typename RT::template MemRef<T> v40 = RT::memref_subview(v1, v38, v39);
      RT::send(v8, v40);
      size_t v41;
      size_t v42;
      std::tie(v41, v42) = RT::chunk_vol(v20, v3, v4, v21);
      typename RT::template MemRef<T> v43 = RT::memref_subview(v1, v41, v42);
      RT::recv(v8, v43);
      ;
    }
    RT::debug_print("allreduce_allpairs: done");
    return;
  }

};
#endif // GET_MSCCL_ALGORITHMS
#define MSCCL_ALGORITHMS_LIST \
  X(allreduce_allpairs, 0)

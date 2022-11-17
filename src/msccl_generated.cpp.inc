MSCCL_GEN_FUNC_ATTRIBUTES void ring_allreduce(MemRef<T> v1, MemRef<T> v2) {
  size_t v3 = 0;
  size_t v4 = 1;
  size_t v5 = proc_id();
  size_t v6 = proc_dim();
  size_t v7 = arith_addi(v5, v4);
  size_t v8 = arith_remui(v7, v6);
  size_t v9 = arith_subi(v5, v4);
  size_t v10 = arith_remui(v9, v6);
  size_t v11 = arith_subi(v6, v4);
  Channel v12 = create_relay_channel(v10, v8, v3);
  size_t v13 = memref_dim(v1, v3);
  size_t v14 = required_tiles(v13, v6);
  memref_copy(v1, v2);
  for (size_t v15 = v3; v15 < v14; v15 += v4) {
    size_t v16;
    size_t v17;
    std::tie(v16, v17) = chunk_vol(v13, v14, v15, v4);
    MemRef<T> v18 = memref_subview(v2, v16, v17);
    for (size_t v19 = v3; v19 < v11; v19 += v4) {
      size_t v20 = arith_subi(v5, v19);
      size_t v21 = arith_remui(v20, v6);
      size_t v22;
      size_t v23;
      std::tie(v22, v23) = chunk_vol(v17, v6, v21, v4);
      MemRef<T> v24 = memref_subview(v18, v22, v23);
      send(v12, v24);
      size_t v25 = arith_subi(v5, v19);
      size_t v26 = arith_subi(v25, v4);
      size_t v27 = arith_remui(v26, v6);
      size_t v28;
      size_t v29;
      std::tie(v28, v29) = chunk_vol(v17, v6, v27, v4);
      MemRef<T> v30 = memref_subview(v18, v28, v29);
      recv_reduce(v12, v30);
    };
    for (size_t v31 = v3; v31 < v11; v31 += v4) {
      size_t v32 = arith_subi(v5, v31);
      size_t v33 = arith_addi(v32, v4);
      size_t v34 = arith_remui(v33, v6);
      size_t v35;
      size_t v36;
      std::tie(v35, v36) = chunk_vol(v17, v6, v34, v4);
      MemRef<T> v37 = memref_subview(v18, v35, v36);
      send(v12, v37);
      size_t v38 = arith_subi(v5, v31);
      size_t v39 = arith_remui(v38, v6);
      size_t v40;
      size_t v41;
      std::tie(v40, v41) = chunk_vol(v17, v6, v39, v4);
      MemRef<T> v42 = memref_subview(v18, v40, v41);
      recv(v12, v42);
    };
  }
  return;
}
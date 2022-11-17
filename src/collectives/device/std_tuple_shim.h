#ifndef STD_TUPLE_SHIM_H_
#define STD_TUPLE_SHIM_H_

// This file implements a subset of the std::tuple interface for CUDA

namespace std {

template<typename T1, typename T2>
struct tuple {
  __device__ tuple(T1 a1, T2 a2) : v1(a1), v2(a2) {}
  
  template<typename U1, typename U2>
  __device__ void operator=(const tuple<U1, U2>& other) {
    v1 = other.v1;
    v2 = other.v2;
  }

  T1 v1;
  T2 v2;
};

// std::tie
template<typename T1, typename T2>
__device__ tuple<T1&, T2&> tie(T1& a1, T2& a2) {
  return tuple<T1&, T2&>(a1, a2);
}

// std::make_tuple
template<typename T1, typename T2>
__device__ tuple<T1, T2> make_tuple(T1 a1, T2 a2) {
  return tuple<T1, T2>(a1, a2);
}

}

#endif // STD_TUPLE_SHIM_H_
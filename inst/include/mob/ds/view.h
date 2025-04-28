#pragma once
#include <Rcpp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

namespace mob {
namespace ds {

template <typename Pointer>
struct span {
  using value_type = typename cuda::std::pointer_traits<Pointer>::element_type;
  using difference_type =
      typename cuda::std::pointer_traits<Pointer>::difference_type;
  using iterator = Pointer;

  __host__ __device__ span() : first(nullptr), last(nullptr) {}
  __host__ __device__ span(Pointer first, Pointer last)
      : first(first), last(last) {}
  __host__ __device__ span(Pointer first, size_t size)
      : first(first), last(first + size) {}

  __host__ __device__ size_t size() const {
    return last - first;
  }

  __host__ __device__ Pointer data() const {
    return first;
  }

  __host__ __device__ iterator begin() const {
    return first;
  }

  __host__ __device__ iterator end() const {
    return last;
  }

  __host__ __device__ value_type operator[](size_t index) const {
    return first[index];
  }

private:
  Pointer first;
  Pointer last;
};

template <typename T>
using host_span = span<T *>;

template <typename T>
using device_span = span<thrust::device_ptr<T>>;

template <typename T>
host_span<const T> view(const std::vector<T> &data) {
  return {data.data(), data.size()};
}

inline host_span<const int> view(const Rcpp::IntegerVector &data) {
  return {data.begin(), static_cast<size_t>(data.size())};
}

template <typename T>
host_span<const T> view(const thrust::host_vector<T> &data) {
  return {data.data(), data.size()};
}

template <typename T>
device_span<const T> view(const thrust::device_vector<T> &data) {
  return {data.data(), data.size()};
}

} // namespace ds
} // namespace mob

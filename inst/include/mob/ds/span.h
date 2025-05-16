#pragma once
#include <mob/system.h>

#include <cuda/std/ranges>
#include <dust/random/cuda_compatibility.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

namespace mob {
namespace ds {

template <typename System, typename T>
struct span {
  using pointer = typename System::pointer<T>;
  using iterator = pointer;
  using reference = decltype(*std::declval<pointer>());
  using value_type = typename cuda::std::pointer_traits<pointer>::element_type;
  using difference_type =
      typename cuda::std::pointer_traits<pointer>::difference_type;

  __host__ __device__ span() : first(nullptr), last(nullptr) {}
  __host__ __device__ span(pointer first, pointer last)
      : first(first), last(last) {}
  __host__ __device__ span(pointer first, size_t size)
      : first(first), last(first + size) {}

  template <typename Container>
  __host__ __device__ span(Container &&c)
      : first(c.data()), last(c.data() + c.size()) {}

  __host__ __device__ size_t size() const {
    return last - first;
  }

  __host__ __device__ pointer data() const {
    return first;
  }

  __host__ __device__ iterator begin() const {
    return first;
  }

  __host__ __device__ iterator end() const {
    return last;
  }

  __host__ __device__ bool empty() const {
    return first == last;
  }

  __host__ __device__ reference front() const {
    return *first;
  }

  __host__ __device__ reference operator[](size_t index) const {
    return first[index];
  }

  span subspan(size_t offset, size_t count) const {
    return span(begin() + offset, begin() + offset + count);
  }

  static_assert(std::random_access_iterator<iterator>);
  static_assert(std::sized_sentinel_for<iterator, iterator>);

private:
  pointer first;
  pointer last;
};

template <typename T>
span(thrust::device_vector<T>) -> span<system::device, T>;

template <typename T>
span(thrust::host_vector<T>) -> span<system::host, T>;

template <typename T>
span(std::vector<T>) -> span<system::host, T>;

} // namespace ds
} // namespace mob

namespace cuda::std::ranges {

template <typename System, typename T>
constexpr bool enable_view<mob::ds::span<System, T>> = true;

template <typename System, typename T>
constexpr bool enable_borrowed_range<mob::ds::span<System, T>> = true;

} // namespace cuda::std::ranges

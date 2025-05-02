#pragma once
#include <mob/compat.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

namespace mob {
namespace ds {

template <typename System, typename T>
struct span {
  using pointer = typename System::pointer<T>;
  using iterator = pointer;
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

  __host__ __device__ value_type operator[](size_t index) const {
    return first[index];
  }

private:
  pointer first;
  pointer last;
};

} // namespace ds
} // namespace mob

namespace std::ranges {

template <typename System, typename T>
constexpr bool enable_view<mob::ds::span<System, T>> = true;

}

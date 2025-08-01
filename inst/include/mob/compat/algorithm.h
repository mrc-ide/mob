#pragma once

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <dust/random/cuda_compatibility.hpp>

namespace mob {
namespace compat {

// Supposedly cccl has a lower_bound implementation, but it doesn't seem to be
// exposed. `#include <cuda/std/algorithm>` doesn't work.
//
// https://en.cppreference.com/w/cpp/algorithm/lower_bound
// https://en.cppreference.com/w/cpp/algorithm/binary_search
__nv_exec_check_disable__ template <
    class ForwardIt, class T = cuda::std::iter_value_t<ForwardIt>,
    class Compare>
__host__ __device__ ForwardIt lower_bound(ForwardIt first, ForwardIt last,
                                          const T &value, Compare comp) {
  ForwardIt it;
  cuda::std::iter_difference_t<ForwardIt> count, step;
  count = cuda::std::distance(first, last);

  while (count > 0) {
    it = first;
    step = count / 2;

    // cuda::std::advance does work with thrust iterators
    it += step; // cuda::std::advance(it, step);

    if (comp(*it, value)) {
      first = ++it;

      count -= step + 1;
    } else
      count = step;
  }

  return first;
}

__nv_exec_check_disable__ template <
    class ForwardIt,
    class T = typename cuda::std::iterator_traits<ForwardIt>::value_type>
__host__ __device__ ForwardIt lower_bound(ForwardIt first, ForwardIt last,
                                          const T &value) {
  return mob::compat::lower_bound(first, last, value, cuda::std::less{});
}

template <class ForwardIt,
          class T = typename cuda::std::iterator_traits<ForwardIt>::value_type,
          class Compare>
bool binary_search(ForwardIt first, ForwardIt last, const T &value,
                   Compare comp) {
  first = mob::compat::lower_bound(first, last, value, comp);
  return (!(first == last) and !(comp(value, *first)));
}

template <class ForwardIt,
          class T = typename cuda::std::iterator_traits<ForwardIt>::value_type>
bool binary_search(ForwardIt first, ForwardIt last, const T &value) {
  return mob::compat::binary_search(first, last, value, cuda::std::less{});
}

// https://en.cppreference.com/w/cpp/algorithm/fill
template <class ForwardIt,
          class T = typename cuda::std::iterator_traits<ForwardIt>::value_type>
__host__ __device__ void fill(ForwardIt first, ForwardIt last, const T &value) {
  for (; first != last; ++first)
    *first = value;
}

template <cuda::std::ranges::input_range R, std::weakly_incrementable O>
// requires std::indirectly_copyable<iterator_t<R>, O>
// https://github.com/NVIDIA/cccl/issues/4621
__host__ __device__ O copy(R &&r, O output) {
  auto it = cuda::std::begin(r);
  auto end = cuda::std::end(r);
  while (it != end) {
    *output = *it;
    ++it;
    ++output;
  }
  return output;
}

// https://en.cppreference.com/w/cpp/algorithm/ranges/contains
template <cuda::std::input_iterator I, cuda::std::sentinel_for<I> S, class T>
__host__ __device__ constexpr bool contains(I first, S last, const T &value) {
  for (auto it = first; it != last; ++it) {
    if (*it == value) {
      return true;
    }
  }
  return false;
}

} // namespace compat
} // namespace mob

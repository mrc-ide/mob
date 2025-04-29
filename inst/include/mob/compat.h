#pragma once

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <dust/random/cuda_compatibility.hpp>

namespace mob {
namespace compat {

// cuda::std::distance behaves in weird ways, especially when used with
// Thrust iterators.
__nv_exec_check_disable__ template <typename I, typename S>
__host__ __device__ cuda::std::iter_difference_t<I> distance(I start, S end) {
  if constexpr (cuda::std::sized_sentinel_for<S, I>) {
    return end - start;
  } else {
    cuda::std::iter_difference_t<I> count = 0;
    for (auto it = start; it != end; ++it) {
      count++;
    }
    return count;
  }
}

// range version of distance
template <typename R>
__host__ __device__ auto distance(R &&r) {
  return mob::compat::distance(r.begin(), r.end());
}

// Supposedly cccl has a lower_bound implementation, but it doesn't seem to be
// exposed. `#include <cuda/std/algorithm>` doesn't work.
//
// From https://en.cppreference.com/w/cpp/algorithm/lower_bound
// https://en.cppreference.com/w/cpp/algorithm/binary_search
__nv_exec_check_disable__ template <
    class ForwardIt,
    class T = typename cuda::std::iterator_traits<ForwardIt>::value_type,
    class Compare>
__host__ __device__ ForwardIt lower_bound(ForwardIt first, ForwardIt last,
                                          const T &value, Compare comp) {
  ForwardIt it;
  typename cuda::std::iterator_traits<ForwardIt>::difference_type count, step;
  count = mob::compat::distance(first, last);

  while (count > 0) {
    it = first;
    step = count / 2;
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

template <class T>
using iterator_t = decltype(cuda::std::begin(std::declval<T &>()));

template <class T>
using sentinel_t = decltype(cuda::std::end(std::declval<T &>()));

template <typename R>
using range_reference_t = cuda::std::iter_reference_t<iterator_t<R>>;

template <typename R>
using range_value_t = cuda::std::iter_value_t<iterator_t<R>>;

} // namespace compat
} // namespace mob

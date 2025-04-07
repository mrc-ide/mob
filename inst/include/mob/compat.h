#pragma once

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <dust/random/cuda_compatibility.hpp>

namespace mob {
namespace compat {

// cuda::std::distance behaves in weird ways, especially when used with
// Thrust iterators. We only care about the random access iterators anyway
// which makes this easy enough to implement.
__nv_exec_check_disable__ template <typename RandomIt>
__host__ __device__
    typename cuda::std::iterator_traits<RandomIt>::difference_type
    distance(RandomIt start, RandomIt end) {
  return end - start;
}

// From https://en.cppreference.com/w/cpp/algorithm/lower_bound
// https://en.cppreference.com/w/cpp/algorithm/binary_search
template <class ForwardIt,
          class T = typename cuda::std::iterator_traits<ForwardIt>::value_type,
          class Compare>
ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T &value,
                      Compare comp) {
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

template <class ForwardIt,
          class T = typename cuda::std::iterator_traits<ForwardIt>::value_type>
ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T &value) {
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

} // namespace compat
} // namespace mob

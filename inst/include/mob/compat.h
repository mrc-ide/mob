/**
 * This file provides standard STL functionality that is not yet available to
 * us. Some of it is available in more recent CCCL versions. The implementations
 * here follow the overall spirit of the standard but not necessarily the letter
 * of it.
 *
 * In general we can and do use concepts and type traits from the host STL, but
 * not any runtime code since that wouldn't have been annotated with __device__.
 */

#pragma once

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <dust/random/cuda_compatibility.hpp>

namespace mob {
namespace compat {

// cuda::std::distance exists but it seems to produce incoherent results.
//
// It makes a call to an internal __distance function, but that call uses ADL
// and ends up calling std::__distance which isn't __device__ annotated (why
// don't I get a compiler warning for this? possibly because it is in
// -isystem). Moreover it uses iterator_category for dispatch, but thrust uses
// the standard library type not libcudacxx.
//
// This has been fixed in recent CCCL releases.
//
// https://github.com/NVIDIA/cccl/commit/97f59d27f949946878c3791fcc489e7feb54f3ad
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
template <std::ranges::range R>
__host__ __device__ std::ranges::range_difference_t<R> distance(R &&r) {
  return mob::compat::distance(r.begin(), r.end());
}

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
  count = mob::compat::distance(first, last);

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

template <std::ranges::range T>
using iterator_t = decltype(cuda::std::begin(std::declval<T &>()));

template <std::ranges::range T>
using sentinel_t = decltype(cuda::std::end(std::declval<T &>()));

template <std::ranges::range R>
using range_reference_t = cuda::std::iter_reference_t<iterator_t<R>>;

template <std::ranges::range R>
using range_value_t = cuda::std::iter_value_t<iterator_t<R>>;

template <std::ranges::range R>
using range_difference_t = cuda::std::iter_difference_t<iterator_t<R>>;

template <std::ranges::input_range R, std::weakly_incrementable O>
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

} // namespace compat
} // namespace mob

#include <mob/compat/views.h>

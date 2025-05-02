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

template <typename R>
using range_difference_t = cuda::std::iter_difference_t<iterator_t<R>>;

template <typename T>
constexpr bool enable_view = false;

template <typename R, std::enable_if_t<std::is_object_v<R>, int> = 0>
struct ref_view {
  __host__ __device__ ref_view(R &range) : underlying(&range) {};
  __host__ __device__ compat::iterator_t<R> begin() const {
    return cuda::std::begin(*underlying);
  }
  __host__ __device__ compat::iterator_t<R> end() const {
    return cuda::std::end(*underlying);
  }

private:
  R *underlying;
};

template <typename T>
constexpr bool enable_view<ref_view<T>> = true;

template <typename R, std::enable_if_t<std::is_object_v<R>, int> = 0>
struct owning_view {
  owning_view(R &&range) : underlying(std::move(range)) {};

  owning_view(owning_view &&) = default;
  owning_view &operator=(owning_view &&other) = default;

  compat::iterator_t<const R> begin() const {
    return cuda::std::begin(underlying);
  }
  compat::iterator_t<const R> end() const {
    return cuda::std::end(underlying);
  }
  compat::iterator_t<R> begin() {
    return cuda::std::begin(underlying);
  }
  compat::iterator_t<R> end() {
    return cuda::std::end(underlying);
  }

private:
  R underlying;
};

template <typename T>
constexpr bool enable_view<owning_view<T>> = true;

template <typename W, typename Bound>
struct iota_view {
  struct sentinel {
    __host__ __device__ sentinel() = default;
    __host__ __device__ sentinel(Bound value) : value(std::move(value)) {}

    Bound value = Bound();
  };

  struct iterator {
    using reference = W;
    using value_type = W;

    // FIXME: ideally this would be the signed equivalent of W
    using difference_type = ssize_t;

    __host__ __device__ bool operator==(const iterator &other) const {
      return value == other.value;
    }
    __host__ __device__ bool operator!=(const iterator &other) const {
      return value != other.value;
    }
    __host__ __device__ bool operator==(const sentinel &other) const {
      return value == other.value;
    }
    __host__ __device__ bool operator!=(const sentinel &other) const {
      return value != other.value;
    }
    friend __host__ __device__ bool operator==(const sentinel &other,
                                               const iterator &self) {
      return self.value == other.value;
    }
    friend __host__ __device__ bool operator!=(const sentinel &other,
                                               const iterator &self) {
      return self.value != other.value;
    }

    __host__ __device__ iterator &operator++() {
      ++value;
      return *this;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __host__ __device__ iterator &operator+=(difference_type increment) {
      value += increment;
      return *this;
    }

    __host__ __device__ W operator*() const {
      return value;
    }

    __host__ __device__ iterator(W value) : value(std::move(value)) {}

  private:
    W value;
  };

  __host__ __device__ iota_view(W value, Bound bound)
      : value(std::move(value)), bound(std::move(bound)) {}

  __host__ __device__ iterator begin() const {
    return iterator(value);
  }

  __host__ __device__ sentinel end() const {
    return sentinel(bound);
  }

private:
  W value;
  Bound bound;
};

template <class W, class Bound>
iota_view(W, Bound) -> iota_view<W, Bound>;

template <typename W, typename Bound>
__host__ __device__ auto iota(W &&value, Bound &&bound) {
  return iota_view(std::forward<W>(value), std::forward<Bound>(bound));
}

template <typename W, typename Bound>
constexpr bool enable_view<iota_view<W, Bound>> = true;

template <typename R, typename O>
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

template <typename R>
__host__ __device__ auto all(R &&r) {
  if constexpr (enable_view<std::decay_t<R>>) {
    return std::decay_t<R>(std::forward<R>(r));
  } else if constexpr (std::is_lvalue_reference_v<R>) {
    return ref_view(std::forward<R>(r));
  } else {
    return owning_view(std::forward<R>(r));
  }
}

template <typename R>
using all_t = decltype(mob::compat::all(std::declval<R>()));

} // namespace compat
} // namespace mob

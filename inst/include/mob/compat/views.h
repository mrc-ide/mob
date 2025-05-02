#pragma once

#include <cuda/std/ranges>
#include <dust/random/cuda_compatibility.hpp>

namespace mob::compat {

template <cuda::std::ranges::range R>
  requires std::is_object_v<R>
struct ref_view : cuda::std::ranges::view_interface<ref_view<R>> {
  __host__ __device__ ref_view(R &range) : underlying(&range) {};
  __host__ __device__ cuda::std::ranges::iterator_t<R> begin() const {
    return cuda::std::begin(*underlying);
  }
  __host__ __device__ cuda::std::ranges::iterator_t<R> end() const {
    return cuda::std::end(*underlying);
  }

private:
  R *underlying;
};

template <cuda::std::ranges::range R>
  requires std::movable<R>
struct owning_view : cuda::std::ranges::view_interface<owning_view<R>> {
  owning_view(R &&range) : underlying(std::move(range)) {};

  owning_view(owning_view &&) = default;
  owning_view &operator=(owning_view &&other) = default;

  cuda::std::ranges::iterator_t<const R> begin() const {
    return cuda::std::begin(underlying);
  }
  cuda::std::ranges::iterator_t<const R> end() const {
    return cuda::std::end(underlying);
  }
  cuda::std::ranges::iterator_t<R> begin() {
    return cuda::std::begin(underlying);
  }
  cuda::std::ranges::iterator_t<R> end() {
    return cuda::std::end(underlying);
  }

private:
  R underlying;
};

template <std::weakly_incrementable W,
          std::semiregular Bound = cuda::std::unreachable_sentinel_t>
struct iota_view : cuda::std::ranges::view_interface<iota_view<W, Bound>> {
  struct sentinel {
    sentinel() = default;
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

template <typename R>
__host__ __device__ auto all(R &&r) {
  if constexpr (cuda::std::ranges::enable_view<std::decay_t<R>>) {
    return std::decay_t<R>(std::forward<R>(r));
  } else if constexpr (std::is_lvalue_reference_v<R>) {
    return ref_view(std::forward<R>(r));
  } else {
    return owning_view(std::forward<R>(r));
  }
}

template <typename R>
using all_t = decltype(mob::compat::all(std::declval<R>()));

} // namespace mob::compat

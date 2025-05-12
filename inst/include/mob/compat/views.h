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
  __device__ __host__ owning_view(R &&range) : underlying(std::move(range)) {};

  owning_view(const owning_view &) = delete;
  owning_view &operator=(const owning_view &) = delete;

  owning_view(owning_view &&) = default;
  owning_view &operator=(owning_view &&other) = default;

  __device__ __host__ cuda::std::ranges::iterator_t<const R> begin() const {
    return cuda::std::ranges::begin(underlying);
  }
  __device__ __host__ cuda::std::ranges::sentinel_t<const R> end() const {
    return cuda::std::ranges::end(underlying);
  }
  __device__ __host__ cuda::std::ranges::iterator_t<R> begin() {
    return cuda::std::ranges::begin(underlying);
  }
  __device__ __host__ cuda::std::ranges::sentinel_t<R> end() {
    return cuda::std::ranges::end(underlying);
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

template <cuda::std::ranges::input_range V,
          std::indirect_unary_predicate<cuda::std::ranges::iterator_t<V>> Pred>
  requires cuda::std::ranges::view<V> && std::is_object_v<Pred>
struct filter_view
    : public cuda::std::ranges::view_interface<filter_view<V, Pred>> {
  V underlying;
  Pred pred;

  __host__ __device__ filter_view(V underlying, Pred pred)
      : underlying(std::move(underlying)), pred(std::move(pred)) {}

  struct sentinel;
  struct iterator {
    using reference = cuda::std::ranges::range_reference_t<V>;
    using value_type = cuda::std::ranges::range_value_t<V>;
    using difference_type = ptrdiff_t;

    __host__ __device__ iterator(filter_view *parent,
                                 cuda::std::ranges::iterator_t<V> inner)
        : parent(parent), inner(std::move(inner)) {
      next();
    }

    __host__ __device__ reference operator*() const {
      return *inner;
    }

    __host__ __device__ iterator &operator++() {
      inner++;
      next();
      return *this;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __host__ __device__ bool operator==(sentinel end) const {
      return inner == end.inner;
    }

  private:
    __host__ __device__ void next() {
      while (inner != parent->underlying.end() && !parent->pred(*inner)) {
        inner++;
      }
    }

    filter_view *parent;
    cuda::std::ranges::iterator_t<V> inner;
  };

  struct sentinel {
    cuda::std::ranges::sentinel_t<V> inner;
  };

  static_assert(std::input_iterator<iterator>);
  static_assert(std::sentinel_for<sentinel, iterator>);

  __host__ __device__ iterator begin() {
    return iterator{this, underlying.begin()};
  }

  __host__ __device__ sentinel end() {
    return sentinel{underlying.end()};
  }
};

template <cuda::std::ranges::input_range V,
          std::indirect_unary_predicate<cuda::std::ranges::iterator_t<V>> Pred>
filter_view(V &&, Pred &&) -> filter_view<compat::all_t<V>, Pred>;

template <cuda::std::ranges::input_range V,
          std::indirect_unary_predicate<cuda::std::ranges::iterator_t<V>> Pred>
__host__ __device__ auto filter(V &&range, Pred &&pred) {
  return filter_view(std::forward<V>(range), std::forward<Pred>(pred));
}

} // namespace mob::compat

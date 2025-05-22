#pragma once

#include <cuda/std/functional>
#include <cuda/std/optional>
#include <cuda/std/ranges>
#include <dust/random/cuda_compatibility.hpp>

namespace mob::compat {

/**
 * Ranges needs to be move-assignable, but lambda closure are not despite being
 * move-constructible.
 *
 * This class wraps any move-constructible type and makes it move-assignable.
 * The assignment operator does a destroy followed by a constructor.
 *
 * TODO: provide a no-op wrapper when T is move-assignable already.
 */
template <cuda::std::move_constructible T>
  requires cuda::std::is_object_v<T>
struct movable_box {
  __host__ __device__ movable_box(T &&value) : inner(std::move(value)) {}
  __host__ __device__ movable_box(movable_box &&other) = default;

  __host__ __device__ movable_box(movable_box &other)
    requires cuda::std::copy_constructible<T>
  = default;

  __host__ __device__ movable_box<T> &operator=(const movable_box &other) {
    if (this != cuda::std::addressof(other)) {
      if (other.inner.has_value()) {
        inner.emplace(*other);
      } else {
        inner.reset();
      }
    }
    return *this;
  }

  __host__ __device__ movable_box<T> &operator=(movable_box &&other) {
    if (this != cuda::std::addressof(other)) {
      if (other.inner.has_value()) {
        inner.emplace(cuda::std::move(*other));
      } else {
        inner.reset();
      }
    }
    return *this;
  }

  __host__ __device__ T &operator*() {
    return *inner;
  }

  __host__ __device__ const T &operator*() const {
    return *inner;
  }

  cuda::std::optional<T> inner;
};

template <typename D>
  requires std::is_object_v<D> && std::same_as<D, std::remove_cv_t<D>>
class range_adaptor_closure {};

template <typename Fn>
struct pipeable : Fn, range_adaptor_closure<pipeable<Fn>> {
  __host__ __device__ explicit pipeable(Fn &&fn) : Fn(std::move(fn)) {}
};

template <typename Fn>
pipeable(Fn &&fn) -> pipeable<Fn>;

template <cuda::std::ranges::viewable_range R, typename T>
  requires std::derived_from<
               cuda::std::remove_cvref_t<T>,
               range_adaptor_closure<cuda::std::remove_cvref_t<T>>> &&
           cuda::std::invocable<T, R>
__host__ __device__ auto operator|(R &&range, T &&adaptor) {
  return cuda::std::invoke(std::forward<T>(adaptor), std::forward<R>(range));
}

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

  __device__ __host__ cuda::std::ranges::iterator_t<R> begin() {
    return cuda::std::ranges::begin(underlying);
  }

  __device__ __host__ cuda::std::ranges::sentinel_t<R> end() {
    return cuda::std::ranges::end(underlying);
  }

  __device__ __host__ auto begin() const
    requires cuda::std::ranges::range<const R>
  {
    return cuda::std::ranges::begin(underlying);
  }

  __device__ __host__ auto end() const
    requires cuda::std::ranges::range<const R>
  {
    return cuda::std::ranges::end(underlying);
  }

private:
  R underlying;
};

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

template <cuda::std::ranges::viewable_range R>
using all_t = decltype(mob::compat::all(std::declval<R>()));

template <cuda::std::ranges::input_range V,
          std::indirect_unary_predicate<cuda::std::ranges::iterator_t<V>> Pred>
  requires cuda::std::ranges::view<V> && std::is_object_v<Pred>
struct filter_view
    : public cuda::std::ranges::view_interface<filter_view<V, Pred>> {
  V underlying;
  movable_box<Pred> pred;

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
      while (inner != parent->underlying.end() && !(*parent->pred)(*inner)) {
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

template <typename R, typename Pred>
filter_view(R &&, Pred) -> filter_view<compat::all_t<R>, Pred>;

struct filter_cpo {
  template <
      cuda::std::ranges::viewable_range R,
      cuda::std::indirect_unary_predicate<cuda::std::ranges::iterator_t<R>>
          Pred>
  __host__ __device__ auto operator()(R &&range, Pred &&pred) const {
    return filter_view(std::forward<R>(range), std::forward<Pred>(pred));
  }

  template <typename Pred>
  __host__ __device__ auto operator()(Pred &&pred) const {
    return pipeable{cuda::std::__bind_back(*this, std::forward<Pred>(pred))};
  }
};

__host__ __device__ static constexpr filter_cpo filter;

template <cuda::std::ranges::input_range V, std::move_constructible F>
  requires cuda::std::ranges::view<V> && cuda::std::is_object_v<F> &&
           cuda::std::regular_invocable<F &,
                                        cuda::std::ranges::range_reference_t<V>>
struct transform_view
    : public cuda::std::ranges::view_interface<transform_view<V, F>> {
  V underlying;
  movable_box<F> func;

  __host__ __device__ transform_view(V underlying, F f)
      : underlying(std::move(underlying)), func(std::move(f)) {}

  struct sentinel;
  struct iterator {
    using value_type = std::remove_cvref_t<
        std::invoke_result_t<F &, cuda::std::ranges::range_reference_t<V>>>;
    using difference_type = cuda::std::ranges::range_difference_t<V>;

    __host__ __device__ iterator(transform_view *parent,
                                 cuda::std::ranges::iterator_t<V> inner)
        : parent(parent), inner(std::move(inner)) {}

    __host__ __device__ value_type operator*() const {
      return (*parent->func)(*inner);
    }

    __host__ __device__ iterator &operator++() {
      inner++;
      return *this;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __host__ __device__ bool operator==(sentinel end) const {
      return inner == end.inner;
    }

  private:
    transform_view *parent;
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

template <class R, class F>
transform_view(R &&, F) -> transform_view<compat::all_t<R>, F>;

struct transform_cpo {
  template <cuda::std::ranges::viewable_range R, class F>
  __host__ __device__ auto operator()(R &&range, F &&f) const {
    return transform_view(std::forward<R>(range), std::forward<F>(f));
  }

  template <typename F>
  __host__ __device__ auto operator()(F &&f) const {
    return pipeable{cuda::std::__bind_back(*this, std::forward<F>(f))};
  }
};

__host__ __device__ static constexpr transform_cpo transform;

template <cuda::std::ranges::input_range V>
  requires cuda::std::ranges::view<V> &&
           cuda::std::ranges::input_range<
               cuda::std::ranges::range_reference_t<V>>
struct join_view {
  using OuterIterator = cuda::std::ranges::iterator_t<V>;
  using InnerRange = cuda::std::ranges::range_reference_t<V>;
  using InnerIterator = cuda::std::ranges::iterator_t<InnerRange>;

  struct iterator {
    join_view *parent;
    OuterIterator outer_it;
    cuda::std::optional<InnerRange> inner_range = cuda::std::nullopt;
    cuda::std::optional<InnerIterator> inner_it = cuda::std::nullopt;

    using value_type = cuda::std::ranges::range_value_t<InnerRange>;
    using difference_type = cuda::std::common_type_t<
        cuda::std::ranges::range_difference_t<V>,
        cuda::std::ranges::range_difference_t<InnerRange>>;

    __host__ __device__ iterator(join_view *parent, OuterIterator outer_it)
        : parent(parent), outer_it(outer_it) {
      satisfy();
    }

    __host__ __device__ decltype(auto) operator*() const {
      return **inner_it;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __host__ __device__ iterator &operator++() {
      if (++(*inner_it) == cuda::std::ranges::end(*inner_range)) {
        ++outer_it;
        inner_range.reset();
        satisfy();
      }
      return *this;
    }

    __host__ __device__ void satisfy() {
      for (; outer_it != cuda::std::ranges::end(parent->underlying);
           ++outer_it) {
        inner_range = *outer_it;
        inner_it = cuda::std::ranges::begin(*inner_range);
        if (inner_it != cuda::std::ranges::end(*inner_range)) {
          return;
        }
      }
      inner_it.reset();
    }
  };
  struct sentinel {
    cuda::std::ranges::sentinel_t<V> end_ = {};

    __host__ __device__ bool operator==(const iterator &it) const {
      return it.outer_it == end_;
    }
  };
  static_assert(std::input_iterator<iterator>);
  static_assert(std::sentinel_for<sentinel, iterator>);

  __host__ __device__ join_view(V underlying)
      : underlying(std::move(underlying)) {}

  __host__ __device__ iterator begin() {
    return iterator{this, cuda::std::ranges::begin(underlying)};
  }

  __host__ __device__ sentinel end() {
    return sentinel{cuda::std::ranges::end(underlying)};
  }

private:
  V underlying;
};

template <class R>
join_view(R &&) -> join_view<compat::all_t<R>>;

struct join_cpo : range_adaptor_closure<join_cpo> {
  template <cuda::std::ranges::viewable_range R>
  __host__ __device__ auto operator()(R &&range) const {
    return join_view(std::forward<R>(range));
  }
};

__host__ __device__ static constexpr join_cpo join;

template <cuda::std::ranges::input_range V>
struct known_size_view : cuda::std::ranges::view_interface<known_size_view<V>> {
  using size_type = std::ranges::range_difference_t<V>;

  __host__ __device__ known_size_view(V underlying, size_type size)
      : underlying_(std::move(underlying)), size_(size) {}

  struct iterator {
    using difference_type = size_type;
    using value_type = cuda::std::ranges::range_value_t<V>;
    using reference = cuda::std::ranges::range_reference_t<V>;

    std::ranges::iterator_t<V> underlying;
    size_type offset;

    __host__ __device__ reference operator*() const {
      return *underlying;
    }

    __host__ __device__ size_type operator-(const iterator &other) const {
      return offset - other.offset;
    };

    __host__ __device__ iterator &operator++() {
      ++underlying;
      offset += 1;
      return *this;
    };

    __host__ __device__ void operator++(int) {
      ++this;
    }

    __host__ __device__ iterator &operator+=(size_type diff) {
      underlying += diff;
      offset += diff;
      return *this;
    };

    __host__ __device__ iterator &operator-=(size_type diff) {
      underlying -= diff;
      offset -= diff;
      return *this;
    };

    __host__ __device__ bool operator==(const iterator &other) const {
      return underlying == other.underlying;
    }
  };

  using sentinel = iterator;

  static_assert(std::input_iterator<iterator>);
  static_assert(std::sentinel_for<sentinel, iterator>);
  static_assert(std::sized_sentinel_for<sentinel, iterator>);

  __host__ __device__ iterator begin() {
    return iterator{underlying_.begin(), 0};
  }

  __host__ __device__ iterator end() {
    return iterator{underlying_.end(), size_};
  }

  __host__ __device__ size_type size() const {
    return size_;
  };

private:
  V underlying_;
  size_type size_;
};

template <typename R>
known_size_view(R &&, std::ranges::range_difference_t<R>)
    -> known_size_view<compat::all_t<R>>;

template <cuda::std::ranges::viewable_range R>
__host__ __device__ auto known_size(R &&range,
                                    std::ranges::range_difference_t<R> size) {
  return known_size_view(std::forward<R>(range), size);
}

} // namespace mob::compat

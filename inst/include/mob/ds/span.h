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

template <typename R, std::enable_if_t<std::is_object_v<R>, int> = 0>
struct ref_view {
  ref_view(R &range) : underlying(&range) {};
  compat::iterator_t<R> begin() const {
    return cuda::std::begin(*underlying);
  }
  compat::iterator_t<R> end() const {
    return cuda::std::end(*underlying);
  }

private:
  R *underlying;
};

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
constexpr bool enable_view = false;

template <typename System, typename T>
constexpr bool enable_view<span<System, T>> = true;

template <typename T>
constexpr bool enable_view<ref_view<T>> = true;

template <typename T>
constexpr bool enable_view<owning_view<T>> = true;

template <typename R>
auto all(R &&r) {
  if constexpr (enable_view<std::decay_t<R>>) {
    return std::decay_t<R>(std::forward<R>(r));
  } else if constexpr (std::is_lvalue_reference_v<R>) {
    return ref_view(std::forward<R>(r));
  } else {
    return owning_view(std::forward<R>(r));
  }
}

template <typename R>
using all_t = decltype(ds::all(std::declval<R>()));

} // namespace ds
} // namespace mob

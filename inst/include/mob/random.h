#pragma once

#include <concepts>
#include <cstddef>
#include <dust/random/cuda_compatibility.hpp>
#include <dust/random/generator.hpp>
#include <thrust/device_reference.h>

namespace mob {

template <typename T>
concept random_state = requires(T rng) {
  { next(rng) } -> std::integral;
};

template <typename T>
concept random_state_storage = requires(T rng, size_t i) {
  requires random_state<T>;
  requires std::integral<typename T::int_type>;
  { next(rng) } -> std::same_as<typename T::int_type>;
  { T::size() } -> std::same_as<size_t>;
  { rng[i] } -> std::same_as<typename T::int_type &>;
};

template <std::integral T>
struct widen_type;

template <std::integral T>
using widen_type_t = widen_type<T>::type;

template <>
struct widen_type<uint8_t> {
  using type = uint16_t;
};

template <>
struct widen_type<uint16_t> {
  using type = uint32_t;
};

template <>
struct widen_type<uint32_t> {
  using type = uint64_t;
};

template <>
struct widen_type<uint64_t> {
  using type = __uint128_t;
};

// https://arxiv.org/pdf/1805.10941
template <std::integral T, random_state R>
__host__ __device__ T random_bounded_int(R &rng, T s) {
  using W = widen_type_t<T>;
  T x = dust::random::random_int<T>(rng);
  W m = static_cast<W>(x) * static_cast<W>(s);
  T l = static_cast<T>(m);
  if (l < s) {
    uint64_t t = -s % s;
    while (l < t) {
      x = dust::random::random_int<uint64_t>(rng);
      m = static_cast<W>(x) * static_cast<W>(s);
      l = static_cast<T>(m);
    }
  }
  return m >> (8 * sizeof(T));
}

template <std::integral T, random_state R>
__host__ __device__ T random_bounded_int(R &rng, T lower, T upper) {
  return random_bounded_int<T>(rng, upper - lower) + lower;
}

} // namespace mob

#ifdef __NVCC__

namespace thrust {

template <typename T>
__device__ decltype(next(std::declval<T &>())) next(device_reference<T> rng) {
  return next(thrust::raw_reference_cast(rng));
}

} // namespace thrust
#endif

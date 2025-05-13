#pragma once

#include <concepts>
#include <cstddef>
#include <dust/random/cuda_compatibility.hpp>
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

} // namespace mob

#ifdef __NVCC__

namespace thrust {

template <typename T>
__device__ decltype(next(std::declval<T &>())) next(device_reference<T> rng) {
  return next(thrust::raw_reference_cast(rng));
}

} // namespace thrust
#endif

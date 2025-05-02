#pragma once
#include <mob/ds/span.h>
#include <mob/parallel_random.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace mob {
namespace system {
struct host {
  template <typename T>
  using vector = thrust::host_vector<T>;

  template <typename T>
  using pointer = T *;

  template <typename T>
  using span = ds::span<host, T>;

  using random = host_random;
};

struct device {
  template <typename T>
  using vector = thrust::device_vector<T>;

  template <typename T>
  using pointer = thrust::device_ptr<T>;

  template <typename T>
  using span = ds::span<device, T>;

  using random = device_random;
};
} // namespace system

namespace ds {

template <typename T>
span(thrust::device_vector<T>) -> span<system::device, T>;

template <typename T>
span(thrust::host_vector<T>) -> span<system::host, T>;

template <typename T>
span(std::vector<T>) -> span<system::host, T>;

} // namespace ds
}; // namespace mob

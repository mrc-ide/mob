#pragma once
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

  using random = host_random;
};

struct device {
  template <typename T>
  using vector = thrust::device_vector<T>;

  template <typename T>
  using pointer = thrust::device_ptr<T>;

  using random = device_random;
};
} // namespace system
}; // namespace mob

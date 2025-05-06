#pragma once
#include <mob/ds/span.h>
#include <mob/parallel_random.h>

#include <thrust/device_make_unique.h>
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

  template <typename T, typename... Args>
  static std::unique_ptr<T> make_unique(Args &&...args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename F>
  static std::invoke_result_t<F> execute(F &&f) {
    return std::forward<F>(f)();
  };
};

struct device {
  template <typename T>
  using vector = thrust::device_vector<T>;

  template <typename T>
  using pointer = thrust::device_ptr<T>;

  template <typename T>
  using span = ds::span<device, T>;

  using random = device_random;

  template <typename T, typename... Args>
  static auto make_unique(Args &&...args) {
    return thrust::device_make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename F>
  static std::invoke_result_t<F> execute(F &&f) {
    if constexpr (std::is_void_v<std::invoke_result_t<F>>) {
      thrust::for_each(
          thrust::counting_iterator(0), thrust::counting_iterator(1),
          [f = std::forward<F>(f)] __device__(int) -> void { f(); });
    } else {
      auto result = thrust::device_make_unique<std::invoke_result_t<F>>();
      thrust::generate(result.get(), result.get() + 1, std::forward<F>(f));
      return std::move(*result.get());
    }
  }
};

} // namespace system

template <typename System, typename T, typename... Args>
auto make_unique(Args &&...args) {
  return System::template make_unique<T>(std::forward<Args>(args)...);
}

template <typename System, typename F>
std::invoke_result_t<F> execute(F &&f) {
  return System::execute(std::forward<F>(f));
}

template <typename System, typename T>
using vector = typename System::vector<T>;

namespace ds {

template <typename T>
span(thrust::device_vector<T>) -> span<system::device, T>;

template <typename T>
span(thrust::host_vector<T>) -> span<system::host, T>;

template <typename T>
span(std::vector<T>) -> span<system::host, T>;

} // namespace ds
}; // namespace mob

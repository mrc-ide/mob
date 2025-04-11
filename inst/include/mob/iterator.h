#pragma once

#include <dust/random/cuda_compatibility.hpp>
#include <utility>

namespace mob {

template <typename T>
__host__ __device__ struct counting_output_iterator {
  struct proxy;

  __host__ __device__ counting_output_iterator(T offset = 0)
      : offset_(offset) {}

  __host__ __device__ proxy operator*() {
    return proxy();
  }

  __host__ __device__ counting_output_iterator &operator++() {
    offset_++;
    return *this;
  }

  __host__ __device__ counting_output_iterator operator++(int) {
    counting_output_iterator temp = *this;
    ++(*this);
    return temp;
  }

  __host__ __device__ T offset() const {
    return offset_;
  }

  struct proxy {
    template <typename U>
    __host__ __device__ void operator=(U &&) {}
  };

private:
  T offset_;
};

// https://devblogs.microsoft.com/oldnewthing/20250129-00/?p=110817
template <typename Container>
struct default_insert_iterator {
  default_insert_iterator(Container &c) : container(&c) {}

  default_insert_iterator &operator*() {
    return *this;
  }
  default_insert_iterator &operator++() {
    return *this;
  }
  default_insert_iterator &operator=(const typename Container::value_type &v) {
    container->insert(v);
    return *this;
  }
  default_insert_iterator &operator=(typename Container::value_type &&v) {
    container->insert(std::move(v));
    return *this;
  }

  Container *container;
};

template <typename Container>
default_insert_iterator<Container>
default_inserter(Container &container) noexcept {
  return default_insert_iterator<Container>(container);
}

} // namespace mob

#pragma once

#include "mob/compat.h"
#include "mob/system.h"

namespace mob {
namespace roaring {
template <typename System = system::host>
struct container_array {
  using value_type = uint16_t;
  using iterator = mob::vector<System, uint16_t>::const_iterator;

  container_array() {}
  explicit container_array(uint16_t index) : values({index}) {}

  template <typename Other>
  explicit container_array(const container_array<Other> &other)
      : values(other.values) {
    // TODO: do we want a `requires (!std::is_same_v<System, Other>()`?
  }

  template <typename ForwardIt>
  explicit container_array(ForwardIt first, ForwardIt last) {
    // TODO: pre-allocate the array if ForwardIt has a well-known distance.
    // TODO: if we assume the input is sorted, this can just be push backs.
    for (auto it = first; it != last; it++) {
      insert(*it);
    }
  }

  iterator begin() const {
    return values.cbegin();
  }

  iterator end() const {
    return values.cend();
  }

  uint32_t size() const {
    return values.size();
  }

  bool contains(uint16_t index) const {
    return std::binary_search(begin(), end(), index);
  }

  mob::vector<System, uint16_t>::iterator lower_bound(uint16_t index) {
    return std::lower_bound(values.begin(), values.end(), index);
  }

  void insert(uint16_t index) {
    auto it = this->lower_bound(index);
    if (it == values.end() || *it != index) {
      values.insert(it, index);
    }
  }

  // position must be correct and preserve the ordering, ie. it is not just a
  // hint. use lower_bound() to find a suitable position.
  void insert(mob::vector<System, uint16_t>::iterator position,
              uint16_t index) {
    // Annoyingly, thrust's vectors require an iterator as the first argument
    // rather than a const_iterator. This is contrary to the STL which accepts
    // a const_iterator.
    //
    // This means `lower_bound()` needs to return a vector::iterator so it can
    // be passed back to this function, which the caller could modify. We could
    // defend against this by defining a read-only wrapper around vector's
    // iterator.
    values.insert(position, index);
  }

  mob::vector<System, uint16_t> values;
};

template <typename System = system::host>
struct container_array_view {
  using iterator = typename System::pointer<const uint16_t>;

  container_array_view(const container_array<System> &container)
      : data_(container.values.data()), size_(container.values.size()) {}

  __host__ __device__ iterator begin() const {
    return data_;
  }

  __host__ __device__ iterator end() const {
    return data_ + size_;
  }

  __host__ __device__ uint32_t size() const {
    return size_;
  }

  __host__ __device__ bool contains(uint16_t index) const {
    return compat::binary_search(begin(), end(), index);
  }

private:
  typename System::pointer<const uint16_t> data_;
  uint32_t size_;
};
} // namespace roaring
} // namespace mob

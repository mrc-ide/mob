#pragma once

#include "mob/bits.h"
#include "mob/system.h"

namespace mob {
namespace roaring {

template <typename System = system::host>
struct container_bitmap {
  using value_type = uint16_t;
  using word_type = uint64_t;
  static inline constexpr uint8_t word_size = sizeof(word_type) * 8;
  static inline constexpr uint32_t capacity = 65536;

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = uint16_t;
    using reference = uint16_t;
    using pointer = const value_type *;
    using difference_type = ptrdiff_t;

    iterator &operator++() {
      if (position < capacity - 1) {
        position = parent->next_position(position + 1, 0);
      } else {
        position = capacity;
      }
      return *this;
    }

    bool operator==(const iterator &other) const {
      return position == other.position;
    }

    bool operator!=(const iterator &other) const {
      return position != other.position;
    }

    uint16_t operator*() const {
      return position;
    }

    const container_bitmap<System> *parent;
    uint32_t position;
  };

  container_bitmap() : data(capacity / word_size) {}

  explicit container_bitmap(const container_array<> &array)
      : data(capacity / word_size) {
    // TODO: There's probably faster ways of doing this
    for (uint16_t v : array) {
      insert(v);
    }
  }

  template <typename Other>
  explicit container_bitmap(const container_bitmap<Other> &other)
      : data(other.data) {}

  template <typename ForwardIt>
  explicit container_bitmap(ForwardIt first, ForwardIt last)
      : container_bitmap() {
    for (auto it = first; it != last; it++) {
      insert(*it);
    }
  }

  bool contains(uint16_t index) const {
    uint16_t bucket = index / word_size;
    uint16_t excess = index % word_size;
    word_type mask = (static_cast<word_type>(1) << excess);
    return (data[bucket] & mask) != 0;
  }

  void insert(uint16_t index) {
    uint16_t bucket = index / word_size;
    uint16_t excess = index % word_size;
    word_type mask = (static_cast<word_type>(1) << excess);
    data[bucket] |= mask;
  }

  iterator begin() const {
    return iterator{this, next_position(0, 0)};
  }

  iterator end() const {
    return iterator{this, capacity};
  }

  uint32_t size() const {
    uint32_t sum = 0;
    for (auto v : data) {
      sum += bits::popcount(v);
    }
    return sum;
  }

  uint32_t next_position(uint16_t p, uint16_t n) const {
    uint16_t bucket = p / word_size;
    uint16_t excess = p % word_size;

    word_type word = data[bucket] >> (uint32_t)excess;
    while (n >= bits::popcount(word) && bucket + 1U < data.size()) {
      n -= bits::popcount(word);
      bucket += 1;
      word = data[bucket];
      excess = 0;
    }

    uint8_t r = bits::select(word, n);
    return std::min<uint32_t>(bucket * word_size + excess + r, capacity);
  }

public:
  mob::vector<System, word_type> data;
};

template <typename System = system::host>
struct container_bitmap_view {
  using word_type = typename container_bitmap<System>::word_type;
  static inline constexpr uint8_t word_size =
      container_bitmap<System>::word_size;
  static inline constexpr uint32_t capacity =
      container_bitmap<System>::capacity;

  container_bitmap_view(const container_bitmap<System> &container)
      : data_(container.data.data()) {}

  __host__ __device__ bool contains(uint16_t index) const {
    uint16_t bucket = index / word_size;
    uint16_t excess = index % word_size;
    word_type mask = (static_cast<word_type>(1) << excess);
    return (data_[bucket] & mask) != 0;
  }

private:
  typename System::pointer<const word_type> data_;
};

} // namespace roaring
} // namespace mob

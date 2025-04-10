#pragma once
#include "bits.h"
#include "iterator.h"
#include <algorithm>
#include <cinttypes>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

namespace mob {
namespace roaring {

// https://quuxplusone.github.io/blog/2019/02/06/arrow-proxy/
template <typename T>
struct arrow_proxy {
  T value;
  T *operator->() {
    return &value;
  }
};

template <typename KeyIt, typename... ValueIt>
struct map_of_arrays_iterator {
  KeyIt key;
  std::tuple<ValueIt...> values;

  using iterator_category = std::forward_iterator_tag;
  using difference_type = typename KeyIt::difference_type;
  using value_type =
      std::tuple<typename KeyIt::value_type, typename ValueIt::value_type...>;
  using reference =
      std::tuple<typename KeyIt::reference, typename ValueIt::reference...>;
  using pointer = arrow_proxy<reference>;

  bool operator==(const map_of_arrays_iterator &other) const {
    return key == other.key;
  }

  bool operator!=(const map_of_arrays_iterator &other) const {
    return key != other.key;
  }

  difference_type operator-(const map_of_arrays_iterator &other) const {
    return key - other.key;
  }

  reference operator*() const {
    return std::apply(
        [this](auto &...values) { return std::tie(*key, *values...); }, values);
  }

  pointer operator->() const {
    return pointer{**this};
  }

  map_of_arrays_iterator &operator++() {
    key++;
    std::apply([this](auto &...values) { ((values++), ...); }, values);
    return *this;
  }
};

template <typename Key, typename... Ts>
struct map_of_arrays {

  using iterator =
      map_of_arrays_iterator<typename std::vector<Key>::iterator,
                             typename std::vector<Ts>::iterator...>;

  using const_iterator =
      map_of_arrays_iterator<typename std::vector<Key>::const_iterator,
                             typename std::vector<Ts>::const_iterator...>;

  iterator begin() {
    return apply([this](auto &...values) {
      return iterator{keys_.begin(), std::make_tuple(values.begin()...)};
    });
  }

  iterator end() {
    return apply([this](auto &...values) {
      return iterator{keys_.end(), std::make_tuple(values.end()...)};
    });
  }

  iterator lower_bound(const Key &key) {
    return apply([this, &key](auto &...values) {
      auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
      std::ptrdiff_t diff = std::distance(keys_.begin(), it);
      return iterator{it, std::make_tuple((values.begin() + diff)...)};
    });
  }

  iterator find(const Key &key) {
    auto it = lower_bound(key);
    if (it != end() && std::get<0>(*it) == key) {
      return it;
    } else {
      return end();
    }
  }

  const_iterator begin() const {
    return apply([this](const auto &...values) {
      return const_iterator{keys_.begin(), std::make_tuple(values.begin()...)};
    });
  }

  const_iterator end() const {
    return apply([this](const auto &...values) {
      return const_iterator{keys_.end(), std::make_tuple(values.end()...)};
    });
  }

  const_iterator lower_bound(const Key &key) const {
    return apply([this, &key](const auto &...values) {
      auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
      std::ptrdiff_t diff = std::distance(keys_.begin(), it);
      return const_iterator{it, std::make_tuple((values.begin() + diff)...)};
    });
  }

  const_iterator find(const Key &key) const {
    auto it = lower_bound(key);
    if (it != end() && std::get<0>(*it) == key) {
      return it;
    } else {
      return end();
    }
  }

  template <std::size_t... I>
  void insert(std::index_sequence<I...>, iterator position, Key key,
              std::tuple<Ts...> values) {
    keys_.insert(position.key, key);
    ((std::get<I>(values_).insert(std::get<I>(position.values),
                                  std::get<I>(values))),
     ...);
  }

  void insert(iterator position, Key key, Ts... values) {
    insert(std::make_index_sequence<sizeof...(Ts)>(), position, key,
           std::make_tuple(values...));
  }

private:
  template <typename Fn>
  auto apply(Fn &&fn) {
    return std::apply(std::forward<Fn>(fn), values_);
  }

  template <typename Fn>
  auto apply(Fn &&fn) const {
    return std::apply(std::forward<Fn>(fn), values_);
  }

  std::vector<Key> keys_;
  std::tuple<std::vector<Ts>...> values_;
};

// TODO: Should we use a bitmap to store the kind instead, or use the low/high
// bits of the void* pointer?
enum class container_kind : uint8_t {
  ARRAY,
  BITMAP,
};

struct container_array {
  using iterator = std::vector<uint16_t>::const_iterator;

  container_array() {}
  explicit container_array(uint16_t index) : values({index}) {}

  bool find(uint16_t index) const {
    return std::binary_search(values.begin(), values.end(), index);
  }

  bool insert(uint16_t index) {
    auto it = std::lower_bound(values.begin(), values.end(), index);
    if (it == values.end() || *it != index) {
      if (values.size() == 4096) {
        return false;
      }
      values.insert(it, index);
    }
    return true;
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

  std::vector<uint16_t> values;
};

struct container_bitmap {
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

    const container_bitmap *parent;
    uint32_t position;
  };

  container_bitmap() : data(capacity / word_size) {}

  explicit container_bitmap(const container_array &array)
      : data(capacity / word_size) {
    // TODO: There's probably faster ways of doing this
    for (uint16_t v : array.values) {
      insert(v);
    }
  }

  bool find(uint16_t index) const {
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
  std::vector<word_type> data;
};

struct bitset {
  using value_type = uint32_t;

  bitset() = default;

  template <typename Iterator>
  bitset(Iterator start, Iterator end) {
    for (auto it = start; it != end; it++) {
      insert(*it);
    }
  }

  ~bitset() {
    for (auto it = containers_.begin(); it != containers_.end(); ++it) {
      switch (std::get<1>(*it)) {
      case container_kind::BITMAP:
        delete static_cast<container_bitmap *>(std::get<2>(*it));
        break;
      case container_kind::ARRAY:
        delete static_cast<container_array *>(std::get<2>(*it));
        break;
      }
    }
  }

  bitset(const bitset &) = delete;
  bitset &operator=(const bitset &) = delete;

  bitset(bitset &&) = default;
  bitset &operator=(bitset &&) = default;

  bool find(uint32_t index) {
    uint16_t high = index >> 16;
    uint16_t low = index & 0xffff;

    auto it = containers_.find(high);
    if (it == containers_.end()) {
      return false;
    } else {
      return dispatch(*it,
                      [low](const auto &c) -> bool { return c.find(low); });
    }
  }

  void insert(uint32_t index) {
    uint16_t high = index >> 16;
    uint16_t low = index & 0xffff;

    auto it = containers_.lower_bound(high);
    if (it == containers_.end() || std::get<0>(*it) != high) {
      container_array *c = new container_array(low);
      containers_.insert(it, high, container_kind::ARRAY, c);
    } else if (std::get<1>(*it) == container_kind::ARRAY) {
      auto *array = static_cast<container_array *>(std::get<2>(*it));
      if (!array->insert(low)) {
        auto *bitmap = new container_bitmap(*array);
        bitmap->insert(low);

        std::get<1>(*it) = container_kind::BITMAP;
        std::get<2>(*it) = bitmap;
        delete array;
      }
    } else if (std::get<1>(*it) == container_kind::BITMAP) {
      auto *bitmap = static_cast<container_bitmap *>(std::get<2>(*it));
      bitmap->insert(low);
    }
  }

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = uint32_t;
    using reference = value_type;
    using pointer = const value_type *;
    using difference_type = ptrdiff_t;

    map_of_arrays<uint16_t, container_kind, void *>::const_iterator toplevel;

    // std::monostate is a placeholder for `toplevel->container.begin()`, but
    // it works even in cases where toplevel is the `end()` of the
    // map_of_arrays.
    //
    // We can only call `container.begin()` in cases where we know the iterator
    // is valid.
    std::variant<container_array::iterator, container_bitmap::iterator,
                 std::monostate>
        nested;

    template <typename Container>
    typename Container::iterator downcast(Container &container) const {
      if (std::holds_alternative<std::monostate>(nested)) {
        return container.begin();
      } else {
        return std::get<typename Container::iterator>(nested);
      }
    }

    template <typename Container>
    typename Container::iterator &downcast(Container &container) {
      if (std::holds_alternative<std::monostate>(nested)) {
        nested = container.begin();
      }
      return std::get<typename Container::iterator>(nested);
    }

    uint32_t operator*() const {
      uint16_t high = std::get<0>(*toplevel);
      uint16_t low = dispatch(*toplevel, [this](const auto &container) {
        return *downcast(container);
      });

      return (static_cast<uint32_t>(high) << 16) | low;
    }

    iterator &operator++() {
      dispatch(*toplevel, [this](const auto &container) {
        auto &it = downcast(container);
        if (++it == container.end()) {
          ++toplevel;
          nested = std::monostate();
        }
      });
      return *this;
    }

    iterator operator++(int) {
      iterator temp = *this;
      ++(*this);
      return temp;
    }

    iterator &operator+=(size_t n) {
      // TODO: this could definitely be sped up, both container types can
      // support faster skips. The issue comes up when you hit the end of
      // one container and need to break n into parts.
      for (size_t i = 0; i < n; i++) {
        (*this)++;
      }
      return *this;
    }

    bool operator==(const iterator &other) const {
      if (toplevel == other.toplevel) {
        if (std::holds_alternative<std::monostate>(nested) ==
            std::holds_alternative<std::monostate>(other.nested)) {
          return true;
        } else {
          return dispatch(*toplevel, [this, &other](auto &container) {
            auto it = downcast(container);
            auto other_it = other.downcast(container);
            return it == other_it;
          });
        }
      } else {
        return false;
      }
    }

    bool operator!=(const iterator &other) const {
      return !(*this == other);
    }
  };

  iterator begin() const {
    // Containers are never empty so using the first one is always valid
    return iterator{containers_.begin(), std::monostate()};
  }

  iterator end() const {
    return iterator{containers_.end(), std::monostate()};
  }

  uint64_t size() const {
    uint64_t sum = 0;
    for (auto entry : containers_) {
      dispatch(entry, [&](const auto &container) { sum += container.size(); });
    }
    return sum;
  }

private:
  template <typename Fn>
  static std::common_type_t<std::invoke_result_t<Fn, container_array &>,
                            std::invoke_result_t<Fn, container_bitmap &>>
  dispatch(std::tuple<uint16_t &, container_kind &, void *&> it, Fn &&fn) {
    switch (std::get<1>(it)) {
    case container_kind::ARRAY: {
      auto *ptr = static_cast<container_array *>(std::get<2>(it));
      return std::forward<Fn>(fn)(*ptr);
    }
    case container_kind::BITMAP: {
      auto *ptr = static_cast<container_bitmap *>(std::get<2>(it));
      return std::forward<Fn>(fn)(*ptr);
    }
    }

    throw std::logic_error("unreachable");
  }

  template <typename Fn>
  static std::common_type_t<std::invoke_result_t<Fn, const container_array &>,
                            std::invoke_result_t<Fn, const container_bitmap &>>
  dispatch(
      std::tuple<const uint16_t &, const container_kind &, void *const &> it,
      Fn &&fn) {
    switch (std::get<1>(it)) {
    case container_kind::ARRAY: {
      auto *ptr = static_cast<container_array *>(std::get<2>(it));
      return std::forward<Fn>(fn)(*ptr);
    }
    case container_kind::BITMAP: {
      auto *ptr = static_cast<container_bitmap *>(std::get<2>(it));
      return std::forward<Fn>(fn)(*ptr);
    }
    }

    throw std::logic_error("unreachable");
  }

private:
  // TODO: Replace the void* with a `union { container_array, container_bitmap
  // }`, maybe?
  map_of_arrays<uint16_t, container_kind, void *> containers_;

  std::iterator_traits<iterator>::value_type x;
};

// These intersection implementations don't take advantage of the bitset
// implementation at all and will obviously be quite slow.
template <typename OutputIt>
OutputIt intersection(const bitset &left, const bitset &right,
                      OutputIt output) {
  return std::set_intersection(left.begin(), left.end(), right.begin(),
                               right.end(), output);
}

inline bitset intersection(const bitset &left, const bitset &right) {
  bitset result;
  intersection(left, right, default_inserter(result));
  return result;
}

inline uint64_t intersection_size(const bitset &left, const bitset &right) {
  auto result =
      intersection(left, right, counting_output_iterator<uint32_t>(0));
  return result.offset();
}

} // namespace roaring
} // namespace mob

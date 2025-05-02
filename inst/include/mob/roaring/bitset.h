#pragma once

#include <mob/ds/soa.h>
#include <mob/iterator.h>
#include <mob/roaring/container_array.h>
#include <mob/roaring/container_bitmap.h>

#include <algorithm>
#include <cinttypes>
#include <stdexcept>
#include <variant>
#include <vector>

namespace mob {
namespace roaring {

struct bitset {
  using value_type = uint32_t;
  using container_storage = ds::soa_map<
      std::vector<uint16_t>,
      ds::soa_variant<std::vector, container_array<>, container_bitmap<>>>;

  bitset() = default;

  template <typename InputIt>
  bitset(InputIt first, InputIt last) {
    insert(first, last);
  }

  bool contains(uint32_t index) const {
    uint16_t high = index >> 16;
    uint16_t low = index & 0xffff;

    auto it = containers_.find(high);
    if (it == containers_.end()) {
      return false;
    } else {
      return it->second.visit<bool>(
          [low](const auto &c) -> bool { return c.contains(low); });
    }
  }

  void insert(uint32_t index) {
    uint16_t high = index >> 16;
    uint16_t low = index & 0xffff;

    auto [it, inserted] = containers_.try_emplace(
        high, std::in_place_type_t<container_array<>>{}, low);

    if (!inserted) {
      if (auto *array = it->second.get_if<container_array<>>()) {
        auto position = array->lower_bound(low);
        if (position == array->end() || *position != low) {
          if (array->size() < 4096) {
            array->insert(position, low);
          } else {
            // We can't do an actual in-place construction of the bitmap, since
            // that would invalidate the array. We construct on the stack and
            // then move to the containers vector.
            container_bitmap<> bitmap(*array);
            bitmap.insert(low);
            it->second.emplace<container_bitmap<>>(std::move(bitmap));
          }
        }
      } else if (auto *bitmap = it->second.get_if<container_bitmap<>>()) {
        bitmap->insert(low);
      } else {
        throw std::logic_error("unreachable");
      }
    }
  }

  template <typename InputIt>
  void insert(InputIt first, InputIt last) {
    for (auto it = first; it != last; it++) {
      insert(*it);
    }
  }

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = uint32_t;
    using reference = value_type;
    using pointer = const value_type *;
    using difference_type = ptrdiff_t;

    container_storage::const_iterator toplevel;

    // std::monostate is a placeholder for `toplevel->container.begin()`, but
    // it works even in cases where toplevel is the `end()` of the
    // map_of_arrays.
    //
    // We can only call `toplevel->container.begin()` in cases where we know
    // the# iterator is valid.
    std::variant<std::monostate, container_array<>::iterator,
                 container_bitmap<>::iterator>
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
      uint16_t high = toplevel->first;
      uint16_t low = toplevel->second.visit<uint16_t>(
          [this](const auto &container) { return *downcast(container); });

      return (static_cast<uint32_t>(high) << 16) | low;
    }

    iterator &operator++() {
      toplevel->second.visit([this](const auto &container) {
        auto &it = downcast(container);
        if (++it == container.end()) {
          // Containers are never empty, so incrementing by just one step is
          // always enough to reach the next element.
          //
          // Dereferencing toplevel isn't safe though as we may have reached the
          // end() of the containers. We use the monostate as a placeholder, and
          // delay dereferencing toplevel until a point where it is safe to do
          // so.
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

    bool operator==(iterator &other) const {
      if (toplevel != other.toplevel) {
        return false;
      }
      if (std::holds_alternative<std::monostate>(nested) ==
          std::holds_alternative<std::monostate>(other.nested)) {
        return true;
      } else {
        return toplevel->second.visit<bool>([this, &other](auto &container) {
          auto it = downcast(container);
          auto other_it = other.downcast(container);
          return it == other_it;
        });
      }
    }

    bool operator!=(iterator &other) const {
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

  const container_storage &containers() const {
    return containers_;
  }

  uint64_t size() const {
    uint64_t sum = 0;
    // TODO: use parallel reduction
    for (auto it : containers_) {
      it.second.visit([&](const auto &c) { sum += c.size(); });
    }
    return sum;
  }

private:
  container_storage containers_;
};

} // namespace roaring
} // namespace mob

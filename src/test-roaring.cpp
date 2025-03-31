#include "roaring.h"

#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <optional>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

template <typename Container, typename T>
static void check_iterable(Container &c, const std::vector<T> &expected) {
  size_t index = 0;
  for (auto v : c) {
    RC_LOG() << index << std::endl;
    RC_ASSERT(index < expected.size());
    RC_ASSERT(v == expected[index]);
    index++;
  }

  RC_ASSERT(index == expected.size());
}

rc::Gen<size_t> sizeGenerator(std::optional<size_t> maxSize) {
  if (maxSize.has_value()) {
    return rc::gen::inRange<size_t>(0, maxSize.value() + 1);
  } else {
    return rc::gen::withSize(
        [](int size) { return rc::gen::inRange<size_t>(0, size); });
  }
}

template <typename T>
rc::Gen<std::vector<T>> uniqueVectorGenerator() {
  return rc::gen::unique<std::vector<T>>(
      rc::gen::noShrink(rc::gen::arbitrary<T>()));
}

template <typename T>
rc::Gen<std::vector<T>> uniqueVectorGenerator(size_t size) {
  return rc::gen::unique<std::vector<T>>(
      size, rc::gen::noShrink(rc::gen::arbitrary<T>()));
}

namespace rc {

template <>
struct Arbitrary<mob::roaring::bitset> {
  static rc::Gen<mob::roaring::bitset> arbitrary() {
    return rc::gen::scale(64, rc::gen::unique<mob::roaring::bitset>(
                                  rc::gen::arbitrary<uint32_t>()));
  }
};

} // namespace rc

template <typename Container>
void container_properties(std::optional<size_t> maxSize) {
  rc::prop("An empty container contains no value", [=] {
    Container container;
    RC_ASSERT(container.size() == 0U);
    RC_ASSERT(container.find(*rc::gen::arbitrary<uint16_t>()) == false);
  });

  rc::prop("Can insert and find a value", [=](uint16_t index) {
    uint16_t x = *rc::gen::arbitrary<uint16_t>();
    Container container;
    container.insert(x);
    RC_ASSERT(container.find(x) == true);
    RC_ASSERT(container.size() == 1U);
  });

  rc::prop("Adding a value multiple times is idempotent", [=](uint16_t index) {
    uint16_t x = *rc::gen::arbitrary<uint16_t>();
    Container container;
    container.insert(x);
    container.insert(x);
    container.insert(x);
    RC_ASSERT(container.find(x) == true);
    RC_ASSERT(container.size() == 1U);
  });

  rc::prop("If one value is added, finding a different one returns false", [=] {
    uint16_t x = *rc::gen::arbitrary<uint16_t>();
    uint16_t y = *rc::gen::distinctFrom(x);
    Container container;
    container.insert(x);
    return container.find(y) == false;
  });

  rc::prop("Can iterate over container", [=] {
    auto size = *sizeGenerator(maxSize);
    auto values = *uniqueVectorGenerator<uint16_t>(size);

    Container container;
    for (auto v : values) {
      container.insert(v);
    }

    std::sort(values.begin(), values.end());
    check_iterable(container, values);
  });

  rc::prop("Values which were inserted can be found", [=] {
    auto size = *sizeGenerator(maxSize);
    auto values = *uniqueVectorGenerator<uint16_t>(size);

    Container container;
    for (auto v : values) {
      container.insert(v);
    }
    RC_ASSERT(container.size() == values.size());

    auto x = *rc::gen::elementOf(values);
    RC_ASSERT(container.find(x) == true);
  });

  rc::prop("Values which weren't inserted are not found", [=] {
    auto size = *sizeGenerator(maxSize);
    auto values = *uniqueVectorGenerator<uint16_t>(size + 1);

    Container container;
    for (auto it = values.begin(); it != values.end() - 1; it++) {
      container.insert(*it);
    }
    RC_ASSERT(container.size() == values.size() - 1);

    RC_ASSERT(container.find(values.back()) == false);
  });
}

TEST_CASE("roaring::container_array") {
  container_properties<mob::roaring::container_array>(4096);

  rc::prop("An array container can contain at most 4096 elements", []() {
    auto values = *uniqueVectorGenerator<uint16_t>(4097);

    mob::roaring::container_array container;
    for (auto it = values.begin(); it != values.end() - 1; it++) {
      RC_ASSERT(container.insert(*it) == true);
    }

    RC_ASSERT(container.size() == 4096U);
    RC_ASSERT(container.insert(values.back()) == false);

    for (auto it = values.begin(); it != values.end() - 1; it++) {
      RC_ASSERT(container.find(*it) == true);
    }
    RC_ASSERT(container.find(values.back()) == false);
  });
}

TEST_CASE("roaring::container_bitmap") {
  container_properties<mob::roaring::container_bitmap>(std::nullopt);
}

TEST_CASE("roaring::bitset") {
  rc::prop("Can iterate over bitset", [] {
    auto values = *rc::gen::scale(64, uniqueVectorGenerator<uint32_t>());
    mob::roaring::bitset bitset(values.begin(), values.end());

    std::sort(values.begin(), values.end());
    check_iterable(bitset, values);
  });

  rc::prop("Can have many values in a single chunk", [] {
    size_t count = *sizeGenerator(8192);

    uint16_t high = *rc::gen::arbitrary<uint16_t>();
    auto lows = *uniqueVectorGenerator<uint16_t>(count);

    std::vector<uint32_t> values;
    for (uint16_t low : lows) {
      values.push_back(high << 16 | low);
    }

    mob::roaring::bitset bitset(values.begin(), values.end());

    std::sort(values.begin(), values.end());
    check_iterable(bitset, values);
  });

  rc::prop("intersection() elements are in both inputs", [] {
    auto left = *rc::gen::arbitrary<mob::roaring::bitset>();
    auto right = *rc::gen::arbitrary<mob::roaring::bitset>();
    auto result = mob::roaring::intersection(left, right);

    for (auto v : result) {
      RC_ASSERT(left.find(v));
      RC_ASSERT(right.find(v));
    }
    for (auto v : left) {
      RC_ASSERT(right.find(v) == result.find(v));
    }
    for (auto v : right) {
      RC_ASSERT(left.find(v) == result.find(v));
    }
  });

  rc::prop("size() matches iteration", [] {
    auto bitmap = *rc::gen::arbitrary<mob::roaring::bitset>();
    uint64_t size = 0;
    for ([[maybe_unused]] auto v : bitmap) {
      size += 1;
    }
    RC_ASSERT(size == bitmap.size());
  });

  rc::prop("intersection_size() matches intersection()", [] {
    auto left = *rc::gen::arbitrary<mob::roaring::bitset>();
    auto right = *rc::gen::arbitrary<mob::roaring::bitset>();

    auto intersection = mob::roaring::intersection(left, right);
    uint32_t size = mob::roaring::intersection_size(left, right);

    RC_ASSERT(intersection.size() == size);
  });
}

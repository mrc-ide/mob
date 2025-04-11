#include <mob/roaring/bitset.h>

#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <optional>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

template <typename Container>
static void
check_iterable(Container &c,
               const std::vector<typename Container::value_type> &expected) {
  size_t index = 0;
  for (auto v : c) {
    RC_LOG() << index << std::endl;
    RC_ASSERT(index < expected.size());
    RC_ASSERT(v == expected[index]);
    index++;
  }

  RC_ASSERT(index == expected.size());
}

rc::Gen<size_t> sizeGenerator(std::optional<size_t> maxSize = std::nullopt) {
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
    auto values = rc::gen::scale(1024, uniqueVectorGenerator<uint32_t>());
    return rc::gen::map(values, [](auto values) {
      std::sort(values.begin(), values.end());
      return mob::roaring::bitset(values.begin(), values.end());
    });
  }
};

} // namespace rc

template <typename Container>
void container_properties(size_t maxSize) {
  using value_type = typename Container::value_type;

  rc::prop("An empty container contains no value", [=] {
    Container container;
    RC_ASSERT(container.size() == 0U);
    RC_ASSERT(container.contains(*rc::gen::arbitrary<value_type>()) == false);
  });

  rc::prop("Can insert and find a value", [=](value_type index) {
    value_type x = *rc::gen::arbitrary<value_type>();
    Container container;
    container.insert(x);
    RC_ASSERT(container.contains(x) == true);
    RC_ASSERT(container.size() == 1U);
  });

  rc::prop("Adding a value multiple times is idempotent",
           [=](value_type index) {
             value_type x = *rc::gen::arbitrary<value_type>();
             Container container;
             container.insert(x);
             container.insert(x);
             container.insert(x);
             RC_ASSERT(container.contains(x) == true);
             RC_ASSERT(container.size() == 1U);
           });

  rc::prop("If one value is added, finding a different one returns false", [=] {
    value_type x = *rc::gen::arbitrary<value_type>();
    value_type y = *rc::gen::distinctFrom(x);
    Container container;
    container.insert(x);
    return container.contains(y) == false;
  });

  rc::prop("Can iterate over container", [=] {
    auto size = *sizeGenerator(maxSize);
    auto values = *uniqueVectorGenerator<value_type>(size);

    Container container;
    for (auto v : values) {
      container.insert(v);
    }

    std::sort(values.begin(), values.end());
    check_iterable(container, values);
  });

  rc::prop("Values which were inserted can be found", [=] {
    auto size = *sizeGenerator(maxSize);
    auto values = *uniqueVectorGenerator<value_type>(size);

    Container container;
    for (auto v : values) {
      container.insert(v);
    }
    RC_ASSERT(container.size() == values.size());

    auto x = *rc::gen::elementOf(values);
    RC_ASSERT(container.contains(x) == true);
  });

  rc::prop("Values which weren't inserted are not found", [=] {
    auto size = *sizeGenerator(maxSize);
    auto values = *uniqueVectorGenerator<value_type>(size + 1);

    Container container;
    for (auto it = values.begin(); it != values.end() - 1; it++) {
      container.insert(*it);
    }
    RC_ASSERT(container.size() == values.size() - 1);

    RC_ASSERT(container.contains(values.back()) == false);
  });
}

TEST_CASE("roaring::container_array") {
  container_properties<mob::roaring::container_array<>>(4096);
}

TEST_CASE("roaring::container_bitmap") {
  container_properties<mob::roaring::container_bitmap<>>(65536);
}

TEST_CASE("roaring::bitset") {
  container_properties<mob::roaring::bitset>(65536);

  rc::prop("Chunk with many values is promoted to a bitmap", [] {
    uint16_t high = *rc::gen::arbitrary<uint16_t>();

    // We need at least 4097 elements to trigger the promotion
    auto lows = *uniqueVectorGenerator<uint16_t>(5000);

    std::vector<uint32_t> values;
    for (uint16_t low : lows) {
      values.push_back(high << 16 | low);
    }

    mob::roaring::bitset bitset;
    const auto &containers = bitset.containers();

    bitset.insert(values.begin(), values.begin() + 4096);

    RC_ASSERT(containers.size() == 1U);
    RC_ASSERT(containers.front().first == high);
    RC_ASSERT(containers.front()
                  .second.holds_alternative<mob::roaring::container_array<>>());

    bitset.insert(*(values.begin() + 4096));

    RC_ASSERT(containers.size() == 1U);
    RC_ASSERT(
        containers.front()
            .second.holds_alternative<mob::roaring::container_bitmap<>>());

    bitset.insert(values.begin() + 4097, values.end());

    RC_ASSERT(containers.size() == 1U);
    RC_ASSERT(
        containers.front()
            .second.holds_alternative<mob::roaring::container_bitmap<>>());

    std::sort(values.begin(), values.end());
    check_iterable(bitset, values);
  });

  rc::prop("intersection() elements are in both inputs", [] {
    // TODO: generate some more interesting inputs
    auto left = *rc::gen::arbitrary<mob::roaring::bitset>();
    auto right = *rc::gen::arbitrary<mob::roaring::bitset>();
    auto result = mob::roaring::intersection(left, right);

    for (auto v : result) {
      RC_ASSERT(left.contains(v));
      RC_ASSERT(right.contains(v));
    }
    for (auto v : left) {
      RC_ASSERT(right.contains(v) == result.contains(v));
    }
    for (auto v : right) {
      RC_ASSERT(left.contains(v) == result.contains(v));
    }
  });

  rc::prop("intersection_size() matches intersection()", [] {
    auto left = *rc::gen::arbitrary<mob::roaring::bitset>();
    auto right = *rc::gen::arbitrary<mob::roaring::bitset>();

    auto intersection = mob::roaring::intersection(left, right);
    uint32_t size = mob::roaring::intersection_size(left, right);

    RC_ASSERT(intersection.size() == size);
  });
}

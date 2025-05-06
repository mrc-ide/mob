#include <mob/roaring/bitset.h>

#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <optional>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>
#include <thrust/copy.h>

void skipOnGitHub() {
  if (getenv("GITHUB_ACTIONS") != nullptr) {
    SKIP("test is skipped on GitHub Action");
  }
}

namespace thrust {
template <typename T>

void showValue(const thrust::host_vector<T> &value, std::ostream &os) {
  rc::showCollection("[", "]", value, os);
}

template <typename T>
void showValue(const thrust::device_vector<T> &value, std::ostream &os) {
  thrust::host_vector<T> copy(value);
  rc::showCollection("[", "]", copy, os);
}

} // namespace thrust

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

TEST_CASE("roaring (CUDA)") {
  skipOnGitHub();

  rc::prop("Can access a container_array from the GPU", [] {
    auto values = *uniqueVectorGenerator<uint16_t>(1024);

    mob::roaring::container_array<mob::system::device> container(values.begin(),
                                                                 values.end());

    thrust::device_vector<uint16_t> result(values.size());
    auto it = thrust::copy_if(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(UINT16_MAX + 1), result.begin(),
        [dc = mob::roaring::container_array_view(container)] __device__(
            uint16_t x) { return dc.contains(x); });
    result.resize(it - result.begin());

    std::sort(values.begin(), values.end());
    RC_ASSERT(result == values);
  });

  rc::prop("Can access a container_bitmap from the GPU", [] {
    auto values = *uniqueVectorGenerator<uint16_t>(1024);

    mob::roaring::container_bitmap<mob::system::device> container(
        values.begin(), values.end());

    thrust::device_vector<uint16_t> result(values.size());
    auto it = thrust::copy_if(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(UINT16_MAX + 1), result.begin(),
        [dc = mob::roaring::container_bitmap_view(container)] __device__(
            uint16_t x) { return dc.contains(x); });
    result.resize(it - result.begin());

    std::sort(values.begin(), values.end());
    RC_ASSERT(result == values);
  });
}

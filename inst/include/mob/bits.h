#pragma once
#include <cstdint>
#include <cuda/std/bit>

namespace mob {
namespace bits {

// Returns the position of the nth set bit, or 64 if out of bounds.
__host__ __device__ constexpr static inline uint8_t select(uint64_t x,
                                                           uint8_t n) {
  if (n >= 64) {
    return 64;
  }

  for (size_t i = 0; i < n; i++) {
    x &= x - 1;
  }

  return cuda::std::countr_zero(x);
}

} // namespace bits
} // namespace mob

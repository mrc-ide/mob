#pragma once
#include <cstddef>
#include <cstdint>

namespace mob {
namespace bits {

// popcount and countr_zero are in C++20
template <typename T>
uint8_t popcount(T x);

template <typename T>
uint8_t countr_zero(T x);

template <>
inline uint8_t popcount<uint64_t>(uint64_t x) {
  return __builtin_popcountll(x);
}

template <>
inline uint8_t countr_zero<uint64_t>(uint64_t x) {
  // ctz(0) is undefined, so we must handle that specially. We could use tzcnt
  // which is well defined on a 0 input, but it only works on newer
  // architectures and with the right compiler flags.
  if (x == 0) {
    return 64;
  } else {
    return __builtin_ctzll(x);
  }
}

// Returns the position of the nth set bit, or 64 if out of bounds.
static inline uint8_t select(uint64_t x, uint8_t n) {
  if (n >= 64) {
    return 64;
  }

  for (size_t i = 0; i < n; i++) {
    x &= x - 1;
  }

  return countr_zero(x);
}

} // namespace bits
} // namespace mob

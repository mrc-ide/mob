#include <mob/compat.h>
#include <mob/ds/view.h>

namespace mob {
namespace ds {

template <typename It, typename Sentinel = It>
struct subrange {
  __host__ __device__ subrange(It begin, Sentinel end)
      : begin_(begin), end_(end) {}

  __host__ __device__ It begin() const {
    return begin_;
  }

  __host__ __device__ Sentinel end() const {
    return end_;
  }

private:
  It begin_;
  Sentinel end_;
};

// This assumes left is much smaller than right
template <typename Pointer>
__host__ __device__ size_t intersection_size(ds::span<Pointer> left,
                                             ds::span<Pointer> right) {
  size_t count = 0;
  auto low = right.begin();
  for (auto v : left) {
    low = mob::compat::lower_bound(low, right.end(), v);
    if (low == right.end()) {
      break;
    } else if (*low == v) {
      count++;
    }
  }
  return count;
}

template <typename LeftIt, typename RightIt>
struct intersection_iterator {
  using reference = cuda::std::iter_reference_t<LeftIt>;
  using value_type = cuda::std::iter_value_t<LeftIt>;
  using difference_type = ptrdiff_t;

  LeftIt left;
  LeftIt left_end;

  RightIt right;
  RightIt right_end;

  __host__ __device__ intersection_iterator(LeftIt left_start, LeftIt left_end,
                                            RightIt right_start,
                                            RightIt right_end)
      : left(left_start), left_end(left_end), right(right_start),
        right_end(right_end) {
    skip();
  }

  __host__ __device__ intersection_iterator &operator++() {
    ++left;
    skip();
    return *this;
  }

  __host__ __device__ intersection_iterator &operator+=(size_t n) {
    for (; n > 0; n--) {
      ++(*this);
    }
    return *this;
  }

  __host__ __device__ bool operator==(cuda::std::default_sentinel_t) {
    return left == left_end;
  }

  __host__ __device__ bool operator!=(cuda::std::default_sentinel_t) {
    return left != left_end;
  }

  __host__ __device__ reference operator*() const {
    return *left;
  }

  __host__ __device__ void skip() {
    for (; left != left_end; left++) {
      right = mob::compat::lower_bound(right, right_end, *left);
      if (right == right_end) {
        left = left_end;
        break;
      } else if (*right == *left) {
        break;
      }
    }
  }
};

template <typename LeftIt, typename RightIt>
using intersection_range = subrange<intersection_iterator<LeftIt, RightIt>,
                                    cuda::std::default_sentinel_t>;

template <typename Pointer>
__host__ __device__ intersection_range<Pointer, Pointer>
lazy_intersection(ds::span<Pointer> left, ds::span<Pointer> right) {
  return subrange(intersection_iterator(left.begin(), left.end(), right.begin(),
                                        right.end()),
                  cuda::std::default_sentinel_t{});
}

} // namespace ds
} // namespace mob

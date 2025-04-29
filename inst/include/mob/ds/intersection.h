#include <mob/compat.h>

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

template <typename LeftRange, typename RightRange>
struct intersection_iterator {
  using reference = compat::range_reference_t<LeftRange>;
  using value_type = compat::range_value_t<LeftRange>;
  using difference_type = ptrdiff_t;

  compat::iterator_t<LeftRange> left;
  compat::sentinel_t<LeftRange> left_end;

  compat::iterator_t<RightRange> right;
  compat::sentinel_t<RightRange> right_end;

  __host__ __device__ intersection_iterator(LeftRange left, RightRange right)
      : left(cuda::std::begin(left)), left_end(cuda::std::end(left)),
        right(cuda::std::begin(right)), right_end(cuda::std::end(right)) {
    skip();
  }

  __host__ __device__ intersection_iterator &operator++() {
    ++left;
    skip();
    return *this;
  }

  __host__ __device__ intersection_iterator operator++(int) {
    auto old = *this;
    ++(*this);
    return old;
  }

  __host__ __device__ bool operator==(cuda::std::default_sentinel_t) const {
    return left == left_end;
  }
  __host__ __device__ bool operator!=(cuda::std::default_sentinel_t) const {
    return left != left_end;
  }
  friend __host__ __device__ bool
  operator==(cuda::std::default_sentinel_t, const intersection_iterator &self) {
    return self.left == self.left_end;
  }
  friend __host__ __device__ bool
  operator!=(cuda::std::default_sentinel_t, const intersection_iterator &self) {
    return self.left != self.left_end;
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

template <typename LeftRange, typename RightRange>
using intersection_range =
    subrange<intersection_iterator<LeftRange, RightRange>,
             cuda::std::default_sentinel_t>;

template <typename LeftRange, typename RightRange>
__host__ __device__ intersection_range<LeftRange, RightRange>
lazy_intersection(LeftRange left, RightRange right) {
  return subrange(intersection_iterator(left, right),
                  cuda::std::default_sentinel_t{});
}

} // namespace ds
} // namespace mob

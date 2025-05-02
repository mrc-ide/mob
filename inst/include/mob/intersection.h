#pragma once
#include <mob/compat.h>
#include <mob/ds/span.h>

namespace mob {

template <std::ranges::input_range LeftRange,
          std::ranges::input_range RightRange>
  requires std::ranges::enable_view<LeftRange> &&
           std::ranges::enable_view<RightRange>
struct intersection_view {
  using sentinel = cuda::std::default_sentinel_t;

  struct iterator {
    using reference = compat::range_reference_t<const LeftRange>;
    using value_type = compat::range_value_t<const LeftRange>;
    using difference_type = ptrdiff_t;

    __host__ __device__ iterator(const LeftRange &left, const RightRange &right)
        : left(cuda::std::begin(left)), left_end(cuda::std::end(left)),
          right(cuda::std::begin(right)), right_end(cuda::std::end(right)) {
      skip();
    }

    __host__ __device__ iterator &operator++() {
      ++left;
      skip();
      return *this;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __host__ __device__ bool operator==(sentinel) const {
      return left == left_end;
    }

    __host__ __device__ reference operator*() const {
      return *left;
    }

  private:
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

    compat::iterator_t<const LeftRange> left;
    compat::sentinel_t<const LeftRange> left_end;

    compat::iterator_t<const RightRange> right;
    compat::sentinel_t<const RightRange> right_end;
  };

  static_assert(std::input_iterator<iterator>);
  static_assert(std::sentinel_for<sentinel, iterator>);

  __host__ __device__ intersection_view(LeftRange left, RightRange right)
      : left(std::move(left)), right(std::move(right)) {}

  __host__ __device__ iterator begin() const {
    return iterator(left, right);
  }

  __host__ __device__ sentinel end() const {
    return {};
  }

private:
  LeftRange left;
  RightRange right;
};

template <std::ranges::input_range LeftRange,
          std::ranges::input_range RightRange>
intersection_view(LeftRange &&, RightRange &&)
    -> intersection_view<compat::all_t<LeftRange>, compat::all_t<RightRange>>;

template <std::ranges::input_range LeftRange,
          std::ranges::input_range RightRange>
__host__ __device__ auto intersection(LeftRange &&left, RightRange &&right) {
  return intersection_view(std::forward<LeftRange>(left),
                           std::forward<RightRange>(right));
}

} // namespace mob

namespace std::ranges {

template <typename L, typename R>
constexpr bool enable_view<mob::intersection_view<L, R>> = true;

}

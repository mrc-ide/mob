#pragma once
#include <mob/compat/algorithm.h>
#include <mob/compat/views.h>
#include <mob/ds/span.h>

namespace mob {

template <cuda::std::ranges::input_range LeftRange,
          cuda::std::ranges::random_access_range RightRange>
  requires cuda::std::ranges::enable_view<LeftRange> &&
           cuda::std::ranges::enable_view<RightRange>
struct intersection_view : cuda::std::ranges::view_interface<
                               intersection_view<LeftRange, RightRange>> {
  using sentinel = cuda::std::default_sentinel_t;

  struct iterator {
    using reference = cuda::std::ranges::range_reference_t<const LeftRange>;
    using value_type = cuda::std::ranges::range_value_t<const LeftRange>;
    using difference_type = ptrdiff_t;

    __host__ __device__ iterator(const LeftRange &left, const RightRange &right)
        : left(cuda::std::begin(left)), left_end(cuda::std::end(left)),
          right(cuda::std::begin(right)), right_end(cuda::std::end(right)) {
      skip();
    }

    __nv_exec_check_disable__ __host__ __device__ iterator &operator++() {
      ++left;
      skip();
      return *this;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __nv_exec_check_disable__ __host__ __device__ bool
    operator==(sentinel) const {
      return left == left_end || right == right_end;
    }

    __nv_exec_check_disable__ __host__ __device__ reference operator*() const {
      return *left;
    }

  private:
    __nv_exec_check_disable__ __host__ __device__ void skip() {
      for (; left != left_end; left++) {
        right = mob::compat::lower_bound(right, right_end, *left);
        if (right == right_end) {
          break;
        } else if (*right == *left) {
          break;
        }
      }
    }

    cuda::std::ranges::iterator_t<const LeftRange> left;
    cuda::std::ranges::sentinel_t<const LeftRange> left_end;

    cuda::std::ranges::iterator_t<const RightRange> right;
    cuda::std::ranges::sentinel_t<const RightRange> right_end;
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

template <cuda::std::ranges::input_range LeftRange,
          cuda::std::ranges::input_range RightRange>
intersection_view(LeftRange &&, RightRange &&)
    -> intersection_view<compat::all_t<LeftRange>, compat::all_t<RightRange>>;

template <cuda::std::ranges::input_range LeftRange,
          cuda::std::ranges::input_range RightRange>
__host__ __device__ auto intersection(LeftRange &&left, RightRange &&right) {
  return intersection_view(std::forward<LeftRange>(left),
                           std::forward<RightRange>(right));
}

} // namespace mob

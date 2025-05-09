#pragma once
#include <mob/compat/views.h>
#include <mob/random.h>

#include <cuda/std/cmath>
#include <dust/random/random.hpp>

namespace mob {

template <typename real_type>
struct fast_bernoulli {
  __host__ __device__ fast_bernoulli(real_type probability)
      : probability(probability) {
    // The maths don't work for probability = 0 or probability = 1
    if (0 < probability && probability < 1) {
      real_type probability_log = log(1 - probability);
      // Probabilities smaller than 2^-53 could end up with a `probability_log`
      // rounded to zero, which we can't inverse. Treat these probabilties the
      // same as 0.
      if (probability_log == 0.0) {
        probability = 0.;
      } else {
        inverse_log = 1 / log(1 - probability);
      }
    }
  }

  template <typename int_type = size_t, typename rng_state_type>
  __host__ __device__ int_type next(rng_state_type &rng_state) {
    constexpr static int_type max = cuda::std::numeric_limits<int_type>::max();

    if (probability == 1) {
      return 0;
    } else if (probability == 0) {
      return max;
    }

    // For very small probability, the skip count can end up being very large
    // and exceeding SIZE_MAX. Returning the double directly would be UB.
    real_type x = dust::random::random_real<real_type>(rng_state);
    real_type skip = cuda::std::floor(cuda::std::log(x) * inverse_log);
    if (skip < real_type(max)) {
      return skip;
    } else {
      return max;
    }
  }

private:
  real_type inverse_log;
  real_type probability;
};

template <cuda::std::ranges::input_range Range, typename real_type,
          random_state rng_state_type>
  requires(cuda::std::ranges::enable_view<Range>)
struct bernoulli_view : cuda::std::ranges::view_interface<
                            bernoulli_view<Range, real_type, rng_state_type>> {
  using sentinel = cuda::std::default_sentinel_t;

  struct iterator {
    using reference = cuda::std::ranges::range_reference_t<const Range>;
    using value_type = cuda::std::ranges::range_value_t<const Range>;
    using difference_type = ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    // The bernoulli iterator does not allow multiple passes: each pass would
    // use a distinct random number generator state
    iterator(const iterator &other) = delete;
    iterator &operator=(const iterator &other) = delete;

    iterator(iterator &&other) = default;
    iterator &operator=(iterator &&other) = default;

    __host__ __device__ iterator(const Range &range, real_type p,
                                 rng_state_type *rng_state)
        : it(cuda::std::begin(range)), end(cuda::std::end(range)),
          rng_state(rng_state), bernoulli(p) {
      skip();
    }

    __host__ __device__ iterator &operator++() {
      ++it;
      skip();
      return *this;
    }

    __host__ __device__ void operator++(int) {
      ++(*this);
    }

    __host__ __device__ bool operator==(sentinel) const {
      return it == end;
    }

    __host__ __device__ reference operator*() const {
      return *it;
    }

  private:
    __host__ __device__ void skip() {
      auto n =
          bernoulli.template next<cuda::std::ranges::range_difference_t<Range>>(
              *rng_state);
      cuda::std::ranges::advance(it, n, end);
    }

    cuda::std::ranges::iterator_t<const Range> it;
    cuda::std::ranges::sentinel_t<const Range> end;
    rng_state_type *rng_state;
    fast_bernoulli<real_type> bernoulli;
  };

  static_assert(std::input_iterator<iterator>);
  static_assert(std::sentinel_for<sentinel, iterator>);

  __host__ __device__ bernoulli_view(Range range, real_type probability,
                                     rng_state_type &rng)
      : range(std::move(range)), probability(probability), rng(&rng) {}

  __host__ __device__ iterator begin() const {
    return iterator(range, probability, rng);
  }

  __host__ __device__ sentinel end() const {
    return {};
  }

private:
  Range range;
  real_type probability;
  rng_state_type *rng;
};

template <cuda::std::ranges::input_range Range, typename real_type,
          random_state rng_state_type>
bernoulli_view(Range &&, real_type, rng_state_type &)
    -> bernoulli_view<compat::all_t<Range>, real_type, rng_state_type>;

template <typename real_type, random_state rng_state_type,
          cuda::std::ranges::input_range Range>
__host__ __device__ auto bernoulli(Range &&range, real_type probability,
                                   rng_state_type &rng) {
  return bernoulli_view(std::forward<Range>(range), probability, rng);
}

} // namespace mob

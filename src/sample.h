#pragma once

#include "iterator.h"
#include <dust/random/binomial.hpp>
#include <dust/random/gamma.hpp>
#include <thrust/iterator/counting_iterator.h>

namespace mob {

template <typename InputIt, typename OutputIt>
__host__ __device__ void sampler_check(InputIt input_start, InputIt input_end,
                                       OutputIt output_start,
                                       OutputIt output_end) {
  size_t n = cuda::std::distance(input_start, input_end);
  size_t k = cuda::std::distance(output_start, output_end);
  if (k > n) {
    dust::utils::fatal_error("Invalid sampler input");
  }
}

/**
 * This is "Algorithm S" from "Faster Methods for Random Sampling" by JS.
 * Vitter. https://dl.acm.org/doi/pdf/10.1145/358105.893
 *
 * It is obviously very slow, since it iterates over the entire input range one
 * entry at a time. Mostly here for exposition purposes.
 */
template <typename rng_state_type, typename InputIt, typename OutputIt>
__host__ __device__ void
selection_sampler(rng_state_type &rng_state, InputIt input_start,
                  InputIt input_end, OutputIt output_start,
                  OutputIt output_end) {
  sampler_check(input_start, input_end, output_start, output_end);
  for (; output_start != output_end; input_start++) {
    size_t n = cuda::std::distance(input_start, input_end);
    size_t k = cuda::std::distance(output_start, output_end);
    double u = dust::random::random_real<double>(rng_state);
    if (n * u < k) {
      *(output_start++) = *input_start;
    }
  }
}

template <typename real_type, typename rng_state_type>
__host__ real_type beta(rng_state_type &rng_state, real_type alpha,
                        real_type beta) {
  // The Ting paper uses `Beta(α = 1, β) ~ 1 - exp(U(0,1), 1/β)`.
  //
  // I can't find any secondary source for this, so for now I am going with
  // whatever Wikipedia says. Possibly the above formula only works for `α=1`
  // (which the sampling algorithm always uses).
  //
  // TODO: dust doesn't support Gamma on the GPU, so using the above expression
  // would be very nice.

  real_type x = dust::random::gamma(rng_state, alpha, 1.0);
  real_type y = dust::random::gamma(rng_state, beta, 1.0);
  return x / (x + y);
}

template <typename real_type, typename rng_state_type>
__host__ real_type betabinomial(rng_state_type &rng_state, real_type a,
                                real_type b, size_t n) {
  real_type p = beta(rng_state, a, b);
  return dust::random::binomial<real_type>(rng_state, n, p);
}

/**
 * "Simple, Optimal Algorithms for Random Sampling Without Replacement" by D.
 * Ting https://arxiv.org/pdf/2104.05091
 *
 * This performs a sample without replacement, filling up the output range from
 * the input one. The input must be longer than the output.
 *
 * The implementation works by generating jump lengths between successive
 * elements. The number of elements to skip each time is given by
 * `BetaBinomal(1, k, n-k)`, where n and k are the remaining inputs and
 * remaining outputs, respectively.
 *
 * Sampling is always performed in-order, meaning the output elements are always
 * ordered in the same way as they were in the input.
 */
template <typename rng_state_type, typename InputIt, typename OutputIt>
__host__ void betabinomial_sampler(rng_state_type &rng_state,
                                   InputIt input_start, InputIt input_end,
                                   OutputIt output_start, OutputIt output_end) {
  sampler_check(input_start, input_end, output_start, output_end);
  for (; output_start != output_end; output_start++) {
    size_t n = cuda::std::distance(input_start, input_end);
    size_t k = cuda::std::distance(output_start, output_end);

    size_t skip = betabinomial<double>(rng_state, 1, k, n - k);
    input_start += skip;

    *output_start = *(input_start++);
  }
}

template <typename real_type>
struct fast_bernouilli {
  __host__ __device__ fast_bernouilli(real_type probability) {
    // TODO: handle case where denominator == 0.
    // This can happen with very small (or zero) probabilities, where the result
    // gets rounded to zero.
    // Also handle p = 1 case, which currently tries to evaluate log(0)
    //
    // Do we even care? Can we rely on 1/0 = Inf and log(0) = -Inf?
    inverse_log = 1 / log(1 - probability);
  }

  template <typename rng_state_type>
  __host__ __device__ size_t next(rng_state_type &rng_state) {
    // For very small probability, the skip count can end up being very large
    // and exceeding SIZE_MAX. Returning the double directly would be UB.
    real_type x = dust::random::random_real<real_type>(rng_state);
    // TODO: use real_type
    double skip = floor(log(x) * inverse_log);
    if (skip < double(SIZE_MAX)) {
      return skip;
    } else {
      return SIZE_MAX;
    }
  }

private:
  real_type inverse_log;
};

template <typename real_type, typename rng_state_type, typename InputIt,
          typename OutputIt>
__host__ __device__ OutputIt bernouilli_sampler(rng_state_type &rng_state,
                                                InputIt input_start,
                                                InputIt input_end,
                                                OutputIt output, real_type p) {
  if (p < 0 || p > 1) {
    dust::utils::fatal_error("Invalid sampler input");
  }

  fast_bernouilli bernoulli(p);
  while (true) {
    size_t skip = bernoulli.next(rng_state);
    if (skip >= cuda::std::distance(input_start, input_end)) {
      break;
    }
    input_start += skip;
    *(output++) = *(input_start++);
  }
  return output;
}

template <typename int_type, typename real_type, typename rng_state_type>
__host__ __device__ int_type bernouilli_sampler_count(rng_state_type &rng_state,
                                                      int_type n, real_type p) {
  auto it = mob::bernouilli_sampler(rng_state,
                                    thrust::make_counting_iterator<int_type>(0),
                                    thrust::make_counting_iterator<int_type>(n),
                                    counting_output_iterator<int_type>(), p);

  return it.offset();
}

} // namespace mob

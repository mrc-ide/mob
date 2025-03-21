#pragma once

#include <dust/random/binomial.hpp>
#include <dust/random/gamma.hpp>

namespace mob {

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
  for (; output_start != output_end; output_start++) {
    size_t n = cuda::std::distance(input_start, input_end);
    size_t k = cuda::std::distance(output_start, output_end);

    size_t skip = betabinomial<double>(rng_state, 1, k, n - k);
    input_start += skip;

    *output_start = *(input_start++);
  }
}

} // namespace mob

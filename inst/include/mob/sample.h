#pragma once

#include "iterator.h"

#include <cuda/std/cmath>
#include <dust/random/binomial.hpp>
#include <dust/random/gamma.hpp>

namespace mob {

template <typename InputIt, typename OutputIt>
__host__ __device__ void sampler_check(InputIt input_start, InputIt input_end,
                                       OutputIt output_start,
                                       OutputIt output_end) {
  size_t n = mob::compat::distance(input_start, input_end);
  size_t k = mob::compat::distance(output_start, output_end);
  if (k > n) {
#ifdef __CUDA_ARCH__
    dust::utils::fatal_error("Invalid sampler input");
#else
    auto s = std::format("Invalid sampler input: n={} k={}", n, k);
    dust::utils::fatal_error(s.c_str());
#endif
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
    size_t n = compat::distance(input_start, input_end);
    size_t k = compat::distance(output_start, output_end);
    double u = dust::random::random_real<double>(rng_state);
    if (n * u < k) {
      *(output_start++) = *input_start;
    }
  }
}

// This is a special case of the Beta distribution for α = 1.
//
// It uses the Inverse Transform Method. The Ting paper uses this expression
// without much commentary. It is supported by `Critical Analysis of Beta
// Random Variable Generation Methods` by Luengo at al, which gives the
// expression for `β = 1` instead.
//
// The CDF of the Beta distribution is Ix(a,b). For α = 1, Wikipedia gives the
// following identity:
//
// Ix(1,b) = 1 - pow(1 - x, β)
//
// This can be inversed as:
//
// x = 1 - pow(1 - Ix(1, β), 1 / β)
//
// Given a uniformly distributed value U:
//
// Beta(α = 1, β) ~ 1 - pow(1 - U, 1 / β)
//
// or equivalently:
//
// Beta(α = 1, β) ~ 1 - pow(U, 1 / β)
//
template <typename real_type, typename rng_state_type>
__host__ __device__ real_type beta_alpha1(rng_state_type &rng_state,
                                          real_type beta) {
  real_type u = dust::random::random_real<real_type>(rng_state);
  return 1. - dust::math::pow(u, 1. / beta);
}

template <typename real_type, typename rng_state_type>
__host__ __device__ real_type betabinomial_alpha1(rng_state_type &rng_state,
                                                  real_type b, size_t n) {
  real_type p = beta_alpha1(rng_state, b);
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
__host__ __device__ void
betabinomial_sampler(rng_state_type &rng_state, InputIt input_start,
                     InputIt input_end, OutputIt output_start,
                     OutputIt output_end) {
  sampler_check(input_start, input_end, output_start, output_end);

  auto n = compat::distance(input_start, input_end);
  auto k = compat::distance(output_start, output_end);
  for (; output_start != output_end; output_start++) {
    size_t skip = betabinomial_alpha1<double>(rng_state, k, n - k);
    input_start += skip;
    n -= skip;

    *output_start = *(input_start++);
    n -= 1;
    k -= 1;
  }
}

} // namespace mob

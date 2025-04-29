#include "interface.h"
#include <mob/infection.h>
#include <mob/iterator.h>
#include <mob/parallel_random.h>
#include <mob/roaring/bitset.h>
#include <mob/sample.h>

#include <Rcpp.h>
#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>

Rcpp::NumericVector betabinomial_sampler_wrapper(Rcpp::NumericVector data,
                                                 size_t k, int seed) {
  auto rng = dust::random::seed<dust::random::xoroshiro128plus>(seed);

  Rcpp::NumericVector result(k);
  mob::betabinomial_sampler(rng, data.cbegin(), data.cend(), result.begin(),
                            result.end());
  return result;
}

Rcpp::NumericVector selection_sampler_wrapper(Rcpp::NumericVector data,
                                              size_t k, int seed) {
  auto rng = dust::random::seed<dust::random::xoroshiro128plus>(seed);

  Rcpp::NumericVector result(k);
  mob::selection_sampler(rng, data.cbegin(), data.cend(), result.begin(),
                         result.end());
  return result;
}

size_t bernouilli_sampler_count_wrapper(size_t n, double p, int seed) {
  auto rng = dust::random::seed<dust::random::xoroshiro128plus>(seed);
  return mob::bernouilli_sampler_count(rng, n, p);
}

std::vector<double> bernouilli_sampler_wrapper(Rcpp::NumericVector data,
                                               double p, int seed) {
  auto rng = dust::random::seed<dust::random::xoroshiro128plus>(seed);

  // Ideally we'd use a NumericVector type here, but it is unfortunately
  // incompatible with std::back_inserter. NumericVector::value_type is a
  // `double&` when it should really be a `double`.
  // https://en.cppreference.com/w/cpp/named_req/Container
  std::vector<double> result;
  mob::bernouilli_sampler(rng, data.cbegin(), data.cend(),
                          cuda::std::back_insert_iterator(result), p);

  return result;
}

size_t bernouilli_sampler_count_gpu_wrapper(size_t n, double p, int seed) {
  mob::device_random rngs(1, seed);
  thrust::device_vector<size_t> result(1);

  thrust::transform(rngs.begin(), rngs.end(), result.begin(),
                    [n, p] __device__(auto &rng) {
                      return mob::bernouilli_sampler_count(rng, n, p);
                    });

  return result.front();
}

Rcpp::NumericVector bernouilli_sampler_gpu_wrapper(Rcpp::NumericVector data,
                                                   double p, int seed) {
  mob::device_random rngs(1, seed);
  thrust::device_vector<size_t> count(1);

  size_t n = data.size();
  thrust::transform(rngs.begin(), rngs.end(), count.begin(),
                    [n, p] __device__(const auto &rng) {
                      auto rng_copy = rng.get();
                      return mob::bernouilli_sampler_count(rng_copy, n, p);
                    });

  thrust::device_vector<double> input(data.begin(), data.end());
  thrust::device_vector<double> result(count.back());

  auto input_begin = result.begin();
  auto input_end = result.end();
  auto result_begin = result.begin();

  thrust::for_each(
      rngs.begin(), rngs.end(),
      [n, p, input_begin, input_end, result_begin] __device__(auto &rng) {
        mob::bernouilli_sampler(rng, input_begin, input_end, result_begin, p);
      });

  return {result.begin(), result.end()};
}

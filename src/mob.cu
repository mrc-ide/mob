#include "iterator.h"
#include "mob.h"
#include "parallel_random.h"
#include "roaring.h"
#include "sample.h"

#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>

Rcpp::NumericVector parallel_runif(size_t n, double min, double max, int seed) {
  mob::device_random<> rng(n, seed);
  thrust::device_vector<double> dv(n);

  thrust::transform(rng.begin(), rng.end(), dv.begin(),
                    [min, max] __device__(auto &rng) {
                      return dust::random::uniform<double>(rng, min, max);
                    });

  Rcpp::NumericVector result(dv.size());
  thrust::copy(dv.begin(), dv.end(), result.begin());
  return result;
}

Rcpp::NumericVector parallel_rbinom(size_t n, size_t size, double prob,
                                    int seed) {
  mob::device_random<> rng(n, seed);
  thrust::device_vector<double> dv(n);

  thrust::transform(rng.begin(), rng.end(), dv.begin(),
                    [size, prob] __device__(auto &rng) -> double {
                      return dust::random::binomial<double>(rng, size, prob);
                    });

  Rcpp::NumericVector result(dv.size());
  thrust::copy(dv.begin(), dv.end(), result.begin());
  return result;
}

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

size_t bernouilli_sampler_count_wrapper(size_t n, double p, int seed) {
  auto rng = dust::random::seed<dust::random::xoroshiro128plus>(seed);

  return mob::bernouilli_sampler_count<size_t, double>(rng, n, p);
}

#include "interface.h"
#include <mob/infection.h>
#include <mob/iterator.h>
#include <mob/parallel_random.h>
#include <mob/sample.h>

#include <Rcpp.h>
#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>

Rcpp::NumericVector
betabinomial_sampler_wrapper(Rcpp::NumericVector data, size_t k,
                             Rcpp::Nullable<Rcpp::NumericVector> seed) {
  auto rng =
      dust::random::seed<dust::random::xoroshiro128plus>(from_seed(seed));

  Rcpp::NumericVector result(k);
  mob::betabinomial_sampler(rng, data.cbegin(), data.cend(), result.begin(),
                            result.end());
  return result;
}

Rcpp::NumericVector
selection_sampler_wrapper(Rcpp::NumericVector data, size_t k,
                          Rcpp::Nullable<Rcpp::NumericVector> seed) {
  auto rng =
      dust::random::seed<dust::random::xoroshiro128plus>(from_seed(seed));

  Rcpp::NumericVector result(k);
  mob::selection_sampler(rng, data.cbegin(), data.cend(), result.begin(),
                         result.end());
  return result;
}

Rcpp::NumericVector
bernoulli_sampler_wrapper(Rcpp::NumericVector data, double p,
                          Rcpp::Nullable<Rcpp::NumericVector> seed) {
  auto rng =
      dust::random::seed<dust::random::xoroshiro128plus>(from_seed(seed));

  Rcpp::NumericVector result;
  for (auto x : mob::bernoulli(data, p, rng)) {
    result.push_back(x);
  }
  return result;
}

Rcpp::NumericVector
bernoulli_sampler_gpu_wrapper(Rcpp::NumericVector data, double p,
                              Rcpp::Nullable<Rcpp::NumericVector> seed) {
  RSystem::random rngs(1, from_seed(seed));

  thrust::device_vector<double> input(data.begin(), data.end());
  mob::ds::span input_view(input);

  thrust::device_vector<size_t> count(1);
  thrust::transform(rngs.begin(), rngs.end(), count.begin(),
                    [p, input_view] __device__(const auto &rng) {
                      auto rng_copy = rng.get();
                      return mob::compat::distance(
                          mob::bernoulli(input_view, p, rng_copy));
                    });

  thrust::device_vector<double> result(count.back());
  auto result_begin = result.begin();
  thrust::for_each(rngs.begin(), rngs.end(),
                   [p, input_view, result_begin] __device__(auto &rng) {
                     mob::compat::copy(mob::bernoulli(input_view, p, rng),
                                       result_begin);
                   });

  return {result.begin(), result.end()};
}

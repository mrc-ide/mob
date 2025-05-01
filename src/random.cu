#include "interface.h"

#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>

Rcpp::XPtr<RSystem::random>
device_random_create(size_t size, Rcpp::Nullable<Rcpp::NumericVector> seed) {
  return Rcpp::XPtr<RSystem::random>(
      new RSystem::random(size, from_seed(seed)));
}

Rcpp::NumericVector parallel_runif(Rcpp::XPtr<RSystem::random> rngs, size_t n,
                                   double min, double max) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  RSystem::vector<double> result(n);
  thrust::transform(rngs->begin(), rngs->end(), result.begin(),
                    [min, max] __device__(auto &rng) {
                      return dust::random::uniform<double>(rng, min, max);
                    });

  return {result.begin(), result.end()};
}

Rcpp::NumericVector parallel_rbinom(Rcpp::XPtr<RSystem::random> rngs, size_t n,
                                    size_t size, double prob) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  RSystem::vector<double> result(n);
  thrust::transform(rngs->begin(), rngs->end(), result.begin(),
                    [size, prob] __device__(auto &rng) -> double {
                      return dust::random::binomial<double>(rng, size, prob);
                    });

  return {result.begin(), result.end()};
}

#pragma once

#include "conversion.h"
#include <mob/parallel_random.h>

#include <Rcpp.h>
#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>
#include <thrust/transform.h>

inline uint64_t from_seed(Rcpp::Nullable<Rcpp::NumericVector> seed) {
  if (seed.isNotNull()) {
    return Rcpp::as<uint64_t>(seed);
  } else {
    return std::ceil(std::abs(R::unif_rand()) *
                     std::numeric_limits<uint64_t>::max());
  }
}

template <typename System>
Rcpp::XPtr<mob::parallel_random<System>>
random_create_wrapper(size_t size, Rcpp::Nullable<Rcpp::NumericVector> seed) {
  return Rcpp::XPtr<mob::parallel_random<System>>(
      new mob::parallel_random<System>(size, from_seed(seed)));
}

template <typename System>
Rcpp::NumericVector
random_uniform_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs, size_t n,
                       double min, double max) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  mob::vector<System, double> result(n);
  thrust::transform(
      rngs->begin(), rngs->begin() + n, result.begin(),
      [min, max] __host__ __device__(mob::random_proxy<System> & rng) {
        return dust::random::uniform<double>(rng, min, max);
      });

  return asRcppVector(result);
}

template <typename System>
Rcpp::NumericVector
random_poisson_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs, size_t n,
                       double lambda) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  mob::vector<System, double> result(n);
  thrust::transform(
      rngs->begin(), rngs->begin() + n, result.begin(),
      [lambda] __host__ __device__(mob::random_proxy<System> & rng) {
        return dust::random::poisson<double>(rng, lambda);
      });

  return asRcppVector(result);
}

template <typename System>
void random_uniform_benchmark_wrapper(
    Rcpp::XPtr<mob::parallel_random<System>> rngs, size_t n, double min,
    double max) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  mob::vector<System, double> result(n);
  thrust::transform(
      rngs->begin(), rngs->begin() + n, result.begin(),
      [min, max] __host__ __device__(mob::random_proxy<System> & rng) {
        return dust::random::uniform<double>(rng, min, max);
      });
}

template <typename System>
Rcpp::NumericVector
random_binomial_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs, size_t n,
                        size_t size, double prob) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  mob::vector<System, double> result(n);
  thrust::transform(rngs->begin(), rngs->begin() + n, result.begin(),
                    [size, prob] __host__ __device__(mob::random_proxy<System> &
                                                     rng) -> double {
                      return dust::random::binomial<double>(rng, size, prob);
                    });

  return asRcppVector(result);
}

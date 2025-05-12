#pragma once

#include "conversion.h"

#include <Rcpp.h>
#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>
#include <thrust/transform.h>

inline size_t from_seed(Rcpp::Nullable<Rcpp::NumericVector> seed) {
  if (seed.isNotNull()) {
    return Rcpp::as<size_t>(seed);
  } else {
    return std::ceil(std::abs(R::unif_rand()) *
                     std::numeric_limits<size_t>::max());
  }
}

template <typename System>
Rcpp::XPtr<typename System::random>
random_create_wrapper(size_t size, Rcpp::Nullable<Rcpp::NumericVector> seed) {
  return Rcpp::XPtr<typename System::random>(
      new typename System::random(size, from_seed(seed)));
}

template <typename System>
Rcpp::NumericVector
random_uniform_wrapper(Rcpp::XPtr<typename System::random> rngs, size_t n,
                       double min, double max) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  typename System::vector<double> result(n);
  thrust::transform(
      rngs->begin(), rngs->begin() + n, result.begin(),
      [min, max] __host__ __device__(typename System::random::proxy rng) {
        return dust::random::uniform<double>(rng, min, max);
      });

  return asRcppVector(result);
}

template <typename System>
Rcpp::NumericVector
random_binomial_wrapper(Rcpp::XPtr<typename System::random> rngs, size_t n,
                        size_t size, double prob) {
  if (rngs->size() < n) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), n);
  }

  typename System::vector<double> result(n);
  thrust::transform(rngs->begin(), rngs->begin() + n, result.begin(),
                    [size, prob] __host__ __device__(
                        typename System::random::proxy rng) -> double {
                      return dust::random::binomial<double>(rng, size, prob);
                    });

  return asRcppVector(result);
}

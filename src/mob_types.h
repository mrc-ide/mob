// This file is included by the auto-generated RcppExports.cpp. It is
// effectively the entrypoint to the C++ parts of the package.
//
// It mostly acts as concrete instantiations of the templated functions
// found in the `*_wrapper.h` files. These are currently a bit verbose,
// since we need to define two new functions each time that can be wrapped
// by Rcpp.
//
// Soon Rcpp might have better built-in support for this kind of thing:
// https://github.com/RcppCore/Rcpp/issues/1368

#pragma once

#include "infection_wrapper.h"
#include "random_wrapper.h"
#include "sample_wrapper.h"
#include <mob/system.h>

#ifdef __NVCC__

// [[Rcpp::export]]
inline Rcpp::XPtr<mob::system::device::random>
random_create_device(size_t size,
                     Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return random_create_wrapper<mob::system::device>(size, seed);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector
runif_device(Rcpp::XPtr<mob::system::device::random> rngs, size_t n, double min,
             double max) {
  return runif_wrapper<mob::system::device>(rngs, n, min, max);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector
rbinom_device(Rcpp::XPtr<mob::system::device::random> rngs, size_t n,
              size_t size, double prob) {
  return rbinom_wrapper<mob::system::device>(rngs, n, size, prob);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector bernoulli_sampler_device(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return bernoulli_sampler_wrapper<mob::system::device>(data, p, seed);
}

// [[Rcpp::export]]
inline Rcpp::DataFrame homogeneous_infection_process_device(
    Rcpp::XPtr<mob::system::device::random> rngs,
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    double infection_probability) {
  return homogeneous_infection_process_wrapper<mob::system::device>(
      rngs, susceptible, infected, infection_probability);
}

// [[Rcpp::export]]
inline Rcpp::XPtr<mob::ds::partition<mob::system::device>>
partition_create_device(std::vector<uint32_t> population) {
  return partition_create_wrapper<mob::system::device>(std::move(population));
}

// [[Rcpp::export]]
inline Rcpp::DataFrame household_infection_process_device(
    Rcpp::XPtr<mob::system::device::random> rngs,
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    Rcpp::XPtr<mob::ds::partition<mob::system::device>> households,
    Rcpp::DoubleVector infection_probability) {
  return household_infection_process_wrapper<mob::system::device>(
      rngs, susceptible, infected, households, infection_probability);
}

#endif // __NVCC__

// [[Rcpp::export]]
inline Rcpp::XPtr<mob::system::host::random>
random_create_host(size_t size,
                   Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return random_create_wrapper<mob::system::host>(size, seed);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector
runif_host(Rcpp::XPtr<mob::system::host::random> rngs, size_t n, double min,
           double max) {
  return runif_wrapper<mob::system::host>(rngs, n, min, max);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector
rbinom_host(Rcpp::XPtr<mob::system::host::random> rngs, size_t n, size_t size,
            double prob) {
  return rbinom_wrapper<mob::system::host>(rngs, n, size, prob);
}

// [[Rcpp::export]]
inline Rcpp::DataFrame homogeneous_infection_process_host(
    Rcpp::XPtr<mob::system::host::random> rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, double infection_probability) {
  return homogeneous_infection_process_wrapper<mob::system::host>(
      rngs, susceptible, infected, infection_probability);
}

// [[Rcpp::export]]
inline Rcpp::XPtr<mob::ds::partition<mob::system::host>>
partition_create_host(std::vector<uint32_t> population) {
  return partition_create_wrapper<mob::system::host>(std::move(population));
}

// [[Rcpp::export]]
inline Rcpp::DataFrame household_infection_process_host(
    Rcpp::XPtr<mob::system::host::random> rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected,
    Rcpp::XPtr<mob::ds::partition<mob::system::host>> households,
    Rcpp::DoubleVector infection_probability) {
  return household_infection_process_wrapper<mob::system::host>(
      rngs, susceptible, infected, households, infection_probability);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector
bernoulli_sampler_host(Rcpp::NumericVector data, double p,
                       Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return bernoulli_sampler_wrapper<mob::system::host>(data, p, seed);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector
selection_sampler_host(Rcpp::NumericVector data, size_t k,
                       Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return selection_sampler_wrapper<mob::system::host>(data, k, seed);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector selection_sampler_device(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return selection_sampler_wrapper<mob::system::device>(data, k, seed);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector betabinomial_sampler_host(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return betabinomial_sampler_wrapper<mob::system::host>(data, k, seed);
}

// [[Rcpp::export]]
inline Rcpp::NumericVector betabinomial_sampler_device(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return betabinomial_sampler_wrapper<mob::system::device>(data, k, seed);
}

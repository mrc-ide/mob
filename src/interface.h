#include "mob_types.h"

#include <Rcpp.h>

inline size_t from_seed(Rcpp::Nullable<Rcpp::NumericVector> seed) {
  if (seed.isNotNull()) {
    return Rcpp::as<size_t>(seed);
  } else {
    return std::ceil(std::abs(R::unif_rand()) *
                     std::numeric_limits<size_t>::max());
  }
}

// Rcpp's compileAttributes doesn't scan `.cu` files for the Rcpp::export
// annotation. It does scan `.h` files though, which is why we put these
// prototypes in here.

// [[Rcpp::export]]
Rcpp::XPtr<RSystem::random>
device_random_create(size_t size,
                     Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export]]
Rcpp::NumericVector parallel_runif(Rcpp::XPtr<RSystem::random> rngs, size_t n,
                                   double min, double max);

// [[Rcpp::export]]
Rcpp::NumericVector parallel_rbinom(Rcpp::XPtr<RSystem::random> rngs, size_t n,
                                    size_t size, double prob);

// [[Rcpp::export("homogeneous_infection_process")]]
Rcpp::DataFrame homogeneous_infection_process_wrapper(
    Rcpp::XPtr<RSystem::random> rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, double infection_probability);

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::partition<RSystem>>
create_partition(std::vector<uint32_t> population);

// [[Rcpp::export("household_infection_process")]]
Rcpp::DataFrame household_infection_process_wrapper(
    Rcpp::XPtr<RSystem::random> rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected,
    Rcpp::XPtr<mob::ds::partition<RSystem>> households,
    Rcpp::DoubleVector infection_probability);

// [[Rcpp::export("betabinomial_sampler")]]
Rcpp::NumericVector betabinomial_sampler_wrapper(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("selection_sampler")]]
Rcpp::NumericVector selection_sampler_wrapper(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("bernoulli_sampler")]]
Rcpp::NumericVector bernoulli_sampler_wrapper(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("bernoulli_sampler_gpu")]]
Rcpp::NumericVector bernoulli_sampler_gpu_wrapper(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

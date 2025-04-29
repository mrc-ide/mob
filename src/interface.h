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
random_ptr
device_random_create(size_t size,
                     Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export]]
Rcpp::NumericVector parallel_runif(random_ptr rngs, size_t n, double min,
                                   double max);

// [[Rcpp::export]]
Rcpp::NumericVector parallel_rbinom(random_ptr rngs, size_t n, size_t size,
                                    double prob);

// [[Rcpp::export("homogeneous_infection_process")]]
Rcpp::DataFrame homogeneous_infection_process_wrapper(
    random_ptr rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, double infection_probability);

// [[Rcpp::export("household_infection_process")]]
Rcpp::DataFrame household_infection_process_wrapper(
    random_ptr rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, Rcpp::IntegerVector households,
    double infection_probability);

// [[Rcpp::export("betabinomial_sampler")]]
Rcpp::NumericVector betabinomial_sampler_wrapper(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("selection_sampler")]]
Rcpp::NumericVector selection_sampler_wrapper(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("bernoulli_sampler")]]
std::vector<double> bernouilli_sampler_wrapper(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("bernoulli_sampler_count")]]
size_t bernouilli_sampler_count_wrapper(
    size_t n, double p, Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("bernoulli_sampler_gpu")]]
Rcpp::NumericVector bernouilli_sampler_gpu_wrapper(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

// [[Rcpp::export("bernouilli_sampler_count_gpu")]]
size_t bernouilli_sampler_count_gpu_wrapper(
    size_t n, double p, Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue);

#include <Rcpp.h>

// Rcpp's compileAttributes doesn't scan `.cu` files for the Rcpp::export
// annotation. It does scan `.h` files though, which is why we put these
// prototypes in here.

// [[Rcpp::export]]
Rcpp::NumericVector parallel_runif(size_t n, double min, double max,
                                   int seed = 0);

// [[Rcpp::export]]
Rcpp::NumericVector parallel_rbinom(size_t n, size_t size, double prob,
                                    int seed = 0);

// [[Rcpp::export("betabinomial_sampler")]]
Rcpp::NumericVector betabinomial_sampler_wrapper(Rcpp::NumericVector data,
                                                 size_t k, int seed = 0);

// [[Rcpp::export("selection_sampler")]]
Rcpp::NumericVector selection_sampler_wrapper(Rcpp::NumericVector data,
                                              size_t k, int seed = 0);

// [[Rcpp::export("bernoulli_sampler")]]
std::vector<double> bernouilli_sampler_wrapper(Rcpp::NumericVector data,
                                               double p, int seed = 0);

// [[Rcpp::export("bernoulli_sampler_count")]]
size_t bernouilli_sampler_count_wrapper(size_t n, double p, int seed = 0);

// [[Rcpp::export("bernoulli_sampler_gpu")]]
Rcpp::NumericVector bernouilli_sampler_gpu_wrapper(Rcpp::NumericVector data,
                                                   double p, int seed = 0);

// [[Rcpp::export("bernouilli_sampler_count_gpu")]]
size_t bernouilli_sampler_count_gpu_wrapper(size_t n, double p, int seed = 0);

// [[Rcpp::export("homogeneous_infection_process")]]
Rcpp::DataFrame homogeneous_infection_process_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    double infection_probability, int seed = 0);

// [[Rcpp::export("homogeneous_infection_process_gpu")]]
Rcpp::DataFrame homogeneous_infection_process_gpu_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    double infection_probability, int seed = 0);

// [[Rcpp::export("household_infection_process")]]
Rcpp::DataFrame household_infection_process_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    Rcpp::IntegerVector households, double infection_probability, int seed = 0);

// [[Rcpp::export("household_infection_process_gpu")]]
Rcpp::DataFrame household_infection_process_gpu_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    Rcpp::IntegerVector households, double infection_probability, int seed = 0);

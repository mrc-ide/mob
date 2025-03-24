#pragma once
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector parallel_runif(size_t n, double min, double max, int seed);

// [[Rcpp::export]]
Rcpp::NumericVector parallel_rbinom(size_t n, size_t size, double prob,
                                    int seed);

// [[Rcpp::export("betabinomial_sampler")]]
Rcpp::NumericVector betabinomial_sampler_wrapper(Rcpp::NumericVector data,
                                                 size_t k, int seed);

// [[Rcpp::export("selection_sampler")]]
Rcpp::NumericVector selection_sampler_wrapper(Rcpp::NumericVector data,
                                              size_t k, int seed);

// [[Rcpp::export("bernoulli_sampler")]]
std::vector<double> bernouilli_sampler_wrapper(Rcpp::NumericVector data,
                                               double p, int seed);

// [[Rcpp::export("bernoulli_sampler_simulate")]]
size_t bernouilli_sampler_simulate(size_t n, double p, int seed);

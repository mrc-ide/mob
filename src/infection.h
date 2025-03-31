#pragma once
#include "parallel_random.h"
#include "roaring.h"
#include <Rcpp.h>

// [[Rcpp::export("homogeneous_infection_process")]]
Rcpp::DataFrame homogeneous_infection_process_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    double infection_probability, int seed = 0);

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
homogeneous_infection_process(mob::host_random<> &rngs,
                              const mob::roaring::bitset &infected,
                              const mob::roaring::bitset &susceptible,
                              double infection_probability);

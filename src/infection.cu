#include "interface.h"
#include "random_wrapper.h"
#include <mob/ds/partition.h>
#include <mob/infection.h>

Rcpp::DataFrame homogeneous_infection_process_wrapper(
    random_ptr rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, double infection_probability) {
  if ((*rngs)->size() < size_t(susceptible.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", (*rngs)->size(),
               susceptible.size());
  }
  if ((*rngs)->size() < size_t(infected.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", (*rngs)->size(),
               infected.size());
  }

  thrust::device_vector<uint32_t> infected_data(infected.begin(),
                                                infected.end());
  thrust::device_vector<uint32_t> susceptible_data(susceptible.begin(),
                                                   susceptible.end());
  auto [source, victim] =
      mob::homogeneous_infection_process<mob::system::device>(
          **rngs, infected_data, susceptible_data, infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

Rcpp::DataFrame household_infection_process_wrapper(
    random_ptr rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, Rcpp::IntegerVector households,
    double infection_probability) {
  if ((*rngs)->size() < size_t(susceptible.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", (*rngs)->size(),
               susceptible.size());
  }
  if ((*rngs)->size() < size_t(infected.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", (*rngs)->size(),
               susceptible.size());
  }
  if (*std::max_element(susceptible.begin(), susceptible.end()) >=
      households.size()) {
    Rcpp::stop("bad susceptible");
  }
  if (*std::max_element(infected.begin(), infected.end()) >=
      households.size()) {
    Rcpp::stop("bad infected");
  }
  // This is needed for binary search / fast intersection
  if (!std::is_sorted(susceptible.begin(), susceptible.end())) {
    Rcpp::stop("susceptible must be sorted");
  }

  thrust::device_vector<uint32_t> infected_data(infected.begin(),
                                                infected.end());
  thrust::device_vector<uint32_t> susceptible_data(susceptible.begin(),
                                                   susceptible.end());

  mob::ds::partition<mob::system::device> household_partition(
      {households.begin(), households.end()});

  auto [source, victim] = mob::household_infection_process<mob::system::device>(
      **rngs, infected_data, susceptible_data, household_partition,
      infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

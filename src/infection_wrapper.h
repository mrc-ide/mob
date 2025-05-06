#pragma once

#include <mob/ds/partition.h>
#include <mob/infection.h>

template <typename System>
Rcpp::DataFrame homogeneous_infection_process_wrapper(
    Rcpp::XPtr<typename System::random> rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected, double infection_probability) {
  if (rngs->size() < size_t(susceptible.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               susceptible.size());
  }
  if (rngs->size() < size_t(infected.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               infected.size());
  }

  typename System::vector<uint32_t> infected_data(infected.begin(),
                                                  infected.end());
  typename System::vector<uint32_t> susceptible_data(susceptible.begin(),
                                                     susceptible.end());
  auto [source, victim] = mob::homogeneous_infection_process<System>(
      *rngs, infected_data, susceptible_data, infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

template <typename System>
Rcpp::XPtr<mob::ds::partition<System>>
partition_create_wrapper(std::vector<uint32_t> population) {
  return Rcpp::XPtr(new mob::ds::partition<System>(std::move(population)));
}

template <typename System>
Rcpp::DataFrame household_infection_process_wrapper(
    Rcpp::XPtr<typename System::random> rngs, Rcpp::IntegerVector susceptible,
    Rcpp::IntegerVector infected,
    Rcpp::XPtr<mob::ds::partition<System>> households,
    Rcpp::DoubleVector infection_probability) {
  if (rngs->size() < size_t(susceptible.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               susceptible.size());
  }
  if (rngs->size() < size_t(infected.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               susceptible.size());
  }
  if (susceptible.size() != 0 &&
      *std::max_element(susceptible.begin(), susceptible.end()) >=
          households->population_size()) {
    Rcpp::stop("bad susceptible");
  }
  if (infected.size() != 0 &&
      *std::max_element(infected.begin(), infected.end()) >=
          households->population_size()) {
    Rcpp::stop("bad infected");
  }
  // This is needed for binary search / fast intersection
  if (!std::is_sorted(susceptible.begin(), susceptible.end())) {
    Rcpp::stop("susceptible must be sorted");
  }
  if (infection_probability.size() != 1 &&
      infection_probability.size() != households->partitions_count()) {
    Rcpp::stop("infection probability size is incorrect: got %d households but "
               "%d probabilities",
               households->partitions_count(), infection_probability.size());
  }

  typename System::vector<uint32_t> infected_data(infected.begin(),
                                                  infected.end());
  typename System::vector<uint32_t> susceptible_data(susceptible.begin(),
                                                     susceptible.end());
  typename System::vector<double> infection_probability_data(
      infection_probability.begin(), infection_probability.end());

  auto [source, victim] = mob::household_infection_process<System>(
      *rngs, infected_data, susceptible_data, *households,
      infection_probability_data);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

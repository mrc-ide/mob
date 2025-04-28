#include "interface.h"
#include <mob/ds/partition.h>
#include <mob/infection.h>

Rcpp::DataFrame
homogeneous_infection_process_wrapper(Rcpp::IntegerVector susceptible,
                                      Rcpp::IntegerVector infected,
                                      double infection_probability, int seed) {
  size_t rng_size = std::max<size_t>(susceptible.size(), infected.size());
  mob::host_random rngs(rng_size, seed);

  thrust::host_vector<uint32_t> infected_data(infected.begin(), infected.end());
  thrust::host_vector<uint32_t> susceptible_data(susceptible.begin(),
                                                 susceptible.end());
  auto [source, victim] = mob::homogeneous_infection_process(
      rngs, infected_data, susceptible_data, infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

Rcpp::DataFrame homogeneous_infection_process_gpu_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    double infection_probability, int seed) {

  size_t rng_size = std::max<size_t>(susceptible.size(), infected.size());
  mob::device_random rngs(rng_size, seed);

  thrust::device_vector<uint32_t> infected_data(infected.begin(),
                                                infected.end());
  thrust::device_vector<uint32_t> susceptible_data(susceptible.begin(),
                                                   susceptible.end());
  auto [source, victim] =
      mob::homogeneous_infection_process<mob::system::device>(
          rngs, infected_data, susceptible_data, infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

Rcpp::DataFrame household_infection_process_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    Rcpp::IntegerVector households, double infection_probability, int seed) {
  size_t population = households.size();
  if (*std::max_element(susceptible.begin(), susceptible.end()) >= population) {
    Rcpp::stop("bad susceptible");
  }
  if (*std::max_element(infected.begin(), infected.end()) >= population) {
    Rcpp::stop("bad infected");
  }
  if (!std::is_sorted(susceptible.begin(), susceptible.end())) {
    Rcpp::stop("susceptible must be sorted");
  }
  if (!std::is_sorted(infected.begin(), infected.end())) {
    Rcpp::stop("infected must be sorted");
  }

  mob::host_random rngs(population, seed);
  thrust::host_vector<uint32_t> infected_data(infected.begin(), infected.end());
  thrust::host_vector<uint32_t> susceptible_data(susceptible.begin(),
                                                 susceptible.end());

  mob::ds::partition<mob::system::host> household_partition(
      {households.begin(), households.end()});

  auto [source, victim] = mob::household_infection_process<mob::system::host>(
      rngs, infected_data, susceptible_data, household_partition,
      infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

Rcpp::DataFrame household_infection_process_gpu_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    Rcpp::IntegerVector households, double infection_probability, int seed) {
  size_t population = households.size();
  if (*std::max_element(susceptible.begin(), susceptible.end()) >= population) {
    Rcpp::stop("bad susceptible");
  }
  if (*std::max_element(infected.begin(), infected.end()) >= population) {
    Rcpp::stop("bad infected");
  }
  if (!std::is_sorted(susceptible.begin(), susceptible.end())) {
    Rcpp::stop("susceptible must be sorted");
  }
  if (!std::is_sorted(infected.begin(), infected.end())) {
    Rcpp::stop("infected must be sorted");
  }

  mob::device_random rngs(population, seed);
  thrust::device_vector<uint32_t> infected_data(infected.begin(),
                                                infected.end());
  thrust::device_vector<uint32_t> susceptible_data(susceptible.begin(),
                                                   susceptible.end());

  mob::ds::partition<mob::system::device> household_partition(
      {households.begin(), households.end()});

  auto [source, victim] = mob::household_infection_process<mob::system::device>(
      rngs, infected_data, susceptible_data, household_partition,
      infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

#include "interface.h"
#include <mob/infection.h>

Rcpp::DataFrame
homogeneous_infection_process_wrapper(Rcpp::IntegerVector susceptible,
                                      Rcpp::IntegerVector infected,
                                      double infection_probability, int seed) {
  size_t rng_size = std::max<size_t>(susceptible.size(), infected.size());
  mob::host_random rngs(rng_size, seed);

  auto [source, victim] = mob::homogeneous_infection_process(
      rngs, infected, susceptible, infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

Rcpp::DataFrame homogeneous_infection_process_gpu_wrapper(
    Rcpp::IntegerVector susceptible, Rcpp::IntegerVector infected,
    double infection_probability, int seed) {

  size_t rng_size = std::max<size_t>(susceptible.size(), infected.size());
  mob::device_random rngs(rng_size, seed);

  auto [source, victim] =
      mob::homogeneous_infection_process<mob::system::device>(
          rngs, infected, susceptible, infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

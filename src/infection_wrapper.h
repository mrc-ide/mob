#pragma once

#include "conversion.h"
#include <mob/ds/partition.h>
#include <mob/infection.h>
#include <mob/spatial.h>

template <typename System>
Rcpp::XPtr<mob::infection_list<System>> infection_list_create_wrapper() {
  return Rcpp::XPtr(new mob::infection_list<System>());
}

template <typename System>
size_t homogeneous_infection_process_wrapper(
    Rcpp::XPtr<mob::parallel_random<System>> rngs,
    Rcpp::XPtr<mob::infection_list<System>> output,
    Rcpp::XPtr<mob::bitset<System>> susceptible,
    Rcpp::XPtr<mob::bitset<System>> infected, double infection_probability) {
  if (rngs->size() < susceptible->capacity()) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               susceptible->capacity());
  }
  if (rngs->size() < infected->capacity()) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               infected->capacity());
  }

  auto infected_data = mob::bitset_view(*infected).to_vector();
  return mob::homogeneous_infection_process<System>(
      *rngs, *output, infected_data, *susceptible, infection_probability);
}

// TODO: why does this not cause an error when infection_probability is NULL?
template <typename System>
size_t household_infection_process_wrapper(
    Rcpp::XPtr<mob::parallel_random<System>> rngs,
    Rcpp::XPtr<mob::infection_list<System>> output,
    Rcpp::XPtr<mob::bitset<System>> susceptible,
    Rcpp::XPtr<mob::bitset<System>> infected,
    Rcpp::XPtr<mob::ds::partition<System>> households,
    Rcpp::DoubleVector infection_probability) {
  if (rngs->size() < susceptible->capacity()) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               susceptible->capacity());
  }
  if (rngs->size() < infected->capacity()) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(),
               infected->capacity());
  }
  if (susceptible->capacity() != households->population_size()) {
    Rcpp::stop("bad susceptible");
  }
  if (infected->capacity() != households->population_size()) {
    Rcpp::stop("bad susceptible");
  }
  if (infection_probability.size() != 1 &&
      households->partitions_count() !=
          static_cast<size_t>(infection_probability.size())) {
    Rcpp::stop("infection probability size is incorrect: got %d households but "
               "%d probabilities",
               households->partitions_count(), infection_probability.size());
  }

  mob::vector<System, double> infection_probability_data(
      infection_probability.begin(), infection_probability.end());

  auto infected_data = mob::bitset_view(*infected).to_vector();
  return mob::household_infection_process<System>(*rngs, *output, infected_data,
                                                  *susceptible, *households,
                                                  infection_probability_data);
}

template <typename System>
size_t
spatial_infection_naive_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs,
                                Rcpp::XPtr<mob::infection_list<System>> output,
                                Rcpp::XPtr<mob::bitset<System>> susceptible,
                                Rcpp::XPtr<mob::bitset<System>> infected,
                                Rcpp::NumericVector x, Rcpp::NumericVector y,
                                double base, double k) {
  if (rngs->size() < susceptible->capacity()) {
    Rcpp::stop("bad rng");
  }
  if (infected->capacity() != susceptible->capacity()) {
    Rcpp::stop("bad infected");
  }
  if (static_cast<size_t>(x.size()) != susceptible->capacity()) {
    Rcpp::stop("bad x");
  }
  if (static_cast<size_t>(y.size()) != susceptible->capacity()) {
    Rcpp::stop("bad x");
  }

  mob::spatial<System> spatial_data{{x.begin(), x.end()}, {y.begin(), y.end()}};
  return mob::spatial_infection_naive<System>(
      *rngs, *output, infected->to_vector(), susceptible->to_vector(),
      spatial_data, base, k);
}

template <typename System>
size_t
spatial_infection_sieve_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs,
                                Rcpp::XPtr<mob::infection_list<System>> output,
                                Rcpp::XPtr<mob::bitset<System>> susceptible,
                                Rcpp::XPtr<mob::bitset<System>> infected,
                                Rcpp::NumericVector x, Rcpp::NumericVector y,
                                double base, double k) {
  if (rngs->size() < susceptible->capacity()) {
    Rcpp::stop("bad rng");
  }
  if (infected->capacity() != susceptible->capacity()) {
    Rcpp::stop("bad infected");
  }
  if (static_cast<size_t>(x.size()) != susceptible->capacity()) {
    Rcpp::stop("bad x");
  }
  if (static_cast<size_t>(y.size()) != susceptible->capacity()) {
    Rcpp::stop("bad x");
  }

  mob::spatial<System> spatial_data{{x.begin(), x.end()}, {y.begin(), y.end()}};
  return mob::spatial_infection_sieve<System>(
      *rngs, *output, infected->to_vector(), susceptible->to_vector(),
      spatial_data, base, k);
}

template <typename System>
Rcpp::IntegerVector
spatial_infection_hybrid_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs,
                                 Rcpp::XPtr<mob::infection_list<System>> output,
                                 Rcpp::XPtr<mob::bitset<System>> susceptible,
                                 Rcpp::XPtr<mob::bitset<System>> infected,
                                 Rcpp::NumericVector x, Rcpp::NumericVector y,
                                 double base, double k, double width) {
  if (rngs->size() < susceptible->capacity()) {
    Rcpp::stop("bad rng");
  }
  if (infected->capacity() != susceptible->capacity()) {
    Rcpp::stop("bad infected");
  }
  if (static_cast<size_t>(x.size()) != susceptible->capacity()) {
    Rcpp::stop("bad x");
  }
  if (static_cast<size_t>(y.size()) != susceptible->capacity()) {
    Rcpp::stop("bad x");
  }

  mob::spatial<System> spatial_data{{x.begin(), x.end()}, {y.begin(), y.end()}};
  auto [n1, n2] = mob::spatial_infection_hybrid<System>(
      *rngs, *output, infected->to_vector(), susceptible->to_vector(),
      spatial_data, base, k, width);
  return {static_cast<int>(n1), static_cast<int>(n2)};
}

template <typename System>
Rcpp::XPtr<mob::bitset<System>>
infection_victims_wrapper(Rcpp::XPtr<mob::infection_list<System>> infections,
                          size_t capacity) {
  // Returning this as a bitset may be overkill - it might be more suitable as a
  // vector.
  //
  // Also having capacity as an argument here is a bit weird. Maybe it needs to
  // be moved to infection_list_create.
  auto result = Rcpp::XPtr(new mob::bitset<System>(capacity));
  result->insert(infection_victims(*infections));
  return result;
}

template <typename System>
Rcpp::DataFrame
infections_as_dataframe(Rcpp::XPtr<mob::infection_list<System>> infections) {
  return Rcpp::DataFrame::create(
      Rcpp::Named("source") =
          asRcppVector<ConvertIndex::Yes>(infections->sources),

      Rcpp::Named("victim") =
          asRcppVector<ConvertIndex::Yes>(infections->victims));
}

template <typename System>
Rcpp::XPtr<mob::infection_list<System>>
infections_from_dataframe(Rcpp::DataFrame df) {
  Rcpp::IntegerVector sources = df["source"];
  Rcpp::IntegerVector victims = df["victim"];

  return Rcpp::XPtr(new mob::infection_list<System>(
      fromRcppVector<System, uint32_t, ConvertIndex::Yes>(sources),
      fromRcppVector<System, uint32_t, ConvertIndex::Yes>(victims)));
}

template <typename System>
Rcpp::XPtr<mob::infection_list<System>>
infections_select_wrapper(Rcpp::XPtr<mob::parallel_random<System>> rngs,
                          Rcpp::XPtr<mob::infection_list<System>> infections) {
  return Rcpp::XPtr(
      new mob::infection_list<System>(infections_select(*rngs, *infections)));
}

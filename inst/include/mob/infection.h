#pragma once
#include <mob/parallel_random.h>
#include <mob/roaring/bitset.h>
#include <mob/sample.h>

#include <Rcpp.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/zip_function.h>

namespace mob {

// Compute the length of contiguous runs
template <typename InputIt, typename OutputIt1, typename OutputIt2>
std::pair<OutputIt1, OutputIt2> run_lengths(InputIt first, InputIt last,
                                            OutputIt1 keys, OutputIt2 values) {
  return thrust::reduce_by_key(
      first, last, thrust::constant_iterator<uint32_t>(1), keys, values);
}

/**
 * Given a segmented range of keys, for each segment, choose one of the indices.
 */
template <typename KeyIt, typename OutputKeyIt, typename OutputValueIt>
std::pair<OutputKeyIt, OutputValueIt>
uniform_index_by_key(mob::host_random &rngs, KeyIt first, KeyIt last,
                     OutputKeyIt output_key, OutputValueIt output_value) {
  size_t maxn = std::distance(first, last);
  std::vector<size_t> boundaries(maxn + 1);

  auto [keys_last, count_last] =
      run_lengths(first, last, output_key, boundaries.begin());
  size_t n = std::distance(output_key, keys_last);

  thrust::exclusive_scan(boundaries.begin(), count_last + 1,
                         boundaries.begin());

  auto output_last = thrust::transform(
      thrust::make_zip_iterator(boundaries.begin(), boundaries.begin() + 1,
                                rngs.begin()),
      thrust::make_zip_iterator(count_last, count_last + 1, rngs.begin() + n),
      output_value,
      thrust::make_zip_function(
          [](uint32_t lower_bound, uint32_t upper_bound, auto &rng) {
            return dust::random::uniform(rng, lower_bound, upper_bound);
          }));

  return {keys_last, output_last};
}

template <typename KeyIt, typename ValueIt, typename OutputKeyIt,
          typename OutputValueIt>
std::pair<OutputKeyIt, OutputValueIt>
random_select_by_key(mob::host_random &rngs, KeyIt first, KeyIt last,
                     ValueIt values, OutputKeyIt output_key,
                     OutputValueIt output_value) {
  std::vector<size_t> indices(last - first);
  auto [keys_last, indices_last] =
      uniform_index_by_key(first, last, output_key, indices.begin());
  auto values_last =
      thrust::gather(indices.begin(), indices_last, values, output_value);
  return {keys_last, values_last};
}

// TODO: bitset to device_vector should have a GPU-based specialization
// TODO: in some cases we should be able to move instead
template <typename System, typename Input>
typename System::vector<uint32_t> to_population_vector(const Input &values) {
  return typename System::vector<uint32_t>(values.begin(), values.end());
}

template <typename System = system::host, typename Infected,
          typename CandidatesSize, typename CandidatesFn>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
infection_process(typename System::random &rngs, Infected &&infected,
                  double infection_probability, CandidatesSize candidates_size,
                  CandidatesFn candidates_fn) {

  auto infectors =
      to_population_vector<System>(std::forward<Infected>(infected));

  // Figure out how many people each I infects - don't store the actual targets
  // anywhere yet since we have nowhere to put the result.
  typename System::vector<uint32_t> victim_count(infectors.size());
  thrust::transform(
      infectors.begin(), infectors.end(), rngs.begin(), victim_count.begin(),
      [candidates_size, infection_probability] __host__ __device__(
          uint32_t i, typename System::random::proxy rng) -> uint32_t {
        size_t n = candidates_size(i);
        auto rng_copy = rng.get();
        return mob::bernouilli_sampler_count(rng_copy, n,
                                             infection_probability);
      });

  // Prepare some space in which to store the infection victims.
  // The cumulative sum of the victim count gives us the offset at which each I
  // should store its victims.
  typename System::vector<size_t> offsets(infectors.size());
  thrust::exclusive_scan(victim_count.begin(), victim_count.end(),
                         offsets.begin());

  size_t total_infections;
  if (infectors.size() > 0) {
    total_infections = offsets.back() + victim_count.back();
  } else {
    total_infections = 0;
  }

  typename System::vector<uint32_t> infection_victim(total_infections);
  typename System::vector<uint32_t> infection_source(total_infections);

  // Select all the victims. This uses the same RNG state as our earlier
  // sampling, guaranteeing that we'll get the same number as we had allocated.
  //
  // This produces (victim, source) tuples for each infection.

  auto infection_victim_begin = infection_victim.begin();
  auto infection_source_begin = infection_source.begin();

  thrust::for_each(
      thrust::make_zip_iterator(infectors.begin(), offsets.begin(),
                                rngs.begin()),
      thrust::make_zip_iterator(infectors.end(), offsets.end(),
                                rngs.begin() + infectors.size()),
      thrust::make_zip_function(
          [candidates_fn, infection_probability, infection_victim_begin,
           infection_source_begin] __host__
              __device__(uint32_t i, size_t offset,
                         typename System::random::proxy rng) {
                auto candidates = candidates_fn(i);

                auto victim_first = infection_victim_begin + offset;
                auto victim_last = mob::bernouilli_sampler<double>(
                    rng, candidates.first, candidates.second, victim_first,
                    infection_probability);

                auto source_first = infection_source_begin + offset;
                auto source_last = infection_source_begin +
                                   (victim_last - infection_victim_begin);

                // TODO: use std::fill
                for (auto it = source_first; it != source_last; it++) {
                  *it = i;
                }
              }));

  return {infection_source, infection_victim};
}

template <typename System = system::host, typename Infected,
          typename Susceptible>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
homogeneous_infection_process(typename System::random &rngs,
                              Infected &&infected, Susceptible &&susceptible,
                              double infection_probability) {

  auto susceptible_data =
      to_population_vector<System>(std::forward<Susceptible>(susceptible));

  auto susceptible_size = susceptible_data.size();
  auto susceptible_begin = susceptible_data.begin();
  auto susceptible_end = susceptible_data.end();

  return infection_process<System>(
      rngs, std::forward<Infected>(infected), infection_probability,
      [=] __host__ __device__(uint32_t) { return susceptible_size; },
      [=] __host__ __device__(uint32_t) {
        // TODO: use spans or ranges
        return thrust::make_pair(susceptible_begin, susceptible_end);
      });
}

/*
std::
    pair<std::vector<uint32_t>, std::vector<uint32_t>> static inline
household_infection_process( mob::host_random &rngs, const
mob::roaring::bitset &infected, const mob::roaring::bitset &susceptible, double
infection_probability, const std::vector<uint32_t> &individual_household, const
std::vector<mob::roaring::bitset> &household_members) { return
infection_process( rngs, infected, infection_probability,
      [&](uint32_t i) {
        return mob::roaring::intersection_size(
            household_members[individual_household[i]], susceptible);
      },
      [&](uint32_t i) {
        return mob::roaring::intersection(
            household_members[individual_household[i]], susceptible);
      });
}
*/

} // namespace mob

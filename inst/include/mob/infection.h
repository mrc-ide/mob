#pragma once
#include <mob/ds/intersection.h>
#include <mob/ds/partition.h>
#include <mob/ds/view.h>
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

template <typename System = system::host, typename CandidatesSize,
          typename CandidatesFn>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
infection_process(typename System::random &rngs,
                  const typename System::vector<uint32_t> &infected,
                  double infection_probability, CandidatesSize candidates_size,
                  CandidatesFn candidates_fn) {
  // Figure out how many people each I infects - don't store the actual targets
  // anywhere yet since we have nowhere to put the result.
  typename System::vector<uint32_t> victim_count(infected.size());
  thrust::transform(
      infected.begin(), infected.end(), rngs.begin(), victim_count.begin(),
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
  typename System::vector<size_t> offsets(infected.size());
  thrust::exclusive_scan(victim_count.begin(), victim_count.end(),
                         offsets.begin());

  size_t total_infections;
  if (infected.size() > 0) {
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
      thrust::make_zip_iterator(infected.begin(), offsets.begin(),
                                rngs.begin()),
      thrust::make_zip_iterator(infected.end(), offsets.end(),
                                rngs.begin() + infected.size()),
      thrust::make_zip_function(
          [=] __host__ __device__(uint32_t i, size_t offset,
                                  typename System::random::proxy rng) {
            auto candidates = candidates_fn(i);

            auto victim_first = infection_victim_begin + offset;

            // TODO: might be beneficial to cache candidates_size, which is
            // needed by the sampler.
            auto victim_last = mob::bernouilli_sampler<double>(
                rng, candidates.begin(), candidates.end(), victim_first,
                infection_probability);

            auto source_first = infection_source_begin + offset;
            auto source_last =
                infection_source_begin + (victim_last - infection_victim_begin);

            compat::fill(source_first, source_last, i);
          }));

  return {infection_source, infection_victim};
}

template <typename System = system::host, typename CandidatesFn>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
infection_process(typename System::random &rngs,
                  const typename System::vector<uint32_t> &infected,
                  double infection_probability, CandidatesFn candidates_fn) {
  return infection_process<System>(
      rngs, infected, infection_probability,
      [=] __host__ __device__(uint32_t i) {
        return compat::distance(candidates_fn(i));
      },
      [=] __host__ __device__(uint32_t i) { return candidates_fn(i); });
}

template <typename System = system::host>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
homogeneous_infection_process(
    typename System::random &rngs,
    const typename System::vector<uint32_t> &infected,
    const typename System::vector<uint32_t> &susceptible,
    double infection_probability) {

  auto susceptible_view = ds::view(susceptible);

  return infection_process<System>(
      rngs, infected, infection_probability,
      [=] __host__ __device__(uint32_t) { return susceptible_view; });
}

template <typename System = system::host>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
household_infection_process(
    typename System::random &rngs,
    const typename System::vector<uint32_t> &infected,
    const typename System::vector<uint32_t> &susceptible,
    const ds::partition<System> &partition, double infection_probability) {

  auto susceptible_view = ds::view(susceptible);
  auto partition_view = ds::view(partition);

  // TODO: this applies the S filter first, and then applies the bernouilli
  // sampler. It may be easier / faster to do the bernouilli sample first and
  // apply the filter second.
  //
  // refactor `infection_process` to not do the bernouilli sampler and instead
  // make it the responsibility of the callbacks.
  //
  // Make bernouilli_sampler a range and use std::range::distance on it.
  return infection_process<System>(rngs, infected, infection_probability,
                                   [=] __host__ __device__(uint32_t i) {
                                     return lazy_intersection(
                                         partition_view.neighbours(i),
                                         susceptible_view);
                                   });
}

} // namespace mob

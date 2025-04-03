#include "infection.h"
#include "parallel_random.h"
#include "roaring.h"
#include "sample.h"
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/zip_function.h>

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
uniform_index_by_key(mob::host_random<> &rngs, KeyIt first, KeyIt last,
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
random_select_by_key(mob::host_random<> &rngs, KeyIt first, KeyIt last,
                     ValueIt values, OutputKeyIt output_key,
                     OutputValueIt output_value) {
  std::vector<size_t> indices(last - first);
  auto [keys_last, indices_last] =
      uniform_index_by_key(first, last, output_key, indices.begin());
  auto values_last =
      thrust::gather(indices.begin(), indices_last, values, output_value);
  return {keys_last, values_last};
}

template <typename CandidatesSize, typename CandidatesFn>
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
infection_process(mob::host_random<> &rngs,
                  const mob::roaring::bitset &infected,
                  double infection_probability, CandidatesSize candidates_size,
                  CandidatesFn candidates_fn) {
  // TODO: parallelize the bitmap to list operation on the GPU
  std::vector<uint32_t> infectors(infected.begin(), infected.end());

  // Figure out how many people each I infects - don't store the actual targets
  // anywhere yet since we have nowhere to put the result.
  std::vector<uint32_t> victim_count(infectors.size());
  thrust::transform(infectors.begin(), infectors.end(), rngs.begin(),
                    victim_count.begin(), [&](uint32_t i, auto &rng) {
                      size_t n = candidates_size(i);
                      auto rng_copy = rng.get();
                      return mob::bernouilli_sampler_count(
                          rng_copy, n, infection_probability);
                    });

  // Prepare some space in which to store the infection victims.
  // The cumulative sum of the victim count gives us the offset at which each I
  // should store its victims.
  std::vector<size_t> offsets(infectors.size());
  thrust::exclusive_scan(victim_count.begin(), victim_count.end(),
                         offsets.begin());

  size_t total_infections;
  if (infectors.size() > 0) {
    total_infections = offsets.back() + victim_count.back();
  } else {
    total_infections = 0;
  }

  std::vector<uint32_t> infection_victim(total_infections);
  std::vector<uint32_t> infection_source(total_infections);

  // Select all the victims. This uses the same RNG state as our earlier
  // sampling, guaranteeing that we'll get the same number as we had allocated.
  //
  // This produces (victim, source) tuples for each infection.
  thrust::for_each(
      thrust::make_zip_iterator(infectors.begin(), offsets.begin(),
                                rngs.begin()),
      thrust::make_zip_iterator(infectors.end(), offsets.end(),
                                rngs.begin() + infectors.size()),
      thrust::make_zip_function([&](uint32_t i, size_t offset, auto &rng) {
        decltype(auto) candidates = candidates_fn(i);
        auto victim_first = infection_victim.begin() + offset;
        auto victim_last =
            mob::bernouilli_sampler(rng, candidates.begin(), candidates.end(),
                                    victim_first, infection_probability);

        auto source_first = infection_source.begin() + offset;
        auto source_last =
            infection_source.begin() + (victim_last - infection_victim.begin());

        std::fill(source_first, source_last, i);
      }));

  return {infection_source, infection_victim};
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
homogeneous_infection_process(mob::host_random<> &rngs,
                              const mob::roaring::bitset &infected,
                              const mob::roaring::bitset &susceptible,
                              double infection_probability) {
  return infection_process(
      rngs, infected, infection_probability,
      [&](uint32_t) { return susceptible.size(); },
      [&](uint32_t) -> const auto & { return susceptible; });
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
household_infection_process(
    mob::host_random<> &rngs, const mob::roaring::bitset &infected,
    const mob::roaring::bitset &susceptible, double infection_probability,
    const std::vector<uint32_t> &individual_household,
    const std::vector<mob::roaring::bitset> &household_members) {
  return infection_process(
      rngs, infected, infection_probability,
      [&](uint32_t i) {
        return mob::roaring::intersection_size(
            household_members[individual_household[i]], susceptible);
      },
      [&](uint32_t i) {
        return mob::roaring::intersection(
            household_members[individual_household[i]], susceptible);
      });
}

Rcpp::DataFrame
homogeneous_infection_process_wrapper(Rcpp::IntegerVector susceptible,
                                      Rcpp::IntegerVector infected,
                                      double infection_probability, int seed) {
  size_t rng_size = std::max<size_t>(susceptible.size(), infected.size());
  mob::host_random<> rngs(rng_size, seed);

  auto [source, victim] = homogeneous_infection_process(
      rngs, mob::roaring::bitset(susceptible.begin(), susceptible.end()),
      mob::roaring::bitset(infected.begin(), infected.end()),
      infection_probability);

  return Rcpp::DataFrame::create(Rcpp::Named("source") = source,
                                 Rcpp::Named("victim") = victim);
}

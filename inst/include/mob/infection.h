#pragma once
#include <mob/bernoulli.h>
#include <mob/ds/partition.h>
#include <mob/ds/span.h>
#include <mob/intersection.h>
#include <mob/parallel_random.h>
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
  size_t maxn = cuda::std::distance(first, last);
  std::vector<size_t> boundaries(maxn + 1);

  auto [keys_last, count_last] =
      run_lengths(first, last, output_key, boundaries.begin());
  size_t n = cuda::std::distance(output_key, keys_last);

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

template <typename System, typename VictimsFn>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
infection_process(typename System::random &rngs,
                  typename System::span<uint32_t> infected,
                  VictimsFn victims_fn) {
  // Figure out how many people each I infects - don't store the actual targets
  // anywhere yet since we have nowhere to put the result.
  typename System::vector<uint32_t> victim_count(infected.size());
  thrust::transform(
      infected.begin(), infected.end(), rngs.begin(), victim_count.begin(),
      [=] __host__ __device__(uint32_t i,
                              typename System::random::proxy rng) -> uint32_t {
        // Ideally we'd pass rng_copy directly to victims_fn. Unfortunately
        // CUDA doesn't support generic lambdas, which mean victims_fn can only
        // support one type of RNG state and rng and rng_copy have different
        // types. We workaround this by doing a get / put, and hope the
        // compiler is clever enough to remove the superfluous stores.
        auto rng_copy = rng.get();
        auto result = cuda::std::ranges::distance(victims_fn(i, rng));
        rng.put(rng_copy);
        return result;
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
            auto victims = victims_fn(i, rng);
            auto victim_first = infection_victim_begin + offset;
            auto victim_last = compat::copy(victims, victim_first);

            auto source_first = infection_source_begin + offset;
            auto source_last =
                infection_source_begin + (victim_last - infection_victim_begin);

            compat::fill(source_first, source_last, i);
          }));

  return {infection_source, infection_victim};
}

template <typename System>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
homogeneous_infection_process(typename System::random &rngs,
                              typename System::span<uint32_t> infected,
                              typename System::span<uint32_t> susceptible,
                              double infection_probability) {

  return infection_process<System>(
      rngs, infected,
      [=] __host__ __device__(uint32_t, typename System::random::proxy &rng) {
        return bernoulli(susceptible, infection_probability, rng);
      });
}

template <typename System>
std::pair<typename System::vector<uint32_t>, typename System::vector<uint32_t>>
household_infection_process(typename System::random &rngs,
                            ds::span<System, uint32_t> infected,
                            ds::span<System, uint32_t> susceptible,
                            ds::partition_view<System> partition,
                            ds::span<System, double> infection_probability) {
  // TODO: this applies the S filter first, and then applies the bernoulli
  // sampler. It may be faster to do the bernoulli sample first and apply
  // the filter second.
  return infection_process<System>(
      rngs, infected,
      [=] __host__ __device__(uint32_t i, typename System::random::proxy &rng) {
        double p;
        if (infection_probability.size() == 1) {
          p = infection_probability[0];
        } else {
          p = infection_probability[partition.get_partition(i)];
        }
        auto candidates = intersection(partition.neighbours(i), susceptible);
        return bernoulli(candidates, p, rng);
      });
}

} // namespace mob

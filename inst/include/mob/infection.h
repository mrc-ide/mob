#pragma once
#include <mob/bernoulli.h>
#include <mob/bitset.h>
#include <mob/ds/partition.h>
#include <mob/ds/span.h>
#include <mob/intersection.h>
#include <mob/parallel_random.h>
#include <mob/sample.h>

#include <Rcpp.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/zip_function.h>

namespace mob {

template <typename System>
struct infection_list {
  mob::vector<System, uint32_t> sources;
  mob::vector<System, uint32_t> victims;

  std::pair<mob::ds::span<System, uint32_t>, mob::ds::span<System, uint32_t>>
  grow(size_t n) {
    size_t size = sources.size();
    // This does value-initialization (ie. zero) of the two vectors, which isn't
    // necessary since they get overwritten by the for_each. Thrust does not
    // have an easy way of avoiding this yet (neither does the STL).
    // https://github.com/NVIDIA/cccl/issues/1992
    // https://github.com/NVIDIA/cccl/pull/4183
    sources.resize(size + n);
    victims.resize(size + n);
    return std::make_pair(mob::ds::span(sources).subspan(size, n),
                          mob::ds::span(victims).subspan(size, n));
  }
};

template <typename System, typename VictimsFn>
  requires requires(VictimsFn victims_fn, uint32_t i,
                    mob::random_proxy<System> &rng) {
    { victims_fn(i, rng) } -> std::ranges::input_range;
  }
size_t infection_process(mob::parallel_random<System> &rngs,
                         infection_list<System> &output,
                         mob::ds::span<System, uint32_t> infected,
                         VictimsFn victims_fn) {
  // Figure out how many people each I infects - don't store the actual targets
  // anywhere yet since we have nowhere to put the result.
  mob::vector<System, size_t> victim_count(infected.size());
  thrust::transform(
      infected.begin(), infected.end(), rngs.begin(), victim_count.begin(),
      [=] __host__ __device__(uint32_t i,
                              mob::random_proxy<System> &rng) -> size_t {
        // Ideally we'd pass rng_copy directly to victims_fn. Unfortunately
        // CUDA doesn't support generic lambdas, which, assuming it is a lambda,
        // means victims_fn can only support one type of RNG state and rng and
        // rng_copy have different types. We workaround this by doing a get /
        // put, and hope the compiler is clever enough to remove the redundant
        // stores.
        auto rng_copy = rng.get();
        auto result = cuda::std::ranges::distance(victims_fn(i, rng));
        rng.put(rng_copy);
        return result;
      });

  // Prepare some space in which to store the infection victims.
  // The cumulative sum of the victim count gives us the offset at which each I
  // should store its victims. The offsets vector is one element longer, giving
  // us the total size as the final element.

  mob::vector<System, size_t> offsets(1 + infected.size());
  thrust::inclusive_scan(victim_count.begin(), victim_count.end(),
                         offsets.begin() + 1);

  size_t total_infections = offsets.back();
  auto [infection_source, infection_victim] = output.grow(total_infections);

  // Select all the victims. This uses the same RNG state as our earlier
  // sampling, guaranteeing that we'll get the same number as we had allocated.
  //
  // This produces (victim, source) tuples for each infection.

  auto infection_victim_begin = infection_victim.begin();
  auto infection_source_begin = infection_source.begin();

  thrust::for_each(thrust::make_zip_iterator(infected.begin(), offsets.begin(),
                                             rngs.begin()),
                   thrust::make_zip_iterator(infected.end(),
                                             offsets.begin() + infected.size(),
                                             rngs.begin() + infected.size()),
                   thrust::make_zip_function(
                       [=] __host__ __device__(uint32_t i, size_t offset,
                                               mob::random_proxy<System> &rng) {
                         auto victims = victims_fn(i, rng);
                         auto victim_first = infection_victim_begin + offset;
                         auto victim_last = compat::copy(victims, victim_first);

                         auto source_first = infection_source_begin + offset;
                         auto source_last =
                             infection_source_begin +
                             (victim_last - infection_victim_begin);

                         compat::fill(source_first, source_last, i);
                       }));

  return total_infections;
}

template <typename System>
size_t homogeneous_infection_process(mob::parallel_random<System> &rngs,
                                     infection_list<System> &output,
                                     mob::ds::span<System, uint32_t> infected,
                                     mob::bitset_view<System> susceptible,
                                     double infection_probability) {
  // Unfortunately bernoulli() on a bitset is not as fast as we'd want it to
  // be. Materializing the bitset into a vector first and then sampling that
  // is much faster, especially given that the `to_vector` call can be
  // parallelized.
  auto susceptible_vector = susceptible.to_vector();
  ds::span susceptible_view(susceptible_vector);

  return infection_process<System>(
      rngs, output, infected,
      [=] __host__ __device__(uint32_t, mob::random_proxy<System> &rng) {
        return bernoulli(susceptible_view, rng, infection_probability);
      });
}

template <typename System>
size_t household_infection_process(
    mob::parallel_random<System> &rngs, infection_list<System> &output,
    ds::span<System, uint32_t> infected, mob::bitset_view<System> susceptible,
    ds::partition_view<System> partition,
    ds::span<System, double> infection_probability) {
  auto is_susceptible = [=] __host__ __device__(uint32_t i) {
    return susceptible.contains(i);
  };
  return infection_process<System>(
      rngs, output, infected,
      [=] __host__ __device__(uint32_t i, mob::random_proxy<System> &rng) {
        double p;
        if (infection_probability.size() == 1) {
          p = infection_probability[0];
        } else {
          p = infection_probability[partition.get_partition(i)];
        }

        // Whether we apply the bernoulli or the `is_susceptible` filter first
        // is equivalent, but the former is much faster since it avoids doing
        // any work in the vast majority of cases.
        return partition.neighbours(i) | bernoulli(rng, p) |
               compat::filter(is_susceptible);
      });
}

template <typename System>
mob::vector<System, uint32_t>
infection_victims(const infection_list<System> &infections) {
  // TODO: this does a copy, which we could avoided in cases where
  // `infections` was passed by move.
  mob::vector<System, uint32_t> victims = infections.victims;
  thrust::sort(victims.begin(), victims.end());
  auto end = thrust::unique(victims.begin(), victims.end());
  victims.erase(end, victims.end());
  return victims;
}

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

} // namespace mob

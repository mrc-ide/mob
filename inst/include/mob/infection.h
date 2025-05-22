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

  size_t size() const {
    return sources.size();
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
template <typename InputIt, typename OutputKeyIt, typename OutputValueIt>
std::pair<OutputKeyIt, OutputValueIt> run_lengths(InputIt first, InputIt last,
                                                  OutputKeyIt output_key,
                                                  OutputValueIt output_value) {
  return thrust::reduce_by_key(first, last,
                               thrust::constant_iterator<uint32_t>(1),
                               output_key, output_value);
}

/**
 * For each group of consecutive keys in the range [keys_first, keys_last) that
 * are equal, this function selects one of the corresponding values at random.
 *
 * The output_key and output_value output iterators must be (at least) as large
 * as the number of key groups. That size can be pre-determined using
 * `thrust::unique_count(keys_first, keys_last)`.
 */
template <typename System, typename KeyIt, typename ValueIt,
          typename OutputKeyIt, typename OutputValueIt>
std::pair<OutputKeyIt, OutputValueIt>
random_select_by_key(mob::parallel_random<System> &rngs, KeyIt keys_first,
                     KeyIt keys_last, ValueIt values_first,
                     OutputKeyIt output_key, OutputValueIt output_value) {
  size_t n = thrust::distance(keys_first, keys_last);
  mob::vector<System, size_t> offsets(n + 1);

  auto [output_key_last, offsets_last] = run_lengths(
      keys_first, keys_last, output_key, cuda::std::next(offsets.begin()));
  size_t k = thrust::distance(output_key, output_key_last);

  thrust::inclusive_scan(cuda::std::next(offsets.begin()), offsets_last,
                         cuda::std::next(offsets.begin()));

  thrust::transform(
      thrust::make_zip_iterator(offsets.begin(),
                                cuda::std::next(offsets.begin()), rngs.begin()),
      thrust::make_zip_iterator(cuda::std::prev(offsets_last), offsets_last,
                                cuda::std::next(rngs.begin(), k)),
      output_value,
      thrust::make_zip_function(
          [=] __host__ __device__(size_t lower_bound, size_t upper_bound,
                                  mob::parallel_random<System>::proxy &rng) {
            size_t i = random_bounded_int(rng, lower_bound, upper_bound);
            return values_first[i];
          }));

  return {output_key_last, output_value + k};
}

template <typename System>
infection_list<System>
infections_select(mob::parallel_random<System> &rngs,
                  const infection_list<System> &infections) {
  mob::vector<System, uint32_t> sources = infections.sources;
  mob::vector<System, uint32_t> victims = infections.victims;

  thrust::sort_by_key(victims.begin(), victims.end(), sources.begin());
  size_t n = thrust::unique_count(victims.begin(), victims.end());

  mob::vector<System, uint32_t> selected_sources(n);
  mob::vector<System, uint32_t> selected_victims(n);

  random_select_by_key(rngs, victims.begin(), victims.end(), sources.begin(),
                       selected_victims.begin(), selected_sources.begin());

  return {std::move(selected_sources), std::move(selected_victims)};
}

} // namespace mob

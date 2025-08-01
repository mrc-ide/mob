#pragma once

#include <mob/compat/algorithm.h>
#include <mob/ds/span.h>
#include <mob/random.h>
#include <mob/system.h>

#include <thrust/sequence.h>
#include <thrust/zip_function.h>

namespace mob {

template <typename System>
struct alias_table_view {
  mob::ds::span<System, const double> probabilities;
  mob::ds::span<System, const size_t> aliases;

  alias_table_view(mob::ds::span<System, const double> probabilities,
                   mob::ds::span<System, const size_t> aliases)
      : probabilities(probabilities), aliases(aliases) {}

  /**
   * Sample a single value from the alias table.
   */
  template <random_state R>
  __device__ __host__ size_t sample(R &rng) const {
    size_t i = mob::random_bounded_int(rng, probabilities.size());
    double x = dust::random::random_real<double>(rng);
    if (x < probabilities[i]) {
      return i;
    } else {
      return aliases[i];
    }
  }

  /**
   * Sample without replacement from the alias table.
   *
   * This uses a very naive rejection sampling. It keeps sampling values and
   * rejects ones them if they've already been sampled. Moreover it uses a
   * linear scan to determine whether the value has been drawn already.
   *
   * This makes this suitable only for values of k that are both small in
   * absolute terms and small with respect to N.
   */
  template <random_state R, typename InputIt>
  __device__ __host__ void sample_wor(R &rng, InputIt first,
                                      InputIt last) const {
    auto current = first;
    while (current != last) {
      size_t value = this->sample(rng);
      if (!mob::compat::contains(first, current, value)) {
        *current = value;
        ++current;
      }
    }
  }
};

/**
 * This type implements Walker's alias method for weighted random sampling. Once
 * the table is constructed, in O(n) time, individual samples can be drawn in
 * constant time, and independently.
 *
 * The algorithm for constructing the table comes from the description found in
 * the paper "Parallel Weighted Random Sampling" by Hübschle-Schneider et al.
 * Despite the name of the paper, we only use the sequential version of it,
 * found in Section 4.1.
 *
 * https://dl.acm.org/doi/pdf/10.1145/3549934
 *
 * There is a follow up paper "Weighted Random Sampling on GPUs" by
 * Lehmann et al. which describes a method to parallelize the construction. We
 * do not implement that yet.
 *
 * https://arxiv.org/pdf/2106.12270
 */
template <typename System>
struct alias_table {
  mob::vector<System, double> probabilities;
  mob::vector<System, size_t> aliases;

  template <cuda::std::ranges::random_access_range R>
    requires std::floating_point<cuda::std::ranges::range_value_t<R>>
  alias_table(const R &weights) {
    // We use thrust for convenience, but this is highly sequential code and
    // runs all in CPU-space. At the very end we move the aliases and
    // probabilities onto the device.
    thrust::host_vector<size_t> aliases(weights.size());
    thrust::host_vector<double> probabilities(weights.size());

    size_t n = weights.size();
    double total = thrust::reduce(weights.begin(), weights.end());
    double wn = total / n;

    size_t light = thrust::find_if(weights.begin(), weights.end(),
                                   [wn](double w) { return w <= wn; }) -
                   weights.begin();

    size_t current = thrust::find_if(weights.begin(), weights.end(),
                                     [wn](double w) { return w > wn; }) -
                     weights.begin();

    // All weights are equal
    if (current == n) {
      thrust::sequence(aliases.begin(), aliases.end());
      thrust::fill(probabilities.begin(), probabilities.end(), 1);
    } else {
      double residual = weights[current];
      while (current < n) {
        if (residual > wn) {
          // This is only reachable because of floating point inaccuracies.
          // It means that we ran out of light elements, while still having
          // heavy elements to deal with. In this case it is expected that
          // residual is very close to wn.
          if (light == n) {
            break;
          }
          // The current item is still too big. We need to take a light item
          // and transfer some weight to it.
          probabilities[light] = weights[light] / wn;
          aliases[light] = current;

          residual -= (wn - weights[light]);

          // Find the next light item.
          light = thrust::find_if(weights.begin() + light + 1, weights.end(),
                                  [wn](double w) { return w <= wn; }) -
                  weights.begin();
        } else {
          // The current item is now small enough to fit in the bucket.
          // We have (probably) overshot and need to alias the extra
          // probability to the next heavy item.

          probabilities[current] = residual / wn;

          size_t next =
              thrust::find_if(weights.begin() + current + 1, weights.end(),
                              [wn](double w) { return w > wn; }) -
              weights.begin();

          if (next < n) {
            aliases[current] = next;
            residual = weights[next] - (wn - residual);
          } else {
            // This should only really happen in cases where `residual == wn`,
            // ie. `probabilities[current] == 1`. Make the alias point to
            // itself just to be safe.
            aliases[current] = current;
          }

          // Skip to the new heavy element.
          current = next;
        }
      }

      // The loop above that terminate without traversing all the items, either
      // due to floating point approximations or because all remaining items in
      // the list have weight (Σw / n). In these cases we still have to populate
      // the rest of the list.
      while (current < n) {
        probabilities[current] = 1;
        aliases[current] = current;
        current = thrust::find_if(weights.begin() + current + 1, weights.end(),
                                  [wn](double w) { return w > wn; }) -
                  weights.begin();
      }
      while (light < n) {
        probabilities[light] = 1;
        aliases[light] = light;
        light = thrust::find_if(weights.begin() + light + 1, weights.end(),
                                [wn](double w) { return w <= wn; }) -
                weights.begin();
      }
    }

    this->probabilities = std::move(probabilities);
    this->aliases = std::move(aliases);
  }

  size_t size() const {
    return probabilities.size();
  }

  /**
   * Sample k values from the alias table, with replacement.
   */
  template <typename rng_state>
  mob::vector<System, size_t> sample(rng_state &rngs, size_t k) const {
    size_t n = probabilities.size();
    mob::vector<System, size_t> result(k);
    alias_table_view<System> table(probabilities, aliases);
    thrust::transform(
        rngs.begin(), rngs.begin() + k, result.begin(),
        [table] __host__ __device__(typename rng_state::proxy & rng) {
          return table.sample(rng);
        });

    return result;
  }

  // This returns a rows * k row-major matrix.
  template <typename rng_state>
  mob::vector<System, size_t> sample_wor(rng_state &rngs, size_t rows,
                                         size_t k) const {
    size_t n = probabilities.size();

    mob::vector<System, size_t> result(k * rows);
    mob::ds::span<System, size_t> result_view(result);

    alias_table_view<System> table(probabilities, aliases);
    thrust::for_each_n(
        thrust::make_zip_iterator(rngs.begin(),
                                  thrust::make_counting_iterator<size_t>(0)),
        rows,
        thrust::make_zip_function(
            [table, result_view,
             k] __host__ __device__(typename rng_state::proxy & rng, size_t i) {
              // TODO: use a stride on the counting iterator instead of
              // multiply. Not available on my version of Thrust yet.
              auto out = result_view.begin() + i * k;
              table.sample_wor(rng, out, out + k);
            }));
    return result;
  }

  // This returns a rows * maxk row-major matrix.
  template <typename rng_state>
  mob::vector<System, size_t>
  sample_wor_ragged_matrix(rng_state &rngs, mob::ds::span<System, size_t> ks,
                           size_t maxk) const {
    size_t n = probabilities.size();

    mob::vector<System, size_t> result(maxk * ks.size());
    mob::ds::span<System, size_t> result_view(result);

    alias_table_view<System> table(probabilities, aliases);

    thrust::for_each_n(
        thrust::make_zip_iterator(rngs.begin(),
                                  thrust::make_counting_iterator<size_t>(0),
                                  ks.begin()),
        ks.size(),
        thrust::make_zip_function(
            [table, result_view, maxk] __host__ __device__(
                typename rng_state::proxy & rng, size_t i, size_t k) {
              // TODO: use a stride on the counting iterator instead of
              // multiply. Not available on my version of Thrust yet.
              auto out = result_view.begin() + i * maxk;
              table.sample_wor(rng, out, out + k);
            }));

    return result;
  }
};

} // namespace mob

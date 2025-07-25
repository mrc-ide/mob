#pragma once
#include <mob/ds/span.h>

#include <algorithm>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>

namespace mob {
namespace ds {

template <typename System, typename T>
struct ragged_vector_view;

template <typename System, typename T>
struct ragged_vector {
public:
  friend ragged_vector_view<System, T>;

  ragged_vector() {
    // TODO: make the empty case not require an allocation.
    offsets_.push_back(0);
  }

  ragged_vector(mob::vector<System, size_t> offsets,
                mob::vector<System, T> values)
      : offsets_(std::move(offsets)), values_(std::move(values)) {
    offsets_.push_back(values_.size());
  }

  ds::span<System, const T> operator[](size_t i) const {
    return slice(i, i + 1);
  }

  ds::span<System, const T> slice(size_t i, size_t j) const {
    return {values_.data() + offsets_[i], values_.data() + offsets_[j]};
  }

  size_t size() const {
    return offsets_.size() - 1;
  }

  mob::vector<System, size_t> sizes() const {
    mob::vector<System, size_t> result(offsets_.size() - 1);
    thrust::adjacent_difference(offsets_.begin() + 1, offsets_.end(),
                                result.begin());
    return result;
  }

private:
  mob::vector<System, size_t> offsets_;
  mob::vector<System, T> values_;
};

template <typename System, typename T>
struct ragged_vector_view {
public:
  ragged_vector_view(const ragged_vector<System, T> &vector)
      : offsets_(vector.offsets_), values_(vector.values_) {}

  __host__ __device__ ds::span<System, const T> operator[](size_t i) const {
    return slice(i, i + 1);
  }

  __host__ __device__ ds::span<System, const T> slice(size_t i,
                                                      size_t j) const {
    if (i > j) {
      dust::utils::fatal_error("invalid slice indices");
    }
    if (j > size()) {
      dust::utils::fatal_error("out-of-bounds index");
    }
    return {values_.begin() + offsets_[i], values_.begin() + offsets_[j]};
  }

  __host__ __device__ size_t size() const {
    return offsets_.size() - 1;
  }

private:
  ds::span<System, const size_t> offsets_;
  ds::span<System, const T> values_;
};

/*
 * Creates a ragged_vector representing a multi-map.
 *
 * Given a list of key-value pairs, this returns a ragged_vector that
 * associates a list of values for each key. The keys must be dense with a known
 * upper-bound (capacity).
 */
template <typename System, typename T>
ragged_vector<System, T>
prepare_ragged_vector(size_t capacity, mob::vector<System, uint32_t> keys,
                      mob::vector<System, T> values) {
  if (keys.size() != values.size()) {
    throw std::logic_error("Mismatching key sizes");
  }

  thrust::sort_by_key(keys.begin(), keys.end(), values.begin());

  // For each segment, we need to know where it begins. Its end is
  // implicitly the start of the next one.
  //
  // There is probably a cleverer way than `lower_bound` to do this.
  //
  // From "Improved GPU Near Neighbours Performance for Multi-Agent
  // Simulations":
  // > When implemented with atomic counting sort this produces
  // > the PBM [ie. the offsets_ array] as a by-product of the neighbour
  // > data array sort.
  //
  // See also "Fast Fixed-Radius Nearest Neighbor Search on the GPU" by
  // Hoetzlein. It is light on details, but the slides suggest an "atomic
  // counting sort" as well.
  // https://ramakarl.com/pdfs/2014_Hoetzlein_FastFixedRadius_Neighbors.pdf
  //
  // On the other hand, Thrust's histogram example just does a binary search
  // like us (albeit a upper_bound instead of lower_bound):
  // https://github.com/NVIDIA/cccl/blob/8c1010a03c81fc2ca139f93ce0ce317339d73430/thrust/examples/histogram.cu#L85
  //
  // Also relevant: https://dl.acm.org/doi/pdf/10.1145/3472456.3472486
  // "Efficient GPU-Implementation for Integer Sorting Based on Histogram and
  // Prefix-Sums" by Kozakai et al.
  //
  mob::vector<System, size_t> offsets(capacity);
  thrust::counting_iterator<uint32_t> start(0);
  thrust::lower_bound(keys.begin(), keys.end(), start, start + capacity,
                      offsets.begin());

  return ragged_vector<System, uint32_t>(std::move(offsets), std::move(values));
}

/**
 * This represents a partition of the population into disjoint sets, eg. into
 * households. It allows quick look ups from one individual to all members in
 * the same subset.
 *
 * It is represented as two structures, once mapping from individuals to
 * subset index, and the other mapping from a subset index to a list of
 * members.
 *
 * Rather than store the latter as a vector<vector<int>>, we use a flat
 * vector<int> and separately store the offset to each subset. This makes it
 * easier to copy and reference the data from the GPU.
 */
template <typename System>
struct partition {
  ragged_vector<System, uint32_t> partitions_;
  mob::vector<System, uint32_t> members_;

  partition(size_t capacity, mob::vector<System, uint32_t> members)
      : members_(std::move(members)) {
    mob::vector<System, uint32_t> values(members_.size());
    thrust::sequence(values.begin(), values.end());

    partitions_ =
        prepare_ragged_vector<System>(capacity, members_, std::move(values));
  }

  size_t partitions_count() const {
    return partitions_.size();
  }

  size_t population_size() const {
    return members_.size();
  }

  // Get the subset index of an individual
  uint32_t get_partition(uint32_t i) const {
    return members_[i];
  }

  // Get a individual's neighbours, ie. all the individuals that are in the same
  // subset as it. This includes the given individual itself.
  ds::span<System, const uint32_t> neighbours(uint32_t i) const {
    return get_members(get_partition(i));
  }

  // Get the members of a given subset.
  ds::span<System, const uint32_t> get_members(uint32_t p) const {
    return partitions_[p];
  }

  // Get the number of individual in each subset of the population.
  mob::vector<System, size_t> sizes() const {
    return partitions_.sizes();
  }
};

template <typename System>
struct partition_view {
  ds::span<System, const uint32_t> members_;
  ragged_vector_view<System, uint32_t> partitions_;

  partition_view(const partition<System> &p)
      : members_(p.members_), partitions_(p.partitions_) {}

  __host__ __device__ size_t partitions_count() const {
    return partitions_.size();
  }

  __host__ __device__ size_t population_size() const {
    return members_.size();
  }

  __host__ __device__ uint32_t get_partition(uint32_t i) const {
    return members_[i];
  }

  __host__ __device__ ds::span<System, const uint32_t>
  get_members(uint32_t p) const {
    return partitions_[p];
  }

  __host__ __device__ ds::span<System, const uint32_t>
  neighbours(uint32_t i) const {
    return get_members(get_partition(i));
  }
};

} // namespace ds
} // namespace mob

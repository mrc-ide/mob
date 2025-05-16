#pragma once
#include <mob/ds/span.h>

#include <algorithm>
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

  ragged_vector(size_t capacity) : offsets_(capacity + 1) {}

  void assign(mob::vector<System, size_t> keys, mob::vector<System, T> values) {
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
    // like us (albeit a higher_bound instead of lower_bound):
    // https://github.com/NVIDIA/cccl/blob/8c1010a03c81fc2ca139f93ce0ce317339d73430/thrust/examples/histogram.cu#L85
    //
    thrust::counting_iterator<uint32_t> start(0);
    thrust::lower_bound(keys.begin(), keys.end(), start, start + size(),
                        offsets_.begin());

    offsets_.back() = values.size();
    data_ = std::move(values);
  }

  ds::span<System, const T> operator[](size_t i) const {
    return slice(i, i + 1);
  }

  ds::span<System, const T> slice(size_t i, size_t j) const {
    return {data_.data() + offsets_[i], data_.data() + offsets_[j]};
  }

  size_t size() const {
    return offsets_.size() - 1;
  }

private:
  mob::vector<System, T> data_;
  mob::vector<System, size_t> offsets_;
};

template <typename System, typename T>
struct ragged_vector_view {
public:
  ragged_vector_view(const ragged_vector<System, T> &vector)
      : data_(vector.data_), offsets_(vector.offsets_) {}

  __host__ __device__ ds::span<System, const T> operator[](size_t i) const {
    return slice(i, i + 1);
  }

  __host__ __device__ ds::span<System, const T> slice(size_t i,
                                                      size_t j) const {
    return {data_.begin() + offsets_[i], data_.begin() + offsets_[j]};
  }

  __host__ __device__ size_t size() const {
    return offsets_.size() - 1;
  }

private:
  ds::span<System, const T> data_;
  ds::span<System, const size_t> offsets_;
};

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

  partition(size_t capacity, thrust::host_vector<uint32_t> members)
      : partitions_(capacity), members_(std::move(members)) {

    mob::vector<System, uint32_t> values(members_.size());
    thrust::sequence(values.begin(), values.end());
    partitions_.assign(members_, values);
  }

  size_t partitions_count() const {
    return partitions_.size();
  }

  size_t population_size() const {
    return members_.size();
  }

  uint32_t get_partition(uint32_t i) const {
    return members_[i];
  }

  ds::span<System, const uint32_t> get_members(uint32_t p) const {
    return partitions_[p];
  }

  ds::span<System, const uint32_t> neighbours(uint32_t i) const {
    return get_members(get_partition(i));
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

#pragma once
#include <mob/ds/span.h>

#include <algorithm>
#include <vector>

namespace mob {
namespace ds {

/**
 * This represents a partition of the population into disjoint sets, eg. into
 * households. It allows quick look ups from one individual to all members in
 * the same subset.
 *
 * It is represented as two structures, once mapping from individuals to subset
 * index, and the other mapping from a subset index to a list of members.
 *
 * Rather than store the latter as a vector<vector<int>>, we use a flat
 * vector<int> and separately store the offset and length of each subset.
 * This makes it easier to copy and reference the data from the GPU.
 */

template <typename System>
struct partition {
  typename System::vector<uint32_t> members_;
  typename System::vector<uint32_t> partitions_data_;
  typename System::vector<uint32_t> partitions_offset_;
  typename System::vector<uint32_t> partitions_size_;

  partition(std::vector<uint32_t> members) {
    uint32_t count = *std::max_element(members.begin(), members.end()) + 1;

    // TODO: some of this can be parallelized. Maybe not worth it if only done
    // once.
    std::vector<std::vector<uint32_t>> partitions(count);
    for (uint32_t i = 0; i < members.size(); i++) {
      partitions[members[i]].push_back(i);
    }

    partitions_data_.reserve(members.size());
    partitions_size_.reserve(count);
    partitions_offset_.reserve(count);
    for (const std::vector<uint32_t> &p : partitions) {
      partitions_offset_.push_back(partitions_data_.size());
      partitions_size_.push_back(p.size());
      partitions_data_.insert(partitions_data_.end(), p.begin(), p.end());
    }

    members_ = std::move(members);
  }

  size_t partitions_count() const {
    return partitions_size_.size();
  }

  size_t population_size() const {
    return members_.size();
  }

  uint32_t get_partition(uint32_t i) const {
    return members_[i];
  }

  ds::span<System, const uint32_t> get_members(uint32_t p) const {
    return {partitions_data_.data() + partitions_offset_[p],
            partitions_size_[p]};
  }

  ds::span<System, const uint32_t> neighbours(uint32_t i) const {
    return get_members(get_partition(i));
  }
};

template <typename System>
struct partition_view {
  ds::span<System, const uint32_t> members_;
  ds::span<System, const uint32_t> partitions_data_;
  ds::span<System, const uint32_t> partitions_offset_;
  ds::span<System, const uint32_t> partitions_size_;

  partition_view(const partition<System> &p)
      : members_(p.members_), partitions_data_(p.partitions_data_),
        partitions_offset_(p.partitions_offset_),
        partitions_size_(p.partitions_size_) {}

  __host__ __device__ size_t partitions_count() const {
    return partitions_data_.size();
  }

  __host__ __device__ size_t population_size() const {
    return members_.size();
  }

  __host__ __device__ uint32_t get_partition(uint32_t i) const {
    return members_[i];
  }

  __host__ __device__ ds::span<System, const uint32_t>
  get_members(uint32_t p) const {
    return {partitions_data_.data() + partitions_offset_[p],
            partitions_size_[p]};
  }

  __host__ __device__ ds::span<System, const uint32_t>
  neighbours(uint32_t i) const {
    return get_members(get_partition(i));
  }
};

} // namespace ds
} // namespace mob

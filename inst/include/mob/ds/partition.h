#pragma once
#include <mob/ds/view.h>

#include <vector>

namespace mob {
namespace ds {

template <typename System>
struct partition {
  typename System::vector<uint32_t> members_;
  typename System::vector<uint32_t> partitions_data_;
  typename System::vector<uint32_t> partitions_offset_;
  typename System::vector<uint32_t> partitions_size_;

  partition(std::vector<uint32_t> members) : members_(std::move(members)) {
    uint32_t count = *std::max_element(members_.begin(), members_.end()) + 1;

    // TODO: some of this can be parallelized. Maybe not worth it if only done
    // once.
    std::vector<std::vector<uint32_t>> partitions(count);
    for (uint32_t i = 0; i < members_.size(); i++) {
      partitions[members_[i]].push_back(i);
    }

    partitions_data_.reserve(members_.size());
    partitions_size_.reserve(count);
    partitions_offset_.reserve(count);
    for (const std::vector<uint32_t> &p : partitions) {
      partitions_offset_.push_back(partitions_data_.size());
      partitions_size_.push_back(p.size());
      partitions_data_.insert(partitions_data_.end(), p.begin(), p.end());
    }
  }

  typename System::span<const uint32_t> get(uint32_t i) const {
    return {partitions_data_.data() + partitions_offset_[i],
            partitions_size_[i]};
  }

  typename System::span<const uint32_t> neighbours(uint32_t i) const {
    return get(members_[i]);
  }
};

template <typename System>
struct partition_view {
  typename System::span<const uint32_t> members_;
  typename System::span<const uint32_t> partitions_data_;
  typename System::span<const uint32_t> partitions_offset_;
  typename System::span<const uint32_t> partitions_size_;

  partition_view(const partition<System> &p)
      : members_(ds::view(p.members_)),
        partitions_data_(ds::view(p.partitions_data_)),
        partitions_offset_(ds::view(p.partitions_offset_)),
        partitions_size_(ds::view(p.partitions_size_)) {}

  __host__ __device__ typename System::span<const uint32_t>
  get(uint32_t i) const {
    return {partitions_data_.data() + partitions_offset_[i],
            partitions_size_[i]};
  }

  __host__ __device__ typename System::span<const uint32_t>
  neighbours(uint32_t i) const {
    return get(members_[i]);
  }
};

template <typename System>
partition_view<System> view(const partition<System> &p) {
  return p;
}

} // namespace ds
} // namespace mob

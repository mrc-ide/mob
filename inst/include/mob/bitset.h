#pragma once

#include <mob/bits.h>
#include <mob/ds/span.h>
#include <mob/system.h>

#include <thrust/transform_reduce.h>
#include <thrust/zip_function.h>

namespace mob {

template <typename System>
struct bitset_view;

template <typename System>
struct bitset {
  using word_type = uint32_t;
  static constexpr size_t num_bits = sizeof(word_type) * 8;

  bitset(size_t capacity)
      : capacity_(capacity), data_((capacity + num_bits - 1) / num_bits) {}

  void operator|=(bitset_view<System> other) {
    if (capacity_ != other.capacity_) {
      throw std::logic_error("bitsets have different sizes");
    }
    thrust::for_each(
        thrust::make_zip_iterator(data_.begin(), other.data_.begin()),
        thrust::make_zip_iterator(data_.end(), other.data_.end()),
        thrust::make_zip_function(
            [] __host__ __device__(word_type & left, word_type right) {
              left |= right;
            }));
  }

  void remove(bitset_view<System> other) {
    if (capacity_ != other.capacity_) {
      throw std::logic_error("bitsets have different sizes");
    }
    thrust::for_each(
        thrust::make_zip_iterator(data_.begin(), other.data_.begin()),
        thrust::make_zip_iterator(data_.end(), other.data_.end()),
        thrust::make_zip_function(
            [] __host__ __device__(word_type & left, word_type right) {
              left &= ~right;
            }));
  }

  void invert() {
    thrust::for_each(
        data_.begin(), data_.end(),
        [] __host__ __device__(word_type & value) { value = ~value; });

    // We need to keep the trailing bits clear
    if ((capacity_ % num_bits) != 0) {
      data_.back() &= (static_cast<word_type>(1) << (capacity_ % num_bits)) - 1;
    }
  }

  // Input values must be sorted.
  //
  // Algorithm works as follows:
  // - Transform each value into its bucket and mask
  // - Do a segmented reduction, using the bucket as the key and taking the
  //    bitwise union of the masks as a reduction
  // - For each reduction result update the corresponding bucket.
  void insert(mob::ds::span<System, uint32_t> values) {
    mob::vector<System, uint32_t> bucket(values.size());
    mob::vector<System, word_type> mask(values.size());

    thrust::transform(values.begin(), values.end(),
                      thrust::make_zip_iterator(bucket.begin(), mask.begin()),
                      [] __host__ __device__(uint32_t v) {
                        uint32_t bucket = v / num_bits;
                        uint32_t excess = v % num_bits;
                        word_type mask = (static_cast<word_type>(1) << excess);
                        return thrust::make_pair(bucket, mask);
                      });

    mob::vector<System, uint32_t> bucket_out(values.size());
    mob::vector<System, word_type> mask_out(values.size());

    // TODO: instead of std::equal_to, use a comparison that only looks at the
    // bucket, then it won't need to be pre-allocated.
    auto last = thrust::reduce_by_key(
        bucket.begin(), bucket.end(), mask.begin(), bucket_out.begin(),
        mask_out.begin(), cuda::std::equal_to<uint32_t>(),
        cuda::std::bit_or<word_type>());

    auto output = data_.begin();
    thrust::for_each(
        thrust::make_zip_iterator(bucket_out.begin(), mask_out.begin()),
        thrust::make_zip_iterator(last.first, last.second),
        thrust::make_zip_function(
            [output] __host__ __device__(uint32_t bucket, word_type mask) {
              // For some reason operator|= isn't __device__
              output[bucket] = output[bucket] | mask;
            }));
  }

  size_t capacity() const {
    return capacity_;
  }

  template <typename rng_state>
  void sample(rng_state &rngs, double p) {
    // TODO: we should be more clever than this and possibly use
    // fast_bernouilli, like individual does. The obvious way to do that is
    // fully sequential though, so we would want a compromise.
    thrust::for_each(
        thrust::make_zip_iterator(data_.begin(), rngs.begin()),
        thrust::make_zip_iterator(data_.end(), rngs.begin() + data_.size()),
        thrust::make_zip_function(
            [p] __host__ __device__(word_type & word,
                                    typename rng_state::proxy rng) {
              word_type mask = 0;
              for (size_t i = 0; i < num_bits; i++) {
                if (dust::random::random_real<double>(rng) < p) {
                  mask |= static_cast<word_type>(1) << i;
                }
              }
              word &= mask;
            }));
  }

private:
  mob::vector<System, word_type> data_;
  size_t capacity_;

  friend bitset_view<System>;
};

template <typename System>
struct bitset_view {
  using word_type = uint32_t;
  static constexpr size_t num_bits = sizeof(word_type) * 8;

  mob::ds::span<System, const word_type> data_;
  size_t capacity_;

  bitset_view(const bitset<System> &bitset)
      : data_(bitset.data_), capacity_(bitset.capacity_) {}

  size_t size() const {
    return thrust::transform_reduce(
        data_.begin(), data_.end(),
        [] __host__ __device__(word_type v) { return cuda::std::popcount(v); },
        0, cuda::std::plus<word_type>());
  }

  // TODO rewrite this for the CPU using the iterator
  mob::vector<System, uint32_t> to_vector() const {
    mob::vector<System, uint32_t> counts(data_.size() + 1);
    thrust::transform(
        data_.begin(), data_.end(), counts.begin(),
        [] __host__ __device__(word_type v) { return cuda::std::popcount(v); });
    thrust::exclusive_scan(counts.begin(), counts.end(), counts.begin());
    size_t size = counts.back();

    mob::vector<System, uint32_t> result(size);
    auto output = result.begin();
    thrust::for_each(
        thrust::make_zip_iterator(thrust::counting_iterator<size_t>(0),
                                  data_.begin(), counts.begin()),
        thrust::make_zip_iterator(
            thrust::counting_iterator<size_t>(data_.size()), data_.end(),
            counts.end() - 1),
        thrust::make_zip_function([output] __host__ __device__(uint32_t bucket,
                                                               word_type word,
                                                               size_t offset) {
          auto it = output + offset;
          // TODO: on the GPU it might just be faster to loop 0..num_bits and
          // test each value?
          while (word != 0) {
            uint32_t excess = cuda::std::countr_zero(word);
            word_type mask = (static_cast<word_type>(1) << excess);
            *it = (bucket * num_bits) | excess;
            it++;
            word &= ~mask;
          }
        }));

    return result;
  }

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = uint32_t;
    using reference = uint32_t;
    using pointer = const value_type *;
    using difference_type = ptrdiff_t;

    __host__ __device__ iterator() : parent_(nullptr), position_(0) {}
    __host__ __device__ iterator(const bitset_view<System> *parent,
                                 uint32_t position)
        : parent_(parent), position_(position) {}

    __host__ __device__ iterator &operator+=(size_t n) {
      if (position_ < parent_->capacity_ - n) {
        position_ = parent_->next_position(position_ + 1, n - 1);
      } else {
        position_ = parent_->capacity_;
      }
      return *this;
    }

    __host__ __device__ iterator &operator++() {
      if (position_ < parent_->capacity_ - 1) {
        position_ = parent_->next_position(position_ + 1, 0);
      } else {
        position_ = parent_->capacity_;
      }
      return *this;
    }

    __host__ __device__ iterator operator++(int) {
      auto t = *this;
      ++(*this);
      return t;
    }

    __host__ __device__ bool operator==(const iterator &other) const {
      return position_ == other.position_;
    }

    __host__ __device__ uint32_t operator*() const {
      return position_;
    }

  private:
    const bitset_view<System> *parent_;
    uint32_t position_;
  };

  static_assert(std::forward_iterator<iterator>);
  static_assert(std::sentinel_for<iterator, iterator>);

  __host__ __device__ iterator begin() const {
    return iterator{this, next_position(0, 0)};
  }

  __host__ __device__ iterator end() const {
    return iterator{this, static_cast<uint32_t>(capacity_)};
  }

  __host__ __device__ bool contains(uint32_t p) const {
    uint32_t bucket = p / num_bits;
    uint32_t excess = p % num_bits;
    word_type mask = (static_cast<word_type>(1) << excess);
    return data_[bucket] & mask;
  }

  __host__ __device__ uint32_t next_position(uint32_t p, uint32_t n) const {
    uint32_t bucket = p / num_bits;
    uint32_t excess = p % num_bits;

    word_type word = data_[bucket] >> excess;
    while (n >= static_cast<uint32_t>(cuda::std::popcount(word)) &&
           bucket + 1U < data_.size()) {
      n -= cuda::std::popcount(word);
      bucket += 1;
      word = data_[bucket];
      excess = 0;
    }

    uint8_t r = bits::select(word, n);
    return cuda::std::min<uint32_t>(bucket * num_bits + excess + r, capacity_);
  }
};

} // namespace mob

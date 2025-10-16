#pragma once

#include "conversion.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

template <typename System>
Rcpp::XPtr<mob::integer_vector<System>>
integer_vector_create(Rcpp::IntegerVector values) {
  auto v = fromRcppVector<System, uint32_t, ConvertIndex::No>(values);
  return make_externalptr<System>(std::move(v));
}

template <typename System>
Rcpp::XPtr<mob::double_vector<System>>
double_vector_create(Rcpp::NumericVector values) {
  auto v = fromRcppVector<System, double>(values);
  return make_externalptr<System>(std::move(v));
}

template <typename System>
Rcpp::IntegerVector vector_values(Rcpp::XPtr<mob::integer_vector<System>> v) {
  return asRcppVector<ConvertIndex::No>(*v);
}

template <typename System>
Rcpp::DoubleVector vector_values(Rcpp::XPtr<mob::double_vector<System>> v) {
  return asRcppVector(*v);
}

template <typename System, typename T>
void vector_scatter_bitset(Rcpp::XPtr<mob::vector<System, T>> vector,
                           Rcpp::XPtr<mob::bitset<System>> indices,
                           Rcpp::XPtr<mob::vector<System, T>> values) {
  mob::bitset_view bs(*indices);
  if (bs.size() != values->size()) {
    Rcpp::stop("argument sizes mismatch: %d != %d", bs.size(), values->size());
  }
  if (bs.capacity() != vector->size()) {
    Rcpp::stop("bitset capacity does not match target vector size: %d != %d",
               indices->capacity(), vector->size());
  }

  bs.scatter(vector->begin(), values->begin());
}

template <typename System, typename T>
void vector_scatter(Rcpp::XPtr<mob::vector<System, T>> vector,
                    Rcpp::XPtr<mob::integer_vector<System>> indices,
                    Rcpp::XPtr<mob::vector<System, T>> values) {
  if (indices->size() != values->size()) {
    Rcpp::stop("argument sizes mismatch: %d != %d", indices->size(),
               values->size());
  }
  checkIndices(*indices, vector->size());

  auto indices_it = thrust::make_transform_iterator(
      indices->begin(),
      [] __device__ __host__(uint32_t idx) { return idx - 1; });

  thrust::scatter(values->begin(), values->end(), indices_it, vector->begin());
}

template <typename System, typename T>
void vector_scatter_scalar(Rcpp::XPtr<mob::vector<System, T>> vector,
                           Rcpp::XPtr<mob::integer_vector<System>> indices,
                           T value) {
  checkIndices(*indices, vector->size());

  auto indices_it = thrust::make_transform_iterator(
      indices->begin(),
      [] __device__ __host__(uint32_t idx) { return idx - 1; });

  thrust::scatter(thrust::constant_iterator<T, size_t>(value, 0),
                  thrust::constant_iterator<T, size_t>(value, indices->size()),
                  indices_it, vector->begin());
}

template <typename System, typename T>
Rcpp::XPtr<mob::vector<System, T>>
vector_gather(Rcpp::XPtr<mob::vector<System, T>> vector,
              Rcpp::XPtr<mob::integer_vector<System>> indices) {
  checkIndices(*indices, vector->size());

  auto indices_it = thrust::make_transform_iterator(
      indices->begin(),
      [] __device__ __host__(uint32_t idx) { return idx - 1; });

  mob::vector<System, T> result(indices->size());
  thrust::gather(indices_it, indices_it + indices->size(), vector->begin(),
                 result.begin());

  return make_externalptr<System>(std::move(result));
}

template <typename System, typename Predicate>
  requires std::predicate<Predicate, uint32_t>
Rcpp::IntegerVector
integer_vector_match(Rcpp::XPtr<mob::integer_vector<System>> v,
                     Predicate pred) {
  mob::integer_vector<System> result(v->size());

  auto last = thrust::copy_if(thrust::counting_iterator<size_t>(0),
                              thrust::counting_iterator<size_t>(v->size()),
                              v->begin(), result.begin(), pred);

  result.erase(last, result.end());

  return asRcppVector<ConvertIndex::Yes>(std::move(result));
}

template <typename System, typename Predicate>
  requires std::predicate<Predicate, uint32_t>
Rcpp::XPtr<mob::bitset<System>>
integer_vector_match_as_bitset(Rcpp::XPtr<mob::integer_vector<System>> v,
                               Predicate pred) {
  size_t capacity = v->size();

  mob::bitset<System> result(capacity);

  using word_type = mob::bitset<System>::word_type;
  constexpr size_t num_bits = mob::bitset<System>::num_bits;
  auto buckets = result.data();
  auto input = v->data();

  thrust::for_each_n(
      thrust::make_zip_iterator(thrust::counting_iterator<size_t>(0),
                                buckets.begin()),
      buckets.size(),
      thrust::make_zip_function([input, pred, capacity] __host__ __device__(
                                    size_t i, word_type &out) {
        word_type word = 0;
        for (size_t j = 0; j < num_bits; j++) {
          size_t offset = num_bits * i + j;
          if (offset < capacity && pred(input[offset])) {
            word |= 1 << j;
          }
          out = word;
        }
      }));

  return make_externalptr<System>(std::move(result));
}

template <typename System>
Rcpp::IntegerVector
integer_vector_match_eq(Rcpp::XPtr<mob::integer_vector<System>> v,
                        size_t value) {
  return integer_vector_match<System>(
      v, [value] __device__ __host__(size_t i) { return i == value; });
}

template <typename System>
Rcpp::XPtr<mob::bitset<System>>
integer_vector_match_eq_as_bitset(Rcpp::XPtr<mob::integer_vector<System>> v,
                                  size_t value) {
  return integer_vector_match_as_bitset<System>(
      v, [value] __device__ __host__(size_t i) { return i == value; });
}

template <typename System>
Rcpp::IntegerVector
integer_vector_match_gt(Rcpp::XPtr<mob::integer_vector<System>> v,
                        size_t value) {
  return integer_vector_match<System>(
      v, [value] __device__ __host__(size_t i) { return i > value; });
}

template <typename System>
Rcpp::XPtr<mob::bitset<System>>
integer_vector_match_gt_as_bitset(Rcpp::XPtr<mob::integer_vector<System>> v,
                                  size_t value) {
  return integer_vector_match_as_bitset<System>(
      v, [value] __device__ __host__(size_t i) { return i > value; });
}

template <typename System>
void vector_add_scalar(Rcpp::XPtr<mob::integer_vector<System>> v,
                       int32_t delta) {
  thrust::for_each(
      v->begin(), v->end(),
      [delta] __host__ __device__(uint32_t &value) { value += delta; });
}

template <typename System>
void vector_add_scalar(Rcpp::XPtr<mob::double_vector<System>> v, double delta) {
  thrust::for_each(
      v->begin(), v->end(),
      [delta] __host__ __device__(double &value) { value += delta; });
}

template <typename System>
void vector_div_scalar(Rcpp::XPtr<mob::double_vector<System>> v,
                       double divisor) {
  thrust::for_each(
      v->begin(), v->end(),
      [divisor] __host__ __device__(double &value) { value /= divisor; });
}

template <typename System>
Rcpp::XPtr<mob::integer_vector<System>>
double_vector_lround(Rcpp::XPtr<mob::double_vector<System>> values) {
  mob::integer_vector<System> result(values->size());
  thrust::transform(values->begin(), values->end(), result.begin(),
                    [] __host__ __device__(double v) -> uint32_t {
                      return cuda::std::lround(v);
                    });

  return make_externalptr<System>(std::move(result));
}

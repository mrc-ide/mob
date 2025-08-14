#pragma once

#include "conversion.h"

#include <cuda/std/functional>
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

template <typename System, typename T>
Rcpp::XPtr<mob::vector<System, T>>
vector_clone(Rcpp::XPtr<mob::vector<System, T>> vector) {
  mob::vector<System, T> result(vector->size());
  thrust::copy(vector->begin(), vector->end(), result.begin());
  return make_externalptr<System>(std::move(result));
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
Rcpp::XPtr<mob::vector<System, T>>
vector_rep(Rcpp::XPtr<mob::vector<System, T>> vector, size_t n) {
  mob::ds::span<System, T> input(*vector);
  size_t width = input.size();

  mob::vector<System, T> result(width * n);
  thrust::tabulate(result.begin(), result.end(),
                   [width, input] __device__ __host__(size_t i) -> T {
                     return input[i % width];
                   });

  return make_externalptr<System>(std::move(result));
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

template <typename System, typename T, typename Predicate>
  requires std::predicate<Predicate, T>
Rcpp::IntegerVector vector_match(Rcpp::XPtr<mob::vector<System, T>> v,
                                 Predicate pred) {
  mob::integer_vector<System> result(v->size());

  auto last = thrust::copy_if(thrust::counting_iterator<size_t>(0),
                              thrust::counting_iterator<size_t>(v->size()),
                              v->begin(), result.begin(), pred);

  result.erase(last, result.end());

  return asRcppVector<ConvertIndex::Yes>(std::move(result));
}

template <typename System, typename T, typename Predicate>
  requires std::predicate<Predicate, T>
Rcpp::XPtr<mob::bitset<System>>
vector_match_as_bitset(Rcpp::XPtr<mob::vector<System, T>> v, Predicate pred) {
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

template <typename System, typename T>
Rcpp::IntegerVector vector_match_eq(Rcpp::XPtr<mob::vector<System, T>> v,
                                    T value) {
  return vector_match<System>(
      v, [value] __device__ __host__(T i) { return i == value; });
}

template <typename System, typename T>
Rcpp::XPtr<mob::bitset<System>>
vector_match_eq_as_bitset(Rcpp::XPtr<mob::vector<System, T>> v, T value) {
  return vector_match_as_bitset<System>(
      v, [value] __device__ __host__(T i) { return i == value; });
}

template <typename System, typename T>
Rcpp::IntegerVector vector_match_gt(Rcpp::XPtr<mob::vector<System, T>> v,
                                    T value) {
  return vector_match<System>(
      v, [value] __device__ __host__(T i) { return i > value; });
}

template <typename System, typename T>
Rcpp::XPtr<mob::bitset<System>>
vector_match_gt_as_bitset(Rcpp::XPtr<mob::vector<System, T>> v, T value) {
  return vector_match_as_bitset<System>(
      v, [value] __device__ __host__(T i) { return i > value; });
}

// Annoyingly, std::make_signed_t is not defined on floating points.
template <typename T>
struct signed_type;

template <>
struct signed_type<double> {
  using type = double;
};

template <>
struct signed_type<uint32_t> {
  using type = int32_t;
};

template <typename System, typename T, typename F>
  requires std::regular_invocable<F, T> &&
           std::convertible_to<std::invoke_result_t<F, T>, T>
void vector_unary_operator(Rcpp::XPtr<mob::vector<System, T>> v, F f) {
  thrust::for_each(v->begin(), v->end(),
                   [f] __host__ __device__(T & value) { value = f(value); });
}

template <typename System, typename T1, typename T2, typename F>
  requires std::regular_invocable<F, T1, T2> &&
           std::convertible_to<std::invoke_result_t<F, T1, T2>, T1>
void vector_binary_operator(Rcpp::XPtr<mob::vector<System, T1>> lhs,
                            Rcpp::XPtr<mob::vector<System, T2>> rhs, F f) {
  if (lhs->size() != rhs->size()) {
    Rcpp::stop("argument sizes mismatch: %d != %d", lhs->size(), rhs->size());
  }
  thrust::for_each(thrust::make_zip_iterator(lhs->begin(), rhs->begin()),
                   thrust::make_zip_iterator(lhs->end(), rhs->end()),
                   thrust::make_zip_function(
                       [f] __host__ __device__(T1 & left, const T2 &right) {
                         left = f(left, right);
                       }));
}

template <typename System, typename T>
void vector_add_scalar(Rcpp::XPtr<mob::vector<System, T>> v,
                       typename signed_type<T>::type addend) {
  vector_unary_operator<System>(
      v, [addend] __host__ __device__(T value) { return value + addend; });
}

template <typename System, typename T>
void vector_mul_scalar(Rcpp::XPtr<mob::vector<System, T>> v,
                       typename signed_type<T>::type factor) {
  vector_unary_operator<System>(
      v, [factor] __host__ __device__(T value) { return value * factor; });
}

template <typename System>
void vector_div_scalar(Rcpp::XPtr<mob::double_vector<System>> v,
                       double divisor) {
  vector_unary_operator<System>(v, [divisor] __host__ __device__(double value) {
    return value / divisor;
  });
}

template <typename System>
void vector_exp(Rcpp::XPtr<mob::double_vector<System>> v) {
  vector_unary_operator<System>(v, [] __host__ __device__(double value) {
    return cuda::std::exp(value);
  });
}

template <typename System>
void vector_reciprocal(Rcpp::XPtr<mob::double_vector<System>> v) {
  vector_unary_operator<System>(
      v, [] __host__ __device__(double value) { return 1 / value; });
}

template <typename System, typename T>
void vector_neg(Rcpp::XPtr<mob::vector<System, T>> v) {
  vector_unary_operator<System>(v, cuda::std::negate<T>());
}

template <typename System, typename T, typename U>
void vector_add(Rcpp::XPtr<mob::vector<System, T>> lhs,
                Rcpp::XPtr<mob::vector<System, U>> rhs) {
  vector_binary_operator<System>(lhs, rhs, cuda::std::plus<T>{});
}

template <typename System, typename T, typename U>
void vector_mul(Rcpp::XPtr<mob::vector<System, T>> lhs,
                Rcpp::XPtr<mob::vector<System, U>> rhs) {
  vector_binary_operator<System>(lhs, rhs, cuda::std::multiplies<T>{});
}

template <typename System, typename T, typename U>
void vector_div(Rcpp::XPtr<mob::vector<System, T>> lhs,
                Rcpp::XPtr<mob::vector<System, U>> rhs) {
  vector_binary_operator<System>(lhs, rhs, cuda::std::divides<T>{});
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

template <typename System>
Rcpp::XPtr<mob::double_vector<System>>
integer_vector_to_double(Rcpp::XPtr<mob::integer_vector<System>> vector) {
  mob::double_vector<System> result(vector->size());
  thrust::copy(vector->begin(), vector->end(), result.begin());
  return make_externalptr<System>(std::move(result));
}

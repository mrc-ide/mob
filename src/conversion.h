#pragma once

#include <mob/bitset.h>
#include <mob/ds/span.h>
#include <mob/system.h>
#include <mob/vector.h>

#include <Rcpp.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

enum class ConvertIndex {
  No,
  Yes,
};

template <ConvertIndex value>
struct convert_index_tag {};

template <typename T>
struct type_tag {};

// Rcpp supports converting a device_vector<T> natively already, but it does so
// by iterating over the elements and copying them one by one. The latency of
// each copy is very large, making the overall operation very slow.
//
// thrust::copy issues a single copy operation instead. If necessary, it will
// apply a type conversion which can execute in parallel device-side.
template <ConvertIndex convert, typename T>
  requires std::integral<cuda::std::ranges::range_value_t<T>>
Rcpp::IntegerVector asRcppVector(T &&data) {
  return asRcppVector(std::forward<T>(data), convert_index_tag<convert>{});
}

template <typename T>
Rcpp::IntegerVector asRcppVector(const T &data,
                                 convert_index_tag<ConvertIndex::No>) {
  Rcpp::IntegerVector v(data.size());
  thrust::copy(data.begin(), data.end(), v.begin());
  return v;
}

template <typename T>
Rcpp::IntegerVector asRcppVector(const thrust::host_vector<T> &data,
                                 convert_index_tag<ConvertIndex::Yes>) {
  Rcpp::IntegerVector result(data.size());
  thrust::transform(data.begin(), data.end(), result.begin(),
                    [] __host__(T i) { return i + 1; });
  return result;
}

// TODO:
// Relax this so it accepts more ranges, not just a host_vector.
//
// Also add an overload for thrust::device_vector&& that does in-place
// transform.
template <typename T>
Rcpp::IntegerVector asRcppVector(const thrust::device_vector<T> &data,
                                 convert_index_tag<ConvertIndex::Yes>) {
  thrust::device_vector<T> converted(data.size());
  thrust::transform(data.begin(), data.end(), converted.begin(),
                    [] __device__(T i) { return i + 1; });
  return asRcppVector<ConvertIndex::No>(std::move(converted));
}

template <typename T>
  requires std::floating_point<cuda::std::ranges::range_value_t<T>>
Rcpp::NumericVector asRcppVector(T &&data) {
  Rcpp::NumericVector v(data.size());
  thrust::copy(data.begin(), data.end(), v.begin());
  return v;
}

template <typename System, typename T>
  requires std::floating_point<T>
mob::vector<System, T> fromRcppVector(Rcpp::NumericVector data) {
  return mob::vector<System, T>(data.begin(), data.end());
}

template <typename System, typename T, ConvertIndex convert, typename Input>
mob::vector<System, T> fromRcppVector(Input data) {
  return fromRcppVector_impl(data, type_tag<System>{}, type_tag<T>{},
                             convert_index_tag<convert>{});
}

template <typename System, typename T>
  requires std::floating_point<T>
mob::vector<System, T>
fromRcppVector_impl(Rcpp::NumericVector data, type_tag<System>, type_tag<T>,
                    convert_index_tag<ConvertIndex::No>) {
  return mob::vector<System, T>(data.begin(), data.end());
}

template <typename System, typename T>
  requires std::integral<T>
mob::vector<System, T>
fromRcppVector_impl(Rcpp::IntegerVector data, type_tag<System>, type_tag<T>,
                    convert_index_tag<ConvertIndex::No>) {
  return mob::vector<System, T>(data.begin(), data.end());
}

template <typename T>
  requires std::integral<T>
thrust::host_vector<T>
fromRcppVector_impl(Rcpp::IntegerVector data, type_tag<mob::system::host>,
                    type_tag<T>, convert_index_tag<ConvertIndex::Yes>) {
  thrust::host_vector<T> result(data.size());
  thrust::transform(data.begin(), data.end(), result.begin(),
                    [](auto i) { return i - 1; });
  return result;
}

template <typename T>
  requires std::integral<T>
thrust::device_vector<T>
fromRcppVector_impl(Rcpp::IntegerVector data, type_tag<mob::system::device>,
                    type_tag<T>, convert_index_tag<ConvertIndex::Yes>) {
  thrust::device_vector<T> result(data.size());

  // This needs to be done as two separate steps. If we did a naive transform
  // it would run host-side and write values one at a time.
  thrust::copy(data.begin(), data.end(), result.begin());
  thrust::for_each(result.begin(), result.end(),
                   [] __device__(T & v) { v -= 1; });

  return result;
}

// These are assumed to be 1-indexed indices
template <typename Range>
  requires cuda::std::ranges::input_range<Range> &&
           std::integral<cuda::std::ranges::range_value_t<Range>>
void checkIndices(const Range &indices, size_t length) {
  auto [min, max] = thrust::minmax_element(indices.begin(), indices.end());
  if (min != indices.end() && size_t(*min) < 1) {
    Rcpp::stop("index out-of-bound: %d (size: %d)", size_t(*min), length);
  }
  if (max != indices.end() && size_t(*max) > length) {
    Rcpp::stop("index out-of-bound: %d (size: %d)", size_t(*max), length);
  }
}

template <typename System>
Rcpp::XPtr<mob::integer_vector<System>>
make_externalptr(mob::integer_vector<System> &&v) {
  Rcpp::XPtr ptr(new mob::integer_vector<System>(std::move(v)));
  ptr.attr("class") = "integer_vector";
  return ptr;
}

template <typename System>
Rcpp::XPtr<mob::double_vector<System>>
make_externalptr(mob::double_vector<System> &&v) {
  Rcpp::XPtr ptr(new mob::double_vector<System>(std::move(v)));
  ptr.attr("class") = "double_vector";
  return ptr;
}

template <typename System>
Rcpp::XPtr<mob::bitset<System>> make_externalptr(mob::bitset<System> &&v) {
  Rcpp::XPtr ptr(new mob::bitset<System>(std::move(v)));
  ptr.attr("class") = "bitset";
  return ptr;
}

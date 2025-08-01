#pragma once

#include <Rcpp.h>
#include <thrust/host_vector.h>

// Rcpp supports converting a device_vector<T> natively already, but it does so
// by iterating over the elements and copying them one by one. The latency of
// each copy is very large, making the overall operation very slow.
//
// thrust::copy issues a single copy operation instead. If necessary, it will
// apply a type conversion which can execute in parallel device-side.
template <typename T>
  requires std::integral<cuda::std::ranges::range_value_t<T>>
Rcpp::IntegerVector asRcppVector(T &&data) {
  Rcpp::IntegerVector v(data.size());
  thrust::copy(data.begin(), data.end(), v.begin());
  return v;
}

template <typename T>
  requires std::floating_point<cuda::std::ranges::range_value_t<T>>
Rcpp::NumericVector asRcppVector(T &&data) {
  Rcpp::NumericVector v(data.size());
  thrust::copy(data.begin(), data.end(), v.begin());
  return v;
}

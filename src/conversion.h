#pragma once

#include <Rcpp.h>
#include <thrust/host_vector.h>

// Rcpp supports converting a device_vector<T> natively already, but it does
// so by iterating over the elements and copying them one by one. The latency
// of each copy is very large making the overall operation very slow. It is
// much faster to do a single copy into a host_vector<T>, and then let Rcpp
// wrap that entirely within CPU memory.
//
// An even faster option would be to copy directly from device memory to R
// memory, in cases where the underlying element type matches.
//
// TODO: use thrust::copy, which already takes care of these details. If a type
// conversion is needed, it can do it in parallel, device-side.
template <typename T>
SEXP asRcppVector(T &&data) {
  using value_type = typename std::remove_cvref_t<T>::value_type;
  return Rcpp::wrap(thrust::host_vector<value_type>{std::forward<T>(data)});
}

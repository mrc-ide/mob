#pragma once

#include "conversion.h"
#include <mob/ds/partition.h>
#include <mob/random.h>

template <typename System>
Rcpp::XPtr<mob::ds::partition<System>>
partition_create_wrapper(size_t capacity, Rcpp::IntegerVector population) {
  if (std::ranges::any_of(population, [=](auto i) { return i >= capacity; })) {
    Rcpp::stop("out-of-range population");
  }

  mob::vector<System, uint32_t> data(population.begin(), population.end());
  return Rcpp::XPtr(new mob::ds::partition<System>(capacity, std::move(data)));
}

template <typename System>
Rcpp::IntegerVector
partition_sizes_wrapper(Rcpp::XPtr<mob::ds::partition<System>> p) {
  return asRcppVector(p->sizes());
}

template <typename System>
Rcpp::XPtr<mob::ds::ragged_vector<System, uint32_t>>
ragged_vector_create_wrapper(Rcpp::List values) {
  // Using a high-level Rcpp wrapper such as NumericVector or IntegerVector to
  // access the entries of the list causes Rcpp to preserve each object from
  // garbage collection, which tends to be really quite slow.
  //
  // We don't actually need to protect each element while accessing them since
  // they are referenced by the List, which is itself protected. This is why we
  // use bare SEXP to access the entries.
  //
  // We could probably use Vector<INTSXP, NoProtectStorage> as a middle ground,
  // but I have some doubts about what happens when an implicit conversion from
  // NumericVector happens (is the casted vector protected by anything?).
  size_t total_size = 0;
  for (int i = 0; i < values.size(); i++) {
    SEXP v = values[i];
    if (TYPEOF(v) != INTSXP) {
      // TODO: support NumericVector as well. Obviously this loop is easy
      // (Rf_length works either way), but we would need the next loop below to
      // pay special attention and do the right conversion.
      Rcpp::stop("Bad ragged vector argument, must be integer");
    }
    total_size += Rf_length(v);
  }

  // Inserting directly into device vectors would create lots of small
  // cross-device memory copies, which has terrible latency. Instead we create
  // these on the host and copy it all at once at the end.
  mob::vector<mob::system::host, uint32_t> offsets;
  mob::vector<mob::system::host, uint32_t> data;
  offsets.reserve(values.size());
  data.reserve(total_size);

  for (int i = 0; i < values.size(); i++) {
    SEXP v = values[i];
    offsets.push_back(data.size());
    data.insert(data.end(), INTEGER(v), INTEGER(v) + Rf_length(v));
  }

  return Rcpp::XPtr(new mob::ds::ragged_vector<System, uint32_t>(
      std::move(offsets), std::move(data)));
}

template <typename System>
Rcpp::IntegerVector ragged_vector_get_wrapper(
    Rcpp::XPtr<mob::ds::ragged_vector<System, uint32_t>> v, size_t i) {
  return asRcppVector((*v)[i]);
}

template <typename System>
Rcpp::IntegerVector ragged_vector_random_select_wrapper(
    Rcpp::XPtr<mob::parallel_random<System>> rngs,
    Rcpp::XPtr<mob::ds::ragged_vector<System, uint32_t>> data) {
  mob::ds::ragged_vector_view<System, uint32_t> data_view = *data;
  mob::vector<System, uint32_t> result(data->size());

  // TODO: it would be nice if ragged_vector was a range itself that yielded
  // slices.
  thrust::counting_iterator<size_t> index(0);
  thrust::transform(
      thrust::make_zip_iterator(index, rngs->begin()),
      thrust::make_zip_iterator(index + data->size(),
                                rngs->begin() + data->size()),
      result.begin(),
      thrust::make_zip_function([data_view] __host__ __device__(
                                    size_t i, mob::random_proxy<System> &rng) {
        return mob::random_select(rng, data_view[i]);
      }));

  return asRcppVector(result);
}

#pragma once

#include "conversion.h"
#include <mob/ds/partition.h>
#include <mob/random.h>

template <typename System>
Rcpp::XPtr<mob::ds::partition<System>>
partition_create_wrapper(size_t capacity, std::vector<uint32_t> population) {
  if (std::ranges::any_of(population, [=](auto i) { return i >= capacity; })) {
    Rcpp::stop("out-of-range population");
  }
  return Rcpp::XPtr(
      new mob::ds::partition<System>(capacity, std::move(population)));
}

template <typename System>
Rcpp::IntegerVector
partition_sizes_wrapper(Rcpp::XPtr<mob::ds::partition<System>> p) {
  return asRcppVector(p->sizes());
}

template <typename System>
Rcpp::XPtr<mob::ds::ragged_vector<System, uint32_t>>
ragged_vector_create_wrapper(Rcpp::List values) {
  size_t total_size = 0;
  for (int i = 0; i < values.size(); i++) {
    Rcpp::IntegerVector v = values[i];
    total_size += v.size();
  }

  // Creating the ragged vector from an R list of vector tends to be lots of
  // small memory cross-device copies, which has terrible latency. Instead we
  // create these on the host and copy it all at once at the end.
  mob::vector<mob::system::host, uint32_t> offsets;
  mob::vector<mob::system::host, uint32_t> data;
  offsets.reserve(values.size());
  data.reserve(total_size);

  for (int i = 0; i < values.size(); i++) {
    Rcpp::IntegerVector v = values[i];
    offsets.push_back(data.size());
    data.insert(data.end(), v.cbegin(), v.cend());
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

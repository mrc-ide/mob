#pragma once

#include "conversion.h"
#include <mob/ds/partition.h>

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

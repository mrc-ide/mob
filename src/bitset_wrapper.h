#pragma once

#include "conversion.h"
#include <mob/bitset.h>
#include <mob/parallel_random.h>

#include <Rcpp.h>

template <typename System>
Rcpp::XPtr<mob::bitset<System>> bitset_create(size_t capacity) {
  return make_externalptr<System>(mob::bitset<System>(capacity));
}

template <typename System>
Rcpp::XPtr<mob::bitset<System>>
bitset_clone(Rcpp::XPtr<mob::bitset<System>> ptr) {
  return make_externalptr<System>(mob::bitset<System>(*ptr));
}

template <typename System>
size_t bitset_size(Rcpp::XPtr<mob::bitset<System>> ptr) {
  return mob::bitset_view(*ptr).size();
}

template <typename System>
void bitset_or(Rcpp::XPtr<mob::bitset<System>> left,
               Rcpp::XPtr<mob::bitset<System>> right) {
  *left |= *right;
}

template <typename System>
bool bitset_equal(Rcpp::XPtr<mob::bitset<System>> left,
                  Rcpp::XPtr<mob::bitset<System>> right) {
  return *left == *right;
}

template <typename System>
void bitset_remove(Rcpp::XPtr<mob::bitset<System>> left,
                   Rcpp::XPtr<mob::bitset<System>> right) {
  (*left).remove(*right);
}

template <typename System>
void bitset_invert(Rcpp::XPtr<mob::bitset<System>> ptr) {
  (*ptr).invert();
}

template <typename System>
void bitset_insert(Rcpp::XPtr<mob::bitset<System>> ptr,
                   Rcpp::IntegerVector values) {
  if (!std::is_sorted(values.begin(), values.end())) {
    Rcpp::stop("values must be sorted before insertion");
  }
  checkIndices(values, ptr->capacity());
  (*ptr).insert(fromRcppVector<System, uint32_t, ConvertIndex::Yes>(values));
}

template <typename System>
void bitset_sample(Rcpp::XPtr<mob::bitset<System>> ptr,
                   Rcpp::XPtr<mob::parallel_random<System>> rngs, double p) {
  (*ptr).sample(*rngs, p);
}

template <typename System>
void bitset_choose(Rcpp::XPtr<mob::bitset<System>> ptr,
                   Rcpp::XPtr<mob::parallel_random<System>> rngs, size_t k) {
  (*ptr).choose(*rngs, k);
}

template <typename System>
Rcpp::IntegerVector bitset_to_vector(Rcpp::XPtr<mob::bitset<System>> ptr) {
  return asRcppVector<ConvertIndex::Yes>(mob::bitset_view(*ptr).to_vector());
}

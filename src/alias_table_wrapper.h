#pragma once

#include "conversion.h"
#include <mob/alias_table.h>
#include <mob/parallel_random.h>

#include <Rcpp.h>

template <typename System>
Rcpp::XPtr<mob::alias_table<System>>
alias_table_create_wrapper(Rcpp::DoubleVector weights) {
  return Rcpp::XPtr(new mob::alias_table<System>(weights));
}

template <typename System>
Rcpp::IntegerVector
alias_table_sample_wrapper(Rcpp::XPtr<mob::alias_table<System>> table,
                           Rcpp::XPtr<mob::parallel_random<System>> rngs,
                           size_t k) {
  if (rngs->size() < k) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), k);
  }

  auto result = table->sample(*rngs, k);
  return asRcppVector<ConvertIndex::Yes>(result);
}

template <typename System>
Rcpp::IntegerMatrix
alias_table_sample_wor_wrapper(Rcpp::XPtr<mob::alias_table<System>> table,
                               Rcpp::XPtr<mob::parallel_random<System>> rngs,
                               size_t rows, size_t k) {
  if (rngs->size() < rows) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), rows);
  }
  if (k > table->size()) {
    Rcpp::stop("k > table->size(): %d > %d", k, table->size());
  }

  auto result = table->sample_wor(*rngs, rows, k);

  // TODO: change sample_wor to a column-major output so we don't have to
  // transpose here. Might even help with coalesced memory access.
  Rcpp::IntegerMatrix out(k, rows);
  thrust::copy(result.begin(), result.end(), out.begin());
  return transpose(out);
}

template <typename System>
Rcpp::IntegerMatrix alias_table_sample_wor_ragged_matrix_wrapper(
    Rcpp::XPtr<mob::alias_table<System>> table,
    Rcpp::XPtr<mob::parallel_random<System>> rngs, Rcpp::IntegerVector ks,
    size_t maxk) {
  if (rngs->size() < static_cast<size_t>(ks.size())) {
    Rcpp::stop("RNG state is too small: %d < %d", rngs->size(), ks.size());
  }
  for (size_t k : ks) {
    if (k > maxk) {
      Rcpp::stop("k > maxk: %d > %d", k, maxk);
    }
    if (k > table->size()) {
      Rcpp::stop("k > table->size(): %d > %d", k, table->size());
    }
  }

  auto ks_v = fromRcppVector<System, size_t, ConvertIndex::No>(ks);
  auto result = table->sample_wor_ragged_matrix(*rngs, ks_v, maxk);

  // TODO: change sample_wor_ragged_matrix to a column-major output so we don't
  // have to transpose here. Might even help with coalesced memory access.
  Rcpp::IntegerMatrix out(maxk, ks.size());
  thrust::copy(result.begin(), result.end(), out.begin());
  return transpose(out);
}

template <typename System>
Rcpp::DataFrame
alias_table_values_wrapper(Rcpp::XPtr<mob::alias_table<System>> table) {
  return Rcpp::DataFrame::create(
      Rcpp::Named("probability") = asRcppVector(table->probabilities),
      Rcpp::Named("alias") = asRcppVector<ConvertIndex::Yes>(table->aliases));
}

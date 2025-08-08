// This file is included by the auto-generated RcppExports.cpp. It is
// effectively the entrypoint to the C++ parts of the package.
//
// It mostly acts as concrete instantiations of the templated functions
// found in the `*_wrapper.h` files. These are currently a bit verbose,
// since we need to define two new functions each time that can be wrapped
// by Rcpp.
//
// Soon Rcpp might have better built-in support for this kind of thing:
// https://github.com/RcppCore/Rcpp/issues/1368

#pragma once

// Keep this first. Both dust and CCCL try to define __host__/__device__ if not
// using nvcc, but at least CCCL does it only if not defined already. If we
// include CCCL first and then dust, we end up with loads of warnings about
// redefinition.
//
// clang-format: off
#include <dust/random/cuda_compatibility.hpp>
// clang-format: on

#include "alias_table_wrapper.h"
#include "bitset_wrapper.h"
#include "infection_wrapper.h"
#include "partition_wrapper.h"
#include "random_wrapper.h"
#include "sample_wrapper.h"
#include "vector_wrapper.h"
#include <mob/system.h>

#ifdef __NVCC__

// [[Rcpp::export]]
Rcpp::XPtr<mob::device_random>
random_create_device(size_t size,
                     Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return random_create_wrapper<mob::system::device>(size, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::device>>
random_uniform_device(Rcpp::XPtr<mob::device_random> rngs, size_t n, double min,
                      double max) {
  return random_uniform_wrapper<mob::system::device>(rngs, n, min, max);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::device>>
random_poisson_device(Rcpp::XPtr<mob::device_random> rngs, size_t n,
                      double lambda) {
  return random_poisson_wrapper<mob::system::device>(rngs, n, lambda);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::device>>
random_binomial_device(Rcpp::XPtr<mob::device_random> rngs, size_t n,
                       size_t size, double prob) {
  return random_binomial_wrapper<mob::system::device>(rngs, n, size, prob);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::device>>
random_gamma_device(Rcpp::XPtr<mob::device_random> rngs, size_t n, double shape,
                    double scale) {
  return random_gamma<mob::system::device>(rngs, n, shape, scale);
}

// [[Rcpp::export]]
Rcpp::NumericVector bernoulli_sampler_device(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return bernoulli_sampler_wrapper<mob::system::device>(data, p, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::device>>
infection_list_create_device() {
  return infection_list_create_wrapper<mob::system::device>();
}

// [[Rcpp::export]]
size_t homogeneous_infection_process_device(
    Rcpp::XPtr<mob::device_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::device>> output,
    Rcpp::XPtr<mob::bitset<mob::system::device>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::device>> infected,
    double infection_probability) {
  return homogeneous_infection_process_wrapper<mob::system::device>(
      rngs, output, susceptible, infected, infection_probability);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::partition<mob::system::device>> partition_create_device(
    size_t capacity,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> population) {
  return partition_create_wrapper<mob::system::device>(capacity, population);
}

// [[Rcpp::export]]
Rcpp::IntegerVector
partition_sizes_device(Rcpp::XPtr<mob::ds::partition<mob::system::device>> p) {
  return partition_sizes_wrapper<mob::system::device>(p);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::ragged_vector<mob::system::device, uint32_t>>
ragged_vector_create_device(Rcpp::List values) {
  return ragged_vector_create_wrapper<mob::system::device>(values);
}

// [[Rcpp::export]]
Rcpp::IntegerVector ragged_vector_get_device(
    Rcpp::XPtr<mob::ds::ragged_vector<mob::system::device, uint32_t>> v,
    size_t i) {
  return ragged_vector_get_wrapper<mob::system::device>(v, i);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::device>>
ragged_vector_random_select_device(
    Rcpp::XPtr<mob::parallel_random<mob::system::device>> rngs,
    Rcpp::XPtr<mob::ds::ragged_vector<mob::system::device, uint32_t>> data) {
  return ragged_vector_random_select_wrapper<mob::system::device>(rngs, data);
}

// [[Rcpp::export]]
size_t household_infection_process_device(
    Rcpp::XPtr<mob::device_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::device>> output,
    Rcpp::XPtr<mob::bitset<mob::system::device>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::device>> infected,
    Rcpp::XPtr<mob::ds::partition<mob::system::device>> households,
    Rcpp::DoubleVector infection_probability) {
  return household_infection_process_wrapper<mob::system::device>(
      rngs, output, susceptible, infected, households, infection_probability);
}

// [[Rcpp::export]]
size_t spatial_infection_naive_device(
    Rcpp::XPtr<mob::device_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::device>> output,
    Rcpp::XPtr<mob::bitset<mob::system::device>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::device>> infected,
    Rcpp::NumericVector x, Rcpp::NumericVector y, double base, double k) {
  return spatial_infection_naive_wrapper(rngs, output, susceptible, infected, x,
                                         y, base, k);
}

// [[Rcpp::export]]
size_t spatial_infection_sieve_device(
    Rcpp::XPtr<mob::device_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::device>> output,
    Rcpp::XPtr<mob::bitset<mob::system::device>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::device>> infected,
    Rcpp::NumericVector x, Rcpp::NumericVector y, double base, double k) {
  return spatial_infection_sieve_wrapper(rngs, output, susceptible, infected, x,
                                         y, base, k);
}

// [[Rcpp::export]]
Rcpp::IntegerVector spatial_infection_hybrid_device(
    Rcpp::XPtr<mob::device_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::device>> output,
    Rcpp::XPtr<mob::bitset<mob::system::device>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::device>> infected,
    Rcpp::NumericVector x, Rcpp::NumericVector y, double base, double k,
    double width) {
  return spatial_infection_hybrid_wrapper(rngs, output, susceptible, infected,
                                          x, y, base, k, width);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::device>> infection_victims_device(
    Rcpp::XPtr<mob::infection_list<mob::system::device>> infections,
    size_t capacity) {
  return infection_victims_wrapper<mob::system::device>(infections, capacity);
}

// [[Rcpp::export]]
Rcpp::DataFrame infections_as_dataframe_device(
    Rcpp::XPtr<mob::infection_list<mob::system::device>> infections) {
  return infections_as_dataframe<mob::system::device>(infections);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::device>>
infections_from_dataframe_device(Rcpp::DataFrame df) {
  return infections_from_dataframe<mob::system::device>(df);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::device>> infections_select_device(
    Rcpp::XPtr<mob::device_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::device>> infections) {
  return infections_select_wrapper<mob::system::device>(rngs, infections);
}

// [[Rcpp::export]]
Rcpp::NumericVector selection_sampler_device(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return selection_sampler_wrapper<mob::system::device>(data, k, seed);
}

// [[Rcpp::export]]
Rcpp::NumericVector betabinomial_sampler_device(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return betabinomial_sampler_wrapper<mob::system::device>(data, k, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::device>>
bitset_create_device(size_t capacity) {
  return bitset_create<mob::system::device>(capacity);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::device>>
bitset_clone_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr) {
  return bitset_clone<mob::system::device>(ptr);
}

// [[Rcpp::export]]
size_t bitset_size_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr) {
  return bitset_size<mob::system::device>(ptr);
}

// [[Rcpp::export]]
void bitset_or_device(Rcpp::XPtr<mob::bitset<mob::system::device>> left,
                      Rcpp::XPtr<mob::bitset<mob::system::device>> right) {
  return bitset_or<mob::system::device>(left, right);
}

// [[Rcpp::export]]
void bitset_remove_device(Rcpp::XPtr<mob::bitset<mob::system::device>> left,
                          Rcpp::XPtr<mob::bitset<mob::system::device>> right) {
  return bitset_remove<mob::system::device>(left, right);
}

// [[Rcpp::export]]
bool bitset_equal_device(Rcpp::XPtr<mob::bitset<mob::system::device>> left,
                         Rcpp::XPtr<mob::bitset<mob::system::device>> right) {
  return bitset_equal<mob::system::device>(left, right);
}

// [[Rcpp::export]]
void bitset_invert_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr) {
  return bitset_invert<mob::system::device>(ptr);
}

// [[Rcpp::export]]
void bitset_insert_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr,
                          Rcpp::IntegerVector values) {
  return bitset_insert<mob::system::device>(ptr, values);
}

// [[Rcpp::export]]
void bitset_sample_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr,
                          Rcpp::XPtr<mob::device_random> rngs, double p) {
  return bitset_sample<mob::system::device>(ptr, rngs, p);
}

// [[Rcpp::export]]
void bitset_choose_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr,
                          Rcpp::XPtr<mob::device_random> rngs, size_t k) {
  return bitset_choose<mob::system::device>(ptr, rngs, k);
}

// [[Rcpp::export]]
Rcpp::IntegerVector
bitset_to_vector_device(Rcpp::XPtr<mob::bitset<mob::system::device>> ptr) {
  return bitset_to_vector<mob::system::device>(ptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::alias_table<mob::system::device>>
alias_table_create_device(Rcpp::DoubleVector weights) {
  return alias_table_create_wrapper<mob::system::device>(weights);
}

// [[Rcpp::export]]
Rcpp::DataFrame alias_table_values_device(
    Rcpp::XPtr<mob::alias_table<mob::system::device>> table) {
  return alias_table_values_wrapper<mob::system::device>(table);
}

// [[Rcpp::export]]
Rcpp::IntegerVector alias_table_sample_device(
    Rcpp::XPtr<mob::alias_table<mob::system::device>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::device>> rngs, size_t k) {
  return alias_table_sample_wrapper<mob::system::device>(table, rngs, k);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix alias_table_sample_wor_device(
    Rcpp::XPtr<mob::alias_table<mob::system::device>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::device>> rngs, size_t rows,
    size_t k) {
  return alias_table_sample_wor_wrapper<mob::system::device>(table, rngs, rows,
                                                             k);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix alias_table_sample_wor_ragged_matrix_device(
    Rcpp::XPtr<mob::alias_table<mob::system::device>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::device>> rngs,
    Rcpp::IntegerVector ks, size_t maxk) {
  return alias_table_sample_wor_ragged_matrix_wrapper<mob::system::device>(
      table, rngs, ks, maxk);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::device>>
integer_vector_create_device(Rcpp::IntegerVector values) {
  return integer_vector_create<mob::system::device>(values);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_values_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> v) {
  return vector_values<mob::system::device>(v);
}

// [[Rcpp::export]]
void integer_vector_scatter_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> indices,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> values) {
  vector_scatter<mob::system::device>(vector, indices, values);
}

// [[Rcpp::export]]
void integer_vector_scatter_scalar_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> indices,
    uint32_t value) {
  vector_scatter_scalar<mob::system::device>(vector, indices, value);
}

// [[Rcpp::export]]
void integer_vector_scatter_bitset_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::bitset<mob::system::device>> indices,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> values) {
  vector_scatter_bitset<mob::system::device>(vector, indices, values);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::device>>
integer_vector_gather_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> indices) {
  return vector_gather<mob::system::device>(vector, indices);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_match_eq_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> v, size_t value) {
  return integer_vector_match_eq<mob::system::device>(v, value);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_match_gt_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> v, size_t value) {
  return integer_vector_match_gt<mob::system::device>(v, value);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::device>>
integer_vector_match_eq_as_bitset_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> v, size_t value) {
  return integer_vector_match_eq_as_bitset<mob::system::device>(v, value);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::device>>
integer_vector_match_gt_as_bitset_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> v, size_t value) {
  return integer_vector_match_gt_as_bitset<mob::system::device>(v, value);
}

// [[Rcpp::export]]
void integer_vector_add_scalar_device(
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> v, int32_t delta) {
  return vector_add_scalar<mob::system::device>(v, delta);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::device>>
double_vector_create_device(Rcpp::NumericVector values) {
  return double_vector_create<mob::system::device>(values);
}

// [[Rcpp::export]]
Rcpp::NumericVector double_vector_values_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> v) {
  return vector_values<mob::system::device>(v);
}

// [[Rcpp::export]]
void double_vector_scatter_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> indices,
    Rcpp::XPtr<mob::double_vector<mob::system::device>> values) {
  vector_scatter<mob::system::device>(vector, indices, values);
}

// [[Rcpp::export]]
void double_vector_scatter_scalar_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> indices,
    double value) {
  vector_scatter_scalar<mob::system::device>(vector, indices, value);
}

// [[Rcpp::export]]
void double_vector_scatter_bitset_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::bitset<mob::system::device>> indices,
    Rcpp::XPtr<mob::double_vector<mob::system::device>> values) {
  vector_scatter_bitset<mob::system::device>(vector, indices, values);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::device>> double_vector_gather_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::device>> indices) {
  return vector_gather<mob::system::device>(vector, indices);
}

// [[Rcpp::export]]
void double_vector_add_scalar_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> v, double delta) {
  return vector_add_scalar<mob::system::device>(v, delta);
}

// [[Rcpp::export]]
void double_vector_div_scalar_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> v, double divisor) {
  return vector_div_scalar<mob::system::device>(v, divisor);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::device>>
double_vector_lround_device(
    Rcpp::XPtr<mob::double_vector<mob::system::device>> values) {
  return double_vector_lround<mob::system::device>(values);
}

#endif // __NVCC__

// [[Rcpp::export]]
Rcpp::XPtr<mob::host_random>
random_create_host(size_t size,
                   Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return random_create_wrapper<mob::system::host>(size, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::host>>
random_uniform_host(Rcpp::XPtr<mob::host_random> rngs, size_t n, double min,
                    double max) {
  return random_uniform_wrapper<mob::system::host>(rngs, n, min, max);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::host>>
random_poisson_host(Rcpp::XPtr<mob::host_random> rngs, size_t n,
                    double lambda) {
  return random_poisson_wrapper<mob::system::host>(rngs, n, lambda);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::host>>
random_binomial_host(Rcpp::XPtr<mob::host_random> rngs, size_t n, size_t size,
                     double prob) {
  return random_binomial_wrapper<mob::system::host>(rngs, n, size, prob);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::host>>
random_gamma_host(Rcpp::XPtr<mob::host_random> rngs, size_t n, double shape,
                  double scale) {
  return random_gamma<mob::system::host>(rngs, n, shape, scale);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::host>>
infection_list_create_host() {
  return infection_list_create_wrapper<mob::system::host>();
}

// [[Rcpp::export]]
size_t homogeneous_infection_process_host(
    Rcpp::XPtr<mob::host_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::host>> output,
    Rcpp::XPtr<mob::bitset<mob::system::host>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::host>> infected,
    double infection_probability) {
  return homogeneous_infection_process_wrapper<mob::system::host>(
      rngs, output, susceptible, infected, infection_probability);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::partition<mob::system::host>> partition_create_host(
    size_t capacity,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> population) {
  return partition_create_wrapper<mob::system::host>(capacity, population);
}

// [[Rcpp::export]]
Rcpp::IntegerVector
partition_sizes_host(Rcpp::XPtr<mob::ds::partition<mob::system::host>> p) {
  return partition_sizes_wrapper<mob::system::host>(p);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::ragged_vector<mob::system::host, uint32_t>>
ragged_vector_create_host(Rcpp::List values) {
  return ragged_vector_create_wrapper<mob::system::host>(values);
}

// [[Rcpp::export]]
Rcpp::IntegerVector ragged_vector_get_host(
    Rcpp::XPtr<mob::ds::ragged_vector<mob::system::host, uint32_t>> v,
    size_t i) {
  return ragged_vector_get_wrapper<mob::system::host>(v, i);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::host>>
ragged_vector_random_select_host(
    Rcpp::XPtr<mob::parallel_random<mob::system::host>> rngs,
    Rcpp::XPtr<mob::ds::ragged_vector<mob::system::host, uint32_t>> data) {
  return ragged_vector_random_select_wrapper<mob::system::host>(rngs, data);
}

// [[Rcpp::export]]
size_t household_infection_process_host(
    Rcpp::XPtr<mob::host_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::host>> output,
    Rcpp::XPtr<mob::bitset<mob::system::host>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::host>> infected,
    Rcpp::XPtr<mob::ds::partition<mob::system::host>> households,
    Rcpp::DoubleVector infection_probability) {
  return household_infection_process_wrapper<mob::system::host>(
      rngs, output, susceptible, infected, households, infection_probability);
}

// [[Rcpp::export]]
size_t spatial_infection_naive_host(
    Rcpp::XPtr<mob::host_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::host>> output,
    Rcpp::XPtr<mob::bitset<mob::system::host>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::host>> infected, Rcpp::NumericVector x,
    Rcpp::NumericVector y, double base, double k) {
  return spatial_infection_naive_wrapper(rngs, output, susceptible, infected, x,
                                         y, base, k);
}

// [[Rcpp::export]]
size_t spatial_infection_sieve_host(
    Rcpp::XPtr<mob::host_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::host>> output,
    Rcpp::XPtr<mob::bitset<mob::system::host>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::host>> infected, Rcpp::NumericVector x,
    Rcpp::NumericVector y, double base, double k) {
  return spatial_infection_sieve_wrapper(rngs, output, susceptible, infected, x,
                                         y, base, k);
}

// [[Rcpp::export]]
Rcpp::IntegerVector spatial_infection_hybrid_host(
    Rcpp::XPtr<mob::host_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::host>> output,
    Rcpp::XPtr<mob::bitset<mob::system::host>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::host>> infected, Rcpp::NumericVector x,
    Rcpp::NumericVector y, double base, double k, double width) {
  return spatial_infection_hybrid_wrapper(rngs, output, susceptible, infected,
                                          x, y, base, k, width);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::host>> infection_victims_host(
    Rcpp::XPtr<mob::infection_list<mob::system::host>> infections,
    size_t capacity) {
  return infection_victims_wrapper<mob::system::host>(infections, capacity);
}

// [[Rcpp::export]]
Rcpp::DataFrame infections_as_dataframe_host(
    Rcpp::XPtr<mob::infection_list<mob::system::host>> infections) {
  return infections_as_dataframe<mob::system::host>(infections);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::host>>
infections_from_dataframe_host(Rcpp::DataFrame df) {
  return infections_from_dataframe<mob::system::host>(df);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::host>> infections_select_host(
    Rcpp::XPtr<mob::host_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::host>> infections) {
  return infections_select_wrapper<mob::system::host>(rngs, infections);
}

// [[Rcpp::export]]
Rcpp::NumericVector
bernoulli_sampler_host(Rcpp::NumericVector data, double p,
                       Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return bernoulli_sampler_wrapper<mob::system::host>(data, p, seed);
}

// [[Rcpp::export]]
Rcpp::NumericVector
selection_sampler_host(Rcpp::NumericVector data, size_t k,
                       Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return selection_sampler_wrapper<mob::system::host>(data, k, seed);
}

// [[Rcpp::export]]
Rcpp::NumericVector betabinomial_sampler_host(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return betabinomial_sampler_wrapper<mob::system::host>(data, k, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::host>> bitset_create_host(size_t capacity) {
  return bitset_create<mob::system::host>(capacity);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::host>>
bitset_clone_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr) {
  return bitset_clone<mob::system::host>(ptr);
}

// [[Rcpp::export]]
size_t bitset_size_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr) {
  return bitset_size<mob::system::host>(ptr);
}

// [[Rcpp::export]]
void bitset_or_host(Rcpp::XPtr<mob::bitset<mob::system::host>> left,
                    Rcpp::XPtr<mob::bitset<mob::system::host>> right) {
  return bitset_or<mob::system::host>(left, right);
}

// [[Rcpp::export]]
void bitset_remove_host(Rcpp::XPtr<mob::bitset<mob::system::host>> left,
                        Rcpp::XPtr<mob::bitset<mob::system::host>> right) {
  return bitset_remove<mob::system::host>(left, right);
}

// [[Rcpp::export]]
void bitset_invert_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr) {
  return bitset_invert<mob::system::host>(ptr);
}

// [[Rcpp::export]]
bool bitset_equal_host(Rcpp::XPtr<mob::bitset<mob::system::host>> left,
                       Rcpp::XPtr<mob::bitset<mob::system::host>> right) {
  return bitset_equal<mob::system::host>(left, right);
}

// [[Rcpp::export]]
void bitset_insert_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr,
                        Rcpp::IntegerVector values) {
  return bitset_insert<mob::system::host>(ptr, values);
}

// [[Rcpp::export]]
void bitset_sample_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr,
                        Rcpp::XPtr<mob::host_random> rngs, double p) {
  return bitset_sample<mob::system::host>(ptr, rngs, p);
}

// [[Rcpp::export]]
void bitset_choose_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr,
                        Rcpp::XPtr<mob::host_random> rngs, size_t k) {
  return bitset_choose<mob::system::host>(ptr, rngs, k);
}

// [[Rcpp::export]]
Rcpp::IntegerVector
bitset_to_vector_host(Rcpp::XPtr<mob::bitset<mob::system::host>> ptr) {
  return bitset_to_vector<mob::system::host>(ptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::alias_table<mob::system::host>>
alias_table_create_host(Rcpp::DoubleVector weights) {
  return alias_table_create_wrapper<mob::system::host>(weights);
}

// [[Rcpp::export]]
Rcpp::DataFrame
alias_table_values_host(Rcpp::XPtr<mob::alias_table<mob::system::host>> table) {
  return alias_table_values_wrapper<mob::system::host>(table);
}

// [[Rcpp::export]]
Rcpp::IntegerVector alias_table_sample_host(
    Rcpp::XPtr<mob::alias_table<mob::system::host>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::host>> rngs, size_t k) {
  return alias_table_sample_wrapper<mob::system::host>(table, rngs, k);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix alias_table_sample_wor_host(
    Rcpp::XPtr<mob::alias_table<mob::system::host>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::host>> rngs, size_t rows,
    size_t k) {
  return alias_table_sample_wor_wrapper<mob::system::host>(table, rngs, rows,
                                                           k);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix alias_table_sample_wor_ragged_matrix_host(
    Rcpp::XPtr<mob::alias_table<mob::system::host>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::host>> rngs,
    Rcpp::IntegerVector ks, size_t maxk) {
  return alias_table_sample_wor_ragged_matrix_wrapper<mob::system::host>(
      table, rngs, ks, maxk);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::host>>
integer_vector_create_host(Rcpp::IntegerVector values) {
  return integer_vector_create<mob::system::host>(values);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_values_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> v) {
  return vector_values<mob::system::host>(v);
}

// [[Rcpp::export]]
void integer_vector_scatter_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> indices,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> values) {
  vector_scatter<mob::system::host>(vector, indices, values);
}

// [[Rcpp::export]]
void integer_vector_scatter_scalar_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> indices,
    uint32_t value) {
  vector_scatter_scalar<mob::system::host>(vector, indices, value);
}

// [[Rcpp::export]]
void integer_vector_scatter_bitset_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::bitset<mob::system::host>> indices,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> values) {
  vector_scatter_bitset<mob::system::host>(vector, indices, values);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::host>> integer_vector_gather_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> indices) {
  return vector_gather<mob::system::host>(vector, indices);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_match_eq_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> v, size_t value) {
  return integer_vector_match_eq<mob::system::host>(v, value);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_match_gt_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> v, size_t value) {
  return integer_vector_match_gt<mob::system::host>(v, value);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::host>>
integer_vector_match_eq_as_bitset_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> v, size_t value) {
  return integer_vector_match_eq_as_bitset<mob::system::host>(v, value);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::host>>
integer_vector_match_gt_as_bitset_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> v, size_t value) {
  return integer_vector_match_gt_as_bitset<mob::system::host>(v, value);
}

// [[Rcpp::export]]
void integer_vector_add_scalar_host(
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> v, int32_t delta) {
  return vector_add_scalar<mob::system::host>(v, delta);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::host>>
double_vector_create_host(Rcpp::NumericVector values) {
  return double_vector_create<mob::system::host>(values);
}

// [[Rcpp::export]]
Rcpp::NumericVector
double_vector_values_host(Rcpp::XPtr<mob::double_vector<mob::system::host>> v) {
  return vector_values<mob::system::host>(v);
}

// [[Rcpp::export]]
void double_vector_scatter_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> indices,
    Rcpp::XPtr<mob::double_vector<mob::system::host>> values) {
  vector_scatter<mob::system::host>(vector, indices, values);
}

// [[Rcpp::export]]
void double_vector_scatter_scalar_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> indices, double value) {
  vector_scatter_scalar<mob::system::host>(vector, indices, value);
}

// [[Rcpp::export]]
void double_vector_scatter_bitset_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::bitset<mob::system::host>> indices,
    Rcpp::XPtr<mob::double_vector<mob::system::host>> values) {
  vector_scatter_bitset<mob::system::host>(vector, indices, values);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::host>> double_vector_gather_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::host>> indices) {
  return vector_gather<mob::system::host>(vector, indices);
}

// [[Rcpp::export]]
void double_vector_add_scalar_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> v, double delta) {
  return vector_add_scalar<mob::system::host>(v, delta);
}

// [[Rcpp::export]]
void double_vector_div_scalar_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> v, double divisor) {
  return vector_div_scalar<mob::system::host>(v, divisor);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::host>> double_vector_lround_host(
    Rcpp::XPtr<mob::double_vector<mob::system::host>> values) {
  return double_vector_lround<mob::system::host>(values);
}

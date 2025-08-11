// Edit this in mob.build/inst/template/mob_types.h and regenerate with
// mob.build::generate_host()

#pragma once

// Keep this first. Both dust and CCCL try to define __host__/__device__ if not
// using nvcc, but at least CCCL does it only if not defined already. If we
// include CCCL first and then dust, we end up with loads of warnings about
// redefinition.
//
// clang-format: off
#include <dust/random/cuda_compatibility.hpp>
// clang-format: on

#include <mob/r/alias_table_wrapper.h>
#include <mob/r/bitset_wrapper.h>
#include <mob/r/infection_wrapper.h>
#include <mob/r/partition_wrapper.h>
#include <mob/r/random_wrapper.h>
#include <mob/r/sample_wrapper.h>
#include <mob/r/vector_wrapper.h>
#include <mob/system.h>

// [[Rcpp::export]]
Rcpp::XPtr<mob::{{ system }}_random>
random_create_{{ system }}(size_t size,
                     Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return random_create_wrapper<mob::system::{{ system }}>(size, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>>
random_uniform_{{ system }}(Rcpp::XPtr<mob::{{ system }}_random> rngs, size_t n, double min,
                      double max) {
  return random_uniform_wrapper<mob::system::{{ system }}>(rngs, n, min, max);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>>
random_poisson_{{ system }}(Rcpp::XPtr<mob::{{ system }}_random> rngs, size_t n,
                      double lambda) {
  return random_poisson_wrapper<mob::system::{{ system }}>(rngs, n, lambda);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>>
random_binomial_{{ system }}(Rcpp::XPtr<mob::{{ system }}_random> rngs, size_t n,
                       size_t size, double prob) {
  return random_binomial_wrapper<mob::system::{{ system }}>(rngs, n, size, prob);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>>
random_gamma_{{ system }}(Rcpp::XPtr<mob::{{ system }}_random> rngs, size_t n, double shape,
                    double scale) {
  return random_gamma<mob::system::{{ system }}>(rngs, n, shape, scale);
}

// [[Rcpp::export]]
Rcpp::NumericVector bernoulli_sampler_{{ system }}(
    Rcpp::NumericVector data, double p,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return bernoulli_sampler_wrapper<mob::system::{{ system }}>(data, p, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>>
infection_list_create_{{ system }}() {
  return infection_list_create_wrapper<mob::system::{{ system }}>();
}

// [[Rcpp::export]]
size_t homogeneous_infection_process_{{ system }}(
    Rcpp::XPtr<mob::{{ system }}_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> output,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> infected,
    double infection_probability) {
  return homogeneous_infection_process_wrapper<mob::system::{{ system }}>(
      rngs, output, susceptible, infected, infection_probability);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::partition<mob::system::{{ system }}>> partition_create_{{ system }}(
    size_t capacity,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> population) {
  return partition_create_wrapper<mob::system::{{ system }}>(capacity, population);
}

// [[Rcpp::export]]
Rcpp::IntegerVector
partition_sizes_{{ system }}(Rcpp::XPtr<mob::ds::partition<mob::system::{{ system }}>> p) {
  return partition_sizes_wrapper<mob::system::{{ system }}>(p);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::ds::ragged_vector<mob::system::{{ system }}, uint32_t>>
ragged_vector_create_{{ system }}(Rcpp::List values) {
  return ragged_vector_create_wrapper<mob::system::{{ system }}>(values);
}

// [[Rcpp::export]]
Rcpp::IntegerVector ragged_vector_get_{{ system }}(
    Rcpp::XPtr<mob::ds::ragged_vector<mob::system::{{ system }}, uint32_t>> v,
    size_t i) {
  return ragged_vector_get_wrapper<mob::system::{{ system }}>(v, i);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>>
ragged_vector_random_select_{{ system }}(
    Rcpp::XPtr<mob::parallel_random<mob::system::{{ system }}>> rngs,
    Rcpp::XPtr<mob::ds::ragged_vector<mob::system::{{ system }}, uint32_t>> data) {
  return ragged_vector_random_select_wrapper<mob::system::{{ system }}>(rngs, data);
}

// [[Rcpp::export]]
size_t household_infection_process_{{ system }}(
    Rcpp::XPtr<mob::{{ system }}_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> output,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> infected,
    Rcpp::XPtr<mob::ds::partition<mob::system::{{ system }}>> households,
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> infection_probability) {
  return household_infection_process_wrapper<mob::system::{{ system }}>(
      rngs, output, susceptible, infected, households, infection_probability);
}

// [[Rcpp::export]]
size_t spatial_infection_naive_{{ system }}(
    Rcpp::XPtr<mob::{{ system }}_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> output,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> infected,
    Rcpp::NumericVector x, Rcpp::NumericVector y, double base, double k) {
  return spatial_infection_naive_wrapper(rngs, output, susceptible, infected, x,
                                         y, base, k);
}

// [[Rcpp::export]]
size_t spatial_infection_sieve_{{ system }}(
    Rcpp::XPtr<mob::{{ system }}_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> output,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> infected,
    Rcpp::NumericVector x, Rcpp::NumericVector y, double base, double k) {
  return spatial_infection_sieve_wrapper(rngs, output, susceptible, infected, x,
                                         y, base, k);
}

// [[Rcpp::export]]
Rcpp::IntegerVector spatial_infection_hybrid_{{ system }}(
    Rcpp::XPtr<mob::{{ system }}_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> output,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> susceptible,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> infected,
    Rcpp::NumericVector x, Rcpp::NumericVector y, double base, double k,
    double width) {
  return spatial_infection_hybrid_wrapper(rngs, output, susceptible, infected,
                                          x, y, base, k, width);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> infection_victims_{{ system }}(
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> infections,
    size_t capacity) {
  return infection_victims_wrapper<mob::system::{{ system }}>(infections, capacity);
}

// [[Rcpp::export]]
Rcpp::DataFrame infections_as_dataframe_{{ system }}(
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> infections) {
  return infections_as_dataframe<mob::system::{{ system }}>(infections);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>>
infections_from_dataframe_{{ system }}(Rcpp::DataFrame df) {
  return infections_from_dataframe<mob::system::{{ system }}>(df);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> infections_select_{{ system }}(
    Rcpp::XPtr<mob::{{ system }}_random> rngs,
    Rcpp::XPtr<mob::infection_list<mob::system::{{ system }}>> infections) {
  return infections_select_wrapper<mob::system::{{ system }}>(rngs, infections);
}

// [[Rcpp::export]]
Rcpp::NumericVector selection_sampler_{{ system }}(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return selection_sampler_wrapper<mob::system::{{ system }}>(data, k, seed);
}

// [[Rcpp::export]]
Rcpp::NumericVector betabinomial_sampler_{{ system }}(
    Rcpp::NumericVector data, size_t k,
    Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
  return betabinomial_sampler_wrapper<mob::system::{{ system }}>(data, k, seed);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>>
bitset_create_{{ system }}(size_t capacity) {
  return bitset_create<mob::system::{{ system }}>(capacity);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>>
bitset_clone_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr) {
  return bitset_clone<mob::system::{{ system }}>(ptr);
}

// [[Rcpp::export]]
size_t bitset_size_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr) {
  return bitset_size<mob::system::{{ system }}>(ptr);
}

// [[Rcpp::export]]
void bitset_or_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> left,
                      Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> right) {
  return bitset_or<mob::system::{{ system }}>(left, right);
}

// [[Rcpp::export]]
void bitset_remove_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> left,
                          Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> right) {
  return bitset_remove<mob::system::{{ system }}>(left, right);
}

// [[Rcpp::export]]
bool bitset_equal_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> left,
                         Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> right) {
  return bitset_equal<mob::system::{{ system }}>(left, right);
}

// [[Rcpp::export]]
void bitset_invert_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr) {
  return bitset_invert<mob::system::{{ system }}>(ptr);
}

// [[Rcpp::export]]
void bitset_insert_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr,
                          Rcpp::IntegerVector values) {
  return bitset_insert<mob::system::{{ system }}>(ptr, values);
}

// [[Rcpp::export]]
void bitset_sample_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr,
                          Rcpp::XPtr<mob::{{ system }}_random> rngs, double p) {
  return bitset_sample<mob::system::{{ system }}>(ptr, rngs, p);
}

// [[Rcpp::export]]
void bitset_choose_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr,
                          Rcpp::XPtr<mob::{{ system }}_random> rngs, size_t k) {
  return bitset_choose<mob::system::{{ system }}>(ptr, rngs, k);
}

// [[Rcpp::export]]
Rcpp::IntegerVector
bitset_to_vector_{{ system }}(Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> ptr) {
  return bitset_to_vector<mob::system::{{ system }}>(ptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::alias_table<mob::system::{{ system }}>>
alias_table_create_{{ system }}(Rcpp::DoubleVector weights) {
  return alias_table_create_wrapper<mob::system::{{ system }}>(weights);
}

// [[Rcpp::export]]
Rcpp::DataFrame alias_table_values_{{ system }}(
    Rcpp::XPtr<mob::alias_table<mob::system::{{ system }}>> table) {
  return alias_table_values_wrapper<mob::system::{{ system }}>(table);
}

// [[Rcpp::export]]
Rcpp::IntegerVector alias_table_sample_{{ system }}(
    Rcpp::XPtr<mob::alias_table<mob::system::{{ system }}>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::{{ system }}>> rngs, size_t k) {
  return alias_table_sample_wrapper<mob::system::{{ system }}>(table, rngs, k);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix alias_table_sample_wor_{{ system }}(
    Rcpp::XPtr<mob::alias_table<mob::system::{{ system }}>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::{{ system }}>> rngs, size_t rows,
    size_t k) {
  return alias_table_sample_wor_wrapper<mob::system::{{ system }}>(table, rngs, rows,
                                                             k);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix alias_table_sample_wor_ragged_matrix_{{ system }}(
    Rcpp::XPtr<mob::alias_table<mob::system::{{ system }}>> table,
    Rcpp::XPtr<mob::parallel_random<mob::system::{{ system }}>> rngs,
    Rcpp::IntegerVector ks, size_t maxk) {
  return alias_table_sample_wor_ragged_matrix_wrapper<mob::system::{{ system }}>(
      table, rngs, ks, maxk);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>>
integer_vector_create_{{ system }}(Rcpp::IntegerVector values) {
  return integer_vector_create<mob::system::{{ system }}>(values);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_values_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> v) {
  return vector_values<mob::system::{{ system }}>(v);
}

// [[Rcpp::export]]
void integer_vector_scatter_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> indices,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> values) {
  vector_scatter<mob::system::{{ system }}>(vector, indices, values);
}

// [[Rcpp::export]]
void integer_vector_scatter_scalar_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> indices,
    uint32_t value) {
  vector_scatter_scalar<mob::system::{{ system }}>(vector, indices, value);
}

// [[Rcpp::export]]
void integer_vector_scatter_bitset_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> indices,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> values) {
  vector_scatter_bitset<mob::system::{{ system }}>(vector, indices, values);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>>
integer_vector_gather_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> indices) {
  return vector_gather<mob::system::{{ system }}>(vector, indices);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_match_eq_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> v, size_t value) {
  return integer_vector_match_eq<mob::system::{{ system }}>(v, value);
}

// [[Rcpp::export]]
Rcpp::IntegerVector integer_vector_match_gt_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> v, size_t value) {
  return integer_vector_match_gt<mob::system::{{ system }}>(v, value);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>>
integer_vector_match_eq_as_bitset_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> v, size_t value) {
  return integer_vector_match_eq_as_bitset<mob::system::{{ system }}>(v, value);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>>
integer_vector_match_gt_as_bitset_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> v, size_t value) {
  return integer_vector_match_gt_as_bitset<mob::system::{{ system }}>(v, value);
}

// [[Rcpp::export]]
void integer_vector_add_scalar_{{ system }}(
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> v, int32_t delta) {
  return vector_add_scalar<mob::system::{{ system }}>(v, delta);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>>
double_vector_create_{{ system }}(Rcpp::NumericVector values) {
  return double_vector_create<mob::system::{{ system }}>(values);
}

// [[Rcpp::export]]
Rcpp::NumericVector double_vector_values_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> v) {
  return vector_values<mob::system::{{ system }}>(v);
}

// [[Rcpp::export]]
void double_vector_scatter_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> indices,
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> values) {
  vector_scatter<mob::system::{{ system }}>(vector, indices, values);
}

// [[Rcpp::export]]
void double_vector_scatter_scalar_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> indices,
    double value) {
  vector_scatter_scalar<mob::system::{{ system }}>(vector, indices, value);
}

// [[Rcpp::export]]
void double_vector_scatter_bitset_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::bitset<mob::system::{{ system }}>> indices,
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> values) {
  vector_scatter_bitset<mob::system::{{ system }}>(vector, indices, values);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> double_vector_gather_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> vector,
    Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>> indices) {
  return vector_gather<mob::system::{{ system }}>(vector, indices);
}

// [[Rcpp::export]]
void double_vector_add_scalar_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> v, double delta) {
  return vector_add_scalar<mob::system::{{ system }}>(v, delta);
}

// [[Rcpp::export]]
void double_vector_div_scalar_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> v, double divisor) {
  return vector_div_scalar<mob::system::{{ system }}>(v, divisor);
}

// [[Rcpp::export]]
Rcpp::XPtr<mob::integer_vector<mob::system::{{ system }}>>
double_vector_lround_{{ system }}(
    Rcpp::XPtr<mob::double_vector<mob::system::{{ system }}>> values) {
  return double_vector_lround<mob::system::{{ system }}>(values);
}

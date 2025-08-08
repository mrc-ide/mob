library(S7)

#' Create a wrapper function that dispatches to either the device or host implementation.
#'
#' This produces a wrapper function that will call either `{name}_host` or
#' `{name}_device`. The wrapper has a `system` named argument that is used to
#' choose the implementation. All other arguments are forwarded to the
#' underlying function.
#'
#' The wrapper function is generated using metaprogramming, allowing it to have
#' an explicit list of formal parameter names. If we just used `...` to forward
#' argument the argument names wouldn't show up in the documentation.
create_wrapper <- function(name) {
  f_host <- sprintf("%s_host", name)
  f_device <- sprintf("%s_device", name)

  if (!identical(formals(f_host), formals(f_device))) {
    abort(sprintf("host and device implementations of %s has different formals", name))
  }

  nm <- names(formals(f_host))
  args <- rlang::exprs(!!!formals(f_host), ... = !!rlang::missing_arg(), system = getOption("mob.system"))
  wrapper <- rlang::new_function(args, rlang::expr({
    rlang::check_dots_empty()
    match_value(system, c("device", "host"))
    switch(system,
           host = !!rlang::call2(f_host, !!!rlang::syms(nm)),
           device = !!rlang::call2(f_device, !!!rlang::syms(nm)))
  }))

  wrapper
}

class_integer_vector <- S7::new_S3_class("integer_vector")
class_double_vector <- S7::new_S3_class("double_vector")
class_bitset <- S7::new_S3_class("bitset")

#' Create a parallel random number generator.
#' 
#' @export
random_create <- create_wrapper("random_create")

#' @export
random_uniform <- create_wrapper("random_uniform")

#' @export
random_poisson <- create_wrapper("random_poisson")

#' @export
random_binomial <- create_wrapper("random_binomial")

#' @export
random_gamma <- create_wrapper("random_gamma")

#' @export
selection_sampler <- create_wrapper("selection_sampler")

#' @export
betabinomial_sampler <- create_wrapper("betabinomial_sampler")

#' @export
bernoulli_sampler <- create_wrapper("bernoulli_sampler")

#' @export
partition_create <- new_generic("partition_create", c("capacity", "population"))
method(partition_create, list(class_numeric, class_integer_vector)) <- create_wrapper("partition_create")
method(partition_create, list(class_numeric, class_numeric)) <- function(capacity, population, ...) {
  partition_create(capacity, integer_vector_create(population), ...)
}

#' @export
partition_sizes <- create_wrapper("partition_sizes")

#' @export
infection_list_create <- create_wrapper("infection_list_create")

#' @export
homogeneous_infection_process <- create_wrapper("homogeneous_infection_process")

#' @export
household_infection_process <- create_wrapper("household_infection_process")

#' @export
infection_victims <- create_wrapper("infection_victims")

#' @export
infections_as_dataframe <- create_wrapper("infections_as_dataframe")

#' @export
infections_from_dataframe <- create_wrapper("infections_from_dataframe")

#' @export
infections_select <- create_wrapper("infections_select")

#' @export
spatial_infection_naive <- create_wrapper("spatial_infection_naive")

#' @export
spatial_infection_sieve <- create_wrapper("spatial_infection_sieve")

#' @export
spatial_infection_hybrid <- create_wrapper("spatial_infection_hybrid")

#' Create a new bitset.
#' @export
bitset_create <- create_wrapper("bitset_create")

#' Create a copy of an existing bitset.
#' @export
bitset_clone <- create_wrapper("bitset_clone")

#' Perform an in-place union of two bitsets.
#' @export
bitset_or <- create_wrapper("bitset_or")

#' Perform an in-place difference of two bitsets.
#' @export
bitset_remove <- create_wrapper("bitset_remove")

#' Perform an in-place negation of a bitset.
#' @export
bitset_invert <- create_wrapper("bitset_invert")

#' Insert an R integer vector into a bitset.
#' @export
bitset_insert <- create_wrapper("bitset_insert")

#' Returns true if the two bitsets are equal.
#' @export
bitset_equal <- create_wrapper("bitset_equal")

#' Return the number of set bits.
#' @export
bitset_size <- create_wrapper("bitset_size")

#' Retain only a random portion of bits, using on independent bernoulli trials.
#' @export
bitset_sample <- create_wrapper("bitset_sample")

#' Retain only a given number of bits.
#' @export
bitset_choose <- create_wrapper("bitset_choose")

#' Convert the bitset into an R integer vector.
#' @export
bitset_to_vector <- create_wrapper("bitset_to_vector")

#' Convert an R integer vector into a bitset.
#' @export
bitset_from_vector <- function(capacity, values) {
  b <- bitset_create(capacity)
  bitset_insert(b, values)
  b
}

#' @export
ragged_vector_create <- create_wrapper("ragged_vector_create")

#' @export
ragged_vector_get <- create_wrapper("ragged_vector_get")

#' @export
ragged_vector_random_select <- create_wrapper("ragged_vector_random_select")

#' @export
alias_table_create <- create_wrapper("alias_table_create")

#' @export
alias_table_values <- create_wrapper("alias_table_values")

#' @export
alias_table_sample <- create_wrapper("alias_table_sample")

#' @export
alias_table_sample_wor <- create_wrapper("alias_table_sample_wor")

#' @export
alias_table_sample_wor_ragged_matrix <- create_wrapper("alias_table_sample_wor_ragged_matrix")

#' @export
integer_vector_create <- create_wrapper("integer_vector_create")

#' Get the contents of the vector as an R atomic vector.
#' @export
vector_values <- new_generic("vector_values", c("v"))
method(vector_values, class_integer_vector) <- create_wrapper("integer_vector_values")
method(vector_values, class_double_vector) <- create_wrapper("double_vector_values")

#' @export
vector_scatter <- new_generic("vector_scatter", c("vector", "indices", "values"))
method(vector_scatter, list(class_any, class_numeric, class_any)) <- function(vector, indices, values) {
  vector_scatter(vector, integer_vector_create(indices), values)
}

method(vector_scatter, list(class_integer_vector, class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_scatter")
method(vector_scatter, list(class_integer_vector, class_bitset, class_integer_vector)) <- create_wrapper("integer_vector_scatter_bitset")
method(vector_scatter, list(class_integer_vector, class_any, class_numeric)) <- function(vector, indices, values, ...) {
  vector_scatter(vector, indices, integer_vector_create(values), ...)
}

method(vector_scatter, list(class_double_vector, class_integer_vector, class_double_vector)) <- create_wrapper("double_vector_scatter")
method(vector_scatter, list(class_double_vector, class_bitset, class_double_vector)) <- create_wrapper("double_vector_scatter_bitset")
method(vector_scatter, list(class_double_vector, class_any, class_numeric)) <- function(vector, indices, values, ...) {
  vector_scatter(vector, indices, double_vector_create(values), ...)
}

#' @export
vector_scatter_scalar <- new_generic("vector_scatter_scalar", c("vector", "indices"))
method(vector_scatter_scalar, list(class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_scatter_scalar")
method(vector_scatter_scalar, list(class_double_vector, class_integer_vector)) <- create_wrapper("double_vector_scatter_scalar")
method(vector_scatter_scalar, list(class_any, class_numeric)) <- function(vector, indices, ...) {
  vector_scatter_scalar(vector, integer_vector_create(indices), ...)
}

#' @export
vector_gather <- new_generic("vector_gather", c("vector", "indices"))
method(vector_gather, list(class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_gather")
method(vector_gather, list(class_double_vector, class_integer_vector)) <- create_wrapper("double_vector_gather")
method(vector_gather, list(class_any, class_numeric)) <- function(vector, indices, ...) {
  vector_gather(vector, integer_vector_create(indices), ...)
}

#' Find indices of elements equal to the given value.
#' @export
integer_vector_match_eq <- create_wrapper("integer_vector_match_eq")

#' Find indices of elements greaher than the given value.
#' @export
integer_vector_match_gt <- create_wrapper("integer_vector_match_gt")

#' @export
integer_vector_match_eq_as_bitset <- create_wrapper("integer_vector_match_eq_as_bitset")

#' @export
integer_vector_match_gt_as_bitset <- create_wrapper("integer_vector_match_gt_as_bitset")

#' @export
vector_add_scalar <- new_generic("vector_add_scalar", c("v"))
method(vector_add_scalar, class_integer_vector) <- create_wrapper("integer_vector_add_scalar")
method(vector_add_scalar, class_double_vector) <- create_wrapper("double_vector_add_scalar")

#' @export
vector_div_scalar <- new_generic("vector_div_scalar", c("v"))
method(vector_div_scalar, class_double_vector) <- create_wrapper("double_vector_div_scalar")

#' Round values of the vector to the nearest integer.
#' @export
double_vector_lround <- create_wrapper("double_vector_lround")

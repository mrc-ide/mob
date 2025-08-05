create_wrapper <- function(name, f) {
  f_host <- sprintf("%s_host", name)
  f_device <- sprintf("%s_host", name)

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

#' Create a parallel random number generator.
#' 
#' @export
random_create <- create_wrapper("random_create")

#' @export
random_uniform <- create_wrapper("random_uniform")

random_uniform_benchmark <- create_wrapper("random_uniform_benchmark")

#' @export
random_poisson <- create_wrapper("random_poisson")

#' @export
random_binomial <- create_wrapper("random_binomial")

#' @export
selection_sampler <- create_wrapper("selection_sampler")

#' @export
betabinomial_sampler <- create_wrapper("betabinomial_sampler")

#' @export
bernoulli_sampler <- create_wrapper("bernoulli_sampler")

#' @export
partition_create <- create_wrapper("partition_create")

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

#' @export
integer_vector_values <- create_wrapper("integer_vector_values")

#' @export
integer_vector_scatter <- create_wrapper("integer_vector_scatter")

#' @export
integer_vector_scatter_bitset <- create_wrapper("integer_vector_scatter_bitset")

#' @export
integer_vector_scatter_scalar <- create_wrapper("integer_vector_scatter_scalar")

#' @export
integer_vector_gather <- create_wrapper("integer_vector_gather")

#' @export
integer_vector_match <- create_wrapper("integer_vector_match")

#' @export
integer_vector_match_bitset <- create_wrapper("integer_vector_match_bitset")

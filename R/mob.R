library(S7)

#' Create a wrapper function that dispatches to either the device or host implementation.
#'
#' This produces a wrapper function that will call either `{name}_host` or
#' `{name}_device`. All arguments are forwarded to the underlying function.
#'
#' The wrapper function is generated using metaprogramming, allowing it to have
#' an explicit list of formal parameter names. If we just used `...` to forward
#' argument the argument names wouldn't show up in the documentation.
create_wrapper <- function(name) {
  f_host <- sprintf("%s_host", name)
  f_device <- rlang::call2(
    rlang::sym(":::"),
    rlang::sym("mob.device"),
    rlang::sym(sprintf("%s_device", name)))

  nm <- names(formals(f_host))
  wrapper <- rlang::new_function(formals(f_host), rlang::expr({
    system <- match_value(getOption("mob.system"), c("device", "host"))
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
method(partition_create, list(class_numeric, class_numeric)) <- function(capacity, population) {
  partition_create(capacity, integer_vector_create(population))
}

#' @export
partition_sizes <- create_wrapper("partition_sizes")

#' @export
infection_list_create <- create_wrapper("infection_list_create")

#' @export
homogeneous_infection_process <- create_wrapper("homogeneous_infection_process")

#' @export
household_infection_process <- new_generic("household_infection_process", c("rngs", "output", "susceptible", "infected", "households", "infection_probability"))
method(household_infection_process,
       list(rngs = class_any,
            output = class_any,
            susceptible = class_any,
            infected = class_any,
            households = class_any,
            infection_probability = class_double_vector)) <- create_wrapper("household_infection_process")
method(household_infection_process,
       list(rngs = class_any,
            output = class_any,
            susceptible = class_any,
            infected = class_any,
            households = class_any,
            infection_probability = class_numeric)) <- function(rngs, output, susceptible, infected, households, infection_probability) {
  household_infection_process(rngs, output, susceptible, infected, households, double_vector_create(infection_probability))
}

create_wrapper("household_infection_process")

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

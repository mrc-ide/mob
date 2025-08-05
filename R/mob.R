create_wrapper <- function(name, f) {
  function(..., system = getOption("mob.system")) {
    match_value(system, c("device", "host"))
    do.call(sprintf("%s_%s", name, system), list(...))
  }
}

random_create <- create_wrapper("random_create")
random_uniform <- create_wrapper("random_uniform")
random_uniform_benchmark <- create_wrapper("random_uniform_benchmark")
random_poisson <- create_wrapper("random_poisson")
random_binomial <- create_wrapper("random_binomial")

selection_sampler <- create_wrapper("selection_sampler")
betabinomial_sampler <- create_wrapper("betabinomial_sampler")
bernoulli_sampler <- create_wrapper("bernoulli_sampler")

partition_create <- create_wrapper("partition_create")
partition_sizes <- create_wrapper("partition_sizes")

infection_list_create <- create_wrapper("infection_list_create")
homogeneous_infection_process <- create_wrapper("homogeneous_infection_process")
household_infection_process <- create_wrapper("household_infection_process")
infection_victims <- create_wrapper("infection_victims")
infections_as_dataframe <- create_wrapper("infections_as_dataframe")
infections_from_dataframe <- create_wrapper("infections_from_dataframe")
infections_select <- create_wrapper("infections_select")
spatial_infection_naive <- create_wrapper("spatial_infection_naive")
spatial_infection_sieve <- create_wrapper("spatial_infection_sieve")
spatial_infection_hybrid <- create_wrapper("spatial_infection_hybrid")

bitset_create <- create_wrapper("bitset_create")
bitset_clone <- create_wrapper("bitset_clone")
bitset_or <- create_wrapper("bitset_or")
bitset_remove <- create_wrapper("bitset_remove")
bitset_invert <- create_wrapper("bitset_invert")
bitset_insert <- create_wrapper("bitset_insert")
bitset_size <- create_wrapper("bitset_size")
bitset_sample <- create_wrapper("bitset_sample")
bitset_choose <- create_wrapper("bitset_choose")
bitset_to_vector <- create_wrapper("bitset_to_vector")

bitset_from_vector <- function(capacity, values) {
  b <- bitset_create(capacity)
  bitset_insert(b, values)
  b
}

ragged_vector_create <- create_wrapper("ragged_vector_create")
ragged_vector_get <- create_wrapper("ragged_vector_get")
ragged_vector_random_select <- create_wrapper("ragged_vector_random_select")

alias_table_create <- create_wrapper("alias_table_create")
alias_table_values <- create_wrapper("alias_table_values")
alias_table_sample <- create_wrapper("alias_table_sample")
alias_table_sample_wor <- create_wrapper("alias_table_sample_wor")
alias_table_sample_wor_ragged_matrix <- create_wrapper("alias_table_sample_wor_ragged_matrix")

integer_vector_create <- create_wrapper("integer_vector_create")
integer_vector_values <- create_wrapper("integer_vector_values")
integer_vector_scatter <- create_wrapper("integer_vector_scatter")
integer_vector_scatter_scalar <- create_wrapper("integer_vector_scatter_scalar")
integer_vector_gather <- create_wrapper("integer_vector_gather")
integer_vector_match <- create_wrapper("integer_vector_match")

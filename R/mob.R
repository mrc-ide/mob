create_wrapper <- function(name, f) {
  function(..., system = getOption("mob.system")) {
    match_value(system, c("device", "host"))
    do.call(sprintf("%s_%s", name, system), list(...))
  }
}

random_create <- create_wrapper("random_create")
random_uniform <- create_wrapper("random_uniform")
random_binomial <- create_wrapper("random_binomial")

selection_sampler <- create_wrapper("selection_sampler")
betabinomial_sampler <- create_wrapper("betabinomial_sampler")
bernoulli_sampler <- create_wrapper("bernoulli_sampler")

partition_create <- create_wrapper("partition_create")
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
bitset_to_vector <- create_wrapper("bitset_to_vector")

bitset_from_vector <- function(capacity, values) {
  b <- bitset_create(capacity)
  bitset_insert(b, values)
  b
}

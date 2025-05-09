create_wrapper <- function(name, f) {
  function(..., system = getOption("mob.system")) {
    match_value(system, c("device", "host"))
    do.call(sprintf("%s_%s", name, system), list(...))
  }
}

selection_sampler <- create_wrapper("selection_sampler")
betabinomial_sampler <- create_wrapper("betabinomial_sampler")
bernoulli_sampler <- create_wrapper("bernoulli_sampler")

random_uniform <- create_wrapper("random_uniform")
random_binomial <- create_wrapper("random_binomial")

random_create <- create_wrapper("random_create")
partition_create <- create_wrapper("partition_create")
homogeneous_infection_process <- create_wrapper("homogeneous_infection_process")
household_infection_process <- create_wrapper("household_infection_process")

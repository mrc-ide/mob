rate_to_p <- function(rate, dt) {
  1 - exp(-rate * dt)
}

beta_to_p <- function(beta, size, dt) {
  rate_to_p(beta / size, dt)
}

run <- function(size, dt = 1, timesteps = 200) {
  population <- 0:(size-1)
  # infected <- mob:::bitset_from_vector(size, sort(sample(population, 0.01 * size)))
  infected <- mob:::bitset_from_vector(size, sample(population, 1))
  susceptible <- mob:::bitset_clone(infected)
  mob:::bitset_invert(susceptible)
  recovered <- mob:::bitset_create(size)

  nhouseholds <- size %/% 5
  households_data <- sample.int(nhouseholds, size, replace=TRUE) - 1
  household_sizes <- rep(0, nhouseholds)
  for (i in households_data) {
    household_sizes[i+1] <- household_sizes[i+1] + 1
  }

  coordinates <- data.frame(x=runif(size), y=runif(size))

  p_community <- 0 # beta_to_p(0.06, size, dt)
  p_household <- 0 # beta_to_p(0.06, household_sizes, dt)
  p_recovery <- 0 # rate_to_p(0.05, dt)
  r_spatial <- 0.000001 / size

  households <- mob:::partition_create(nhouseholds, households_data)
  rngs <- mob:::random_create(size)

  render <- individual::Render$new(timesteps)

  data <- list()
  for (t in seq_len(timesteps)) {
    before <- Sys.time()
    infections <- mob:::infection_list_create()
    n_community <- mob:::homogeneous_infection_process(rngs, infections, susceptible, infected, p_community)
    n_household <- mob:::household_infection_process(rngs, infections, susceptible, infected, households, p_household)
    n_spatial <- mob:::spatial_infection_hybrid(
      rngs, infections, susceptible, infected, coordinates$x, coordinates$y,
      base=r_spatial, k=-4, width=0.015)

    victims <- mob:::infection_victims(infections, size)

    recovery <- mob:::bitset_clone(infected)
    mob:::bitset_sample(recovery, rngs, p_recovery)

    mob:::bitset_or(infected, victims)
    mob:::bitset_remove(susceptible, victims)

    mob:::bitset_or(recovered, recovery)
    mob:::bitset_remove(infected, recovery)

    state <- rep(NA_character_, size)
    state[mob:::bitset_to_vector(susceptible)+1] <- "S"
    state[mob:::bitset_to_vector(infected)+1] <- "I"
    state[mob:::bitset_to_vector(recovered)+1] <- "R"

    after <- Sys.time()

    data <- c(data, list(data.frame(timestep=t, state=state, i=seq_len(size), x=coordinates$x, y=coordinates$y)))

    render$render("S", mob:::bitset_size(susceptible), t)
    render$render("I", mob:::bitset_size(infected), t)
    render$render("R", mob:::bitset_size(recovered), t)
    render$render("n_community", n_community, t)
    render$render("n_household", n_household, t)
    render$render("n_spatial", sum(n_spatial), t)
    render$render("n_spatial_local", n_spatial[[1]], t)
    render$render("n_spatial_distant", n_spatial[[2]], t)
    render$render("n_recovery", mob:::bitset_size(recovery), t)
    render$render("time", as.numeric(after - before, units="secs"), t)
    cat(sprintf("%f %f %f\n",
                as.numeric(after - before, units="secs"),
                n_spatial[[1]],
                n_spatial[[2]]))
  }

  list(statistics=render$to_dataframe(), particles=dplyr::bind_rows(data))
}

# results <- bench::press(
#   system = c("host", "device"),
#   size = c(1e3, 1e4, 1e5, 1e6),
#   withr::with_options(list("mob.system" = system), bench::mark(run(size)))
# )
# print(results)

# withr::with_options(list("mob.system" = "device"), {
#   run(size = 1e7)
# })

# source("../profile.R")
# withr::with_options(list("mob.system" = "device"), {
#   profiler_run("output.prof", )
#   # proffer::record_pprof(pprof = "output.prof", run(size = 1e6))
#   # profiler_run("output.prof", run(size=1e6))
# })

library(ggplot2)
library(gganimate)
library(magrittr)

source("profile.R")
data <- withr::with_options(list("mob.system" = "host"), {
  profiler_run("output.prof", {
    run(50000)
  })
})

p <- ggplot(data$particles, aes(x, y, colour=state)) +
  geom_point(shape = ".") +
  labs(title = 't={current_frame}') +
  transition_manual(timestep)

# df <- data$statistics %>% tidyr::pivot_longer(cols=!timestep)
# 
# library(gridExtra)
# base <- ggplot(df, aes(x=timestep, y=value, color=name))
# p1 <- base + geom_line(data = . %>% dplyr::filter(name %in% c("S","I","R")))
# p2 <- base + geom_line(data = . %>% dplyr::filter(name %in% c("n_community", "n_household", "n_spatial")))
# p3 <- base + geom_line(data = . %>% dplyr::filter(name %in% c("n_spatial_local", "n_spatial_distant")))
# print(grid.arrange(p1, p2, p3, ncol=2))

library(ggplot2)
library(gganimate)
library(magrittr)
library(gridExtra)

options(viewer = function(url) {
    utils::browseURL(url, browser="eog")
})

rate_to_p <- function(rate, dt) {
  1 - exp(-rate * dt)
}

beta_to_p <- function(beta, size, dt) {
  rate_to_p(beta / size, dt)
}

# https://doi.org/10.5285/7beefde9-c520-4ddf-897a-0167e8918595
sample_population <- function(map, size) {
  weights <- tidyr::replace_na(terra::values(map), 0)
  selected <- sample(seq_along(weights), size, replace = TRUE, prob = weights)

  terra::xyFromCell(map, selected) |>
    as.data.frame() |>
    sf::st_as_sf(coords = c(1,2), crs = terra::crs(map)) |>
    tibble::add_column(i = 0:(size-1))
}

run <- function(
  population,
  dt = 1,
  timesteps = 200,
  beta_community = 0.06,
  beta_household = 0.06,
  gamma = 0.05)
{
  size <- nrow(population)
  rngs <- mob:::random_create(size)

  infected <- mob:::bitset_create(size)
  mob:::bitset_invert(infected)
  mob:::bitset_choose(infected, rngs, 1)

  susceptible <- mob:::bitset_clone(infected)
  mob:::bitset_invert(susceptible)

  recovered <- mob:::bitset_create(size)

  nhouseholds <- size %/% 5
  households_data <- sample.int(nhouseholds, size, replace=TRUE) - 1
  household_sizes <- rep(0, nhouseholds)
  for (i in households_data) {
    household_sizes[i+1] <- household_sizes[i+1] + 1
  }

  coordinates <- sf::st_coordinates(population)

  p_community <- beta_to_p(beta_community, size, dt)
  p_household <- beta_to_p(beta_household, household_sizes, dt)
  p_recovery <- rate_to_p(gamma, dt)
  r_spatial <- 0.01 / size

  households <- mob:::partition_create(nhouseholds, households_data)

  render <- individual::Render$new(timesteps)

  depth <- rep(NA_real_, size)
  depth[mob:::bitset_to_vector(infected) + 1] <- 0

  data <- list()
  for (t in seq_len(timesteps)) {
    before <- Sys.time()
    infections <- mob:::infection_list_create()
    n_community <- mob:::homogeneous_infection_process(rngs, infections, susceptible, infected, p_community)
    n_household <- mob:::household_infection_process(rngs, infections, susceptible, infected, households, p_household)
    n_spatial <- mob:::spatial_infection_hybrid(
      rngs, infections, susceptible, infected, coordinates[,1], coordinates[,2],
      base=r_spatial, k=-4, width=0.1)

    infections <- mob:::infections_select(rngs, infections)
    infections_df <- mob:::infections_as_dataframe(infections)
    depth[infections_df$victim + 1] <- depth[infections_df$source + 1] + 1 #abs(rnorm(nrow(infections_df)))

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
    elapsed <- after - before

    data <- c(data, list(data.frame(timestep=t, state=state, i=0:(size-1))))

    render$render("S", mob:::bitset_size(susceptible), t)
    render$render("I", mob:::bitset_size(infected), t)
    render$render("R", mob:::bitset_size(recovered), t)
    render$render("n_community", n_community, t)
    render$render("n_household", n_household, t)
    render$render("n_spatial", sum(n_spatial), t)
    render$render("n_spatial_local", n_spatial[[1]], t)
    render$render("n_spatial_distant", n_spatial[[2]], t)
    render$render("n_recovery", mob:::bitset_size(recovery), t)
    render$render("time", as.numeric(elapsed, units="secs"), t)

    cli::cli_alert("{t} local={n_spatial[[1]]} distant={n_spatial[[2]]} {elapsed}")
  }

  list(
    statistics = render$to_dataframe(),
    state = data,
    depth = data.frame(i=0:(size-1), depth=depth)
  )
}

render <- function(population, state, ncols=200, nrows=200) {
  bbox <- sf::st_bbox(population)
  r <- terra::rast(ncols=ncols, nrows=nrows,
                   xmin = bbox$xmin,
                   xmax = bbox$xmax,
                   ymin = bbox$ymin,
                   ymax = bbox$ymax,
                   crs = terra::crs(population))

  r <- terra::rasterize(population, r, field="depth")
  r
}

outline <- sf::read_sf("data/CTRY_DEC_2021_UK_BUC/")
map <- terra::rast("data/uk_residential_population_2021.tif") %>%
  terra::project("OGC:CRS84")

withr::local_options("mob.system" = "device")

population <- sample_population(map, 1e5)
result <- run(population, timesteps = 100, beta_community = 0, beta_household = 0, gamma = 0)

p <- ggplot() +
  geom_sf(data = outline) +
  tidyterra::geom_spatraster(
    data = render(population %>% dplyr::inner_join(result$depth), result$state[[100]]),
    aes(fill = after_stat(value))
  ) +
  scale_fill_viridis_c(option = "plasma", na.value = "transparent")

ggsave("map.pdf", p)

df <- result$statistics %>% tidyr::pivot_longer(cols=!timestep)
base <- ggplot(df, aes(x=timestep, y=value, color=name))
p1 <- base + geom_line(data = . %>% dplyr::filter(name %in% c("S","I","R")))
p2 <- base + geom_line(data = . %>% dplyr::filter(name %in% c("n_community", "n_household", "n_spatial")))
p3 <- base + geom_line(data = . %>% dplyr::filter(name %in% c("n_spatial_local", "n_spatial_distant")))
p <- arrangeGrob(p1, p2, p3, ncol=2)
ggsave("statistics.pdf", p)

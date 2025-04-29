test_that("homogeneous_infection_process", {
  population <- 0:99
  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)

  rngs <- device_random_create(length(population))

  result <- homogeneous_infection_process(rngs, susceptible, infected, 0.1)
  expect_in(result$source, infected)
  expect_in(result$victim, susceptible)
})

test_that("household_infection_process", {
  population <- 0:99
  households <- 50

  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)
  individual_household <- sample.int(households, length(population), replace = TRUE) - 1

  rngs <- device_random_create(length(population))

  result <- household_infection_process(rngs, sort(susceptible), infected, individual_household, 0.1)
  expect_in(result$source, infected)
  expect_in(result$victim, susceptible)
  expect_equal(individual_household[result$source + 1],
               individual_household[result$victim + 1])
})

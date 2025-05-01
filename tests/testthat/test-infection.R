skip_on_ci()

test_that("homogeneous_infection_process with p = 0", {
  population <- 0:99
  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)

  rngs <- device_random_create(length(population))

  result <- homogeneous_infection_process(rngs, susceptible, infected, 0)
  expect_equal(nrow(result), 0)
})

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

  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)
  households <- sample.int(50, length(population), replace = TRUE) - 1
  households_partition <- create_partition(households)

  rngs <- device_random_create(length(population))

  result <- household_infection_process(rngs, sort(susceptible), infected,
                                        households_partition, 0.3)
  expect_in(result$source, infected)
  expect_in(result$victim, susceptible)
  expect_equal(households[result$source + 1],
               households[result$victim + 1])
})

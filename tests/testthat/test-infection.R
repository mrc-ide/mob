mob_test("homogeneous_infection_process with p = 0", {
  population <- 0:99
  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)

  rngs <- random_create(length(population))
  result <- homogeneous_infection_process(rngs, susceptible, infected, 0)
  expect_equal(nrow(result), 0)
})

mob_test("homogeneous_infection_process", {
  population <- 0:99
  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)

  rngs <- random_create(length(population))
  result <- homogeneous_infection_process(rngs, susceptible, infected, 0.1)
  expect_in(result$source, infected)
  expect_in(result$victim, susceptible)
})

mob_test("household_infection_process", {
  population <- 0:99

  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)
  households <- sample.int(50, length(population), replace = TRUE) - 1
  households_partition <- partition_create(households)

  rngs <- random_create(length(population))

  result <- household_infection_process(rngs, sort(susceptible), infected,
                                        households_partition, 0.3)
  expect_in(result$source, infected)
  expect_in(result$victim, susceptible)
  expect_equal(households[result$source + 1],
               households[result$victim + 1])
})

mob_test("household_infection_process can use different probabilities per household", {
  susceptible <- 0:3
  infected <- 4:7
  households <- partition_create(c(0,1,0,1,0,1,0,1))
  probabilities <- c(0, 1)

  rngs <- random_create(8)
  result <- household_infection_process(rngs, susceptible, infected, households, probabilities)

  # Household 0 sees no infection at all.
  # Household 1 sees complete infection, eg. both 5 & 7 infect 1 & 3.
  expect_equal(result$source, c(5, 5, 7, 7))
  expect_equal(result$victim, c(1, 3, 1, 3))
})

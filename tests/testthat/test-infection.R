mob_test("homogeneous_infection_process with p = 0", {
  population <- 0:99
  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)

  rngs <- random_create(length(population))
  result <- infection_list_create()
  count <- homogeneous_infection_process(
    rngs, result,
    bitset_from_vector(length(population), sort(susceptible)),
    bitset_from_vector(length(population), sort(infected)),
    0)

  expect_equal(count, 0)
  expect_equal(nrow(infections_as_dataframe(result)), 0)
  expect_length(bitset_to_vector(infection_victims(result, length(population))), 0)
})

mob_test("homogeneous_infection_process", {
  population <- 0:99
  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)

  rngs <- random_create(length(population))
  result <- infection_list_create()
  homogeneous_infection_process(
    rngs, result,
    bitset_from_vector(length(population), sort(susceptible)),
    bitset_from_vector(length(population), sort(infected)),
    0.1)

  df <- infections_as_dataframe(result)
  expect_in(df$source, infected)
  expect_in(df$victim, susceptible)
  expect_in(bitset_to_vector(infection_victims(result, length(population))), susceptible)
})

mob_test("household_infection_process", {
  population <- 0:99

  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)
  households <- sample.int(50, length(population), replace = TRUE) - 1
  households_partition <- partition_create(50, households)

  rngs <- random_create(length(population))

  result <- infection_list_create()
  household_infection_process(
    rngs, result,
    bitset_from_vector(length(population), sort(susceptible)),
    bitset_from_vector(length(population), sort(infected)),
    households_partition, 0.3)

  df <- infections_as_dataframe(result)
  expect_in(df$source, infected)
  expect_in(df$victim, susceptible)
  expect_equal(households[df$source + 1],
               households[df$victim + 1])
})

mob_test("household_infection_process can use different probabilities per household", {
  households <- partition_create(2, c(0,1,0,1,0,1,0,1))
  probabilities <- c(0, 1)

  susceptible <- bitset_create(8)
  bitset_insert(susceptible, 0:3)

  infected <- bitset_create(8)
  bitset_insert(infected, 4:7)

  rngs <- random_create(8)
  result <- infection_list_create()
  household_infection_process(rngs, result, susceptible, infected, households,
                              probabilities)

  # Household 0 sees no infection at all.
  # Household 1 sees complete infection, eg. both 5 & 7 infect 1 & 3.
  df <- infections_as_dataframe(result)
  expect_equal(df$source, c(5, 5, 7, 7))
  expect_equal(df$victim, c(1, 3, 1, 3))
  expect_equal(bitset_to_vector(infection_victims(result, 8)), c(1, 3))
})

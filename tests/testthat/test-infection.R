mob_test("homogeneous_infection_process with empty I", {
  population <- 1:100
  susceptible <- bitset_from_vector(length(population), population)
  infected <- bitset_create(length(population))

  rngs <- random_create(length(population))
  result <- infection_list_create()

  count <- homogeneous_infection_process(
    rngs, result,
    susceptible, infected,
    0.5)

  expect_equal(count, 0)
  expect_equal(nrow(infections_as_dataframe(result)), 0)
  expect_length(bitset_to_vector(infection_victims(result, length(population))), 0)
})

mob_test("homogeneous_infection_process with p = 0", {
  population <- 1:100
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
  population <- 1:100
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
  population <- 1:100

  susceptible <- sample(population, length(population) / 2)
  infected <- setdiff(population, susceptible)
  households <- sample.int(50, length(population), replace = TRUE)
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
  expect_equal(households[df$source],
               households[df$victim])
})

mob_test("household_infection_process can use different probabilities per household", {
  households <- partition_create(2, c(1,2,1,2,1,2,1,2))
  probabilities <- c(0, 1)

  susceptible <- bitset_create(8)
  bitset_insert(susceptible, 1:4)

  infected <- bitset_create(8)
  bitset_insert(infected, 5:8)

  rngs <- random_create(8)
  result <- infection_list_create()
  household_infection_process(rngs, result, susceptible, infected, households,
                              probabilities)

  # Household 1 sees no infection at all.
  # Household 2 sees complete infection, eg. both 6 & 8 infect 2 & 4.
  df <- infections_as_dataframe(result)
  expect_equal(df$source, c(6, 6, 8, 8))
  expect_equal(df$victim, c(2, 4, 2, 4))
  expect_equal(bitset_to_vector(infection_victims(result, 8)), c(2, 4))
})

mob_test("infection_list", {
  df <- data.frame(
    source = c(1,1,2,2,3,3,4,4),
    victim = c(5,6,5,6,7,8,7,8))
  infections <- infections_from_dataframe(df)
  expect_equal(df, infections_as_dataframe(infections))
})

mob_test("infections_select", {
  rngs <- random_create(4)
  df <- data.frame(
    source = c(1,1,2,2,3,3,4,4),
    victim = c(5,6,5,6,7,8,7,8))

  infections <- infections_from_dataframe(df)
  selected <- infections_as_dataframe(infections_select(rngs, infections))

  # The selected infection contains each victim exactly once
  expect_setequal(selected$victim, unique(df$victim))

  # The source of infection for each victim matches the input dataframe
  join <- dplyr::inner_join(df, selected, by = dplyr::join_by(source, victim))
  expect_equal(nrow(join), nrow(selected))
})

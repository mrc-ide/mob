mob_test("partition_sizes", {
  data <- c(2,2,4,2,0,5)
  p <- partition_create(7, data)

  expect_equal(partition_sizes(p), c(1, 0, 3, 0, 1, 1, 0))
})

mob_test("partition_sizes random", {
  population <- 100
  households <- 20

  assignments <- sample.int(households, population, replace = TRUE)
  p <- partition_create(households, assignments - 1) # partition_create wants 0-based indices

  expect_equal(sum(partition_sizes(p)), population)
  expect_equal(partition_sizes(p), tabulate(assignments, households))
})

mob_test("ragged_vector", {
  data <- list(c(1, 3, 2, 45), c(4, 1), c(0, 6, 1, 7))
  v <- ragged_vector_create(data)

  expect_equal(ragged_vector_get(v, 0), c(1, 3, 2, 45))
  expect_equal(ragged_vector_get(v, 1), c(4, 1))
  expect_equal(ragged_vector_get(v, 2), c(0, 6, 1, 7))

  rngs <- mob:::random_create(3)
  values <- ragged_vector_random_select(rngs, v)
  expect_length(values, 3)
  for (j in seq_along(values)) {
    expect_in(values[[j]], data[[j]])
  }
})

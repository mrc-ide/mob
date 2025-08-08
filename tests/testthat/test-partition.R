mob_test("partition_sizes", {
  data <- c(2,2,4,2,5)
  p <- partition_create(7, integer_vector_create(data))

  expect_equal(partition_sizes(p), c(0, 3, 0, 1, 1, 0, 0))
})

mob_test("partition_sizes random", {
  population <- 100
  households <- 20

  assignments <- sample.int(households, population, replace = TRUE)
  p <- partition_create(households, integer_vector_create(assignments))

  expect_equal(sum(partition_sizes(p)), population)
  expect_equal(partition_sizes(p), tabulate(assignments, households))
})

mob_test("ragged_vector", {
  data <- list(c(1L, 3L, 2L, 45L), c(4L, 1L), c(0L, 6L, 1L, 7L))
  v <- ragged_vector_create(data)

  expect_equal(ragged_vector_get(v, 0), c(1, 3, 2, 45))
  expect_equal(ragged_vector_get(v, 1), c(4, 1))
  expect_equal(ragged_vector_get(v, 2), c(0, 6, 1, 7))

  rngs <- mob:::random_create(3)
  values <- vector_values(ragged_vector_random_select(rngs, v))
  expect_length(values, 3)
  for (j in seq_along(values)) {
    expect_in(values[[j]], data[[j]])
  }
})

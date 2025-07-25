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

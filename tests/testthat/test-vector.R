mob_test("integer vector create", {
  values <- c(9L, 4L, 12L, 0L, 56L)
  v <- integer_vector_create(values)
  expect_identical(integer_vector_values(v), values)
})

mob_test("integer vector scatter", {
  values <- c(9L, 4L, 12L, 0L, 56L)
  v <- integer_vector_create(values)

  integer_vector_scatter(v, c(1L, 3L), c(42L, 67L))

  expect_identical(integer_vector_values(v), c(42L, 4L, 67L, 0L, 56L))
})

mob_test("integer vector bitset scatter", {
  values <- c(9L, 4L, 12L, 0L, 56L)
  v <- integer_vector_create(values)

  integer_vector_scatter_bitset(v, bitset_from_vector(5, c(1L, 3L)), c(42L, 67L))

  expect_identical(integer_vector_values(v), c(42L, 4L, 67L, 0L, 56L))
})

mob_test("integer vector scatter scalar", {
  values <- c(9L, 4L, 12L, 0L, 56L)
  v <- integer_vector_create(values)

  integer_vector_scatter_scalar(v, c(1L, 3L), 42L)

  expect_identical(integer_vector_values(v), c(42L, 4L, 42L, 0L, 56L))
})

mob_test("integer vector gather", {
  values <- c(9L, 4L, 12L, 0L, 56L)
  v <- integer_vector_create(values)

  expect_identical(integer_vector_gather(v, c(3L, 5L, 2L)), c(12L, 56L, 4L))
  expect_identical(integer_vector_gather(v, c(3L, 2L, 2L)), c(12L, 4L, 4L))
  expect_identical(integer_vector_gather(v, integer(0)), integer(0))
})

mob_test("integer vector match", {
  values <- c(9L, 4L, 12L, 0L, 56L, 12L, 3L, 4L)
  v <- integer_vector_create(values)

  expect_identical(integer_vector_match(v, 12L), c(3L, 6L))
  expect_identical(integer_vector_match(v, 56L), 5L)
  expect_identical(integer_vector_match(v, 42L), integer(0))
})

mob_test("integer vector match bitset", {
  values <- c(9L, 4L, 12L, 0L, 56L, 12L, 3L, 4L)
  v <- integer_vector_create(values)

  expect_identical(bitset_to_vector(integer_vector_match_bitset(v, 12L)), c(3L, 6L))
  expect_identical(bitset_to_vector(integer_vector_match_bitset(v, 56L)), 5L)
  expect_identical(bitset_to_vector(integer_vector_match_bitset(v, 42L)), integer(0))
})

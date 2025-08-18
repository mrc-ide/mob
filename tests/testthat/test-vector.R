mob_test("integer vector create", {
  values <- c(9L, 4L, 12L, 0L, 56L)
  v <- integer_vector_create(values)
  expect_identical(vector_values(v), values)
})

mob_test("integer vector scatter", {
  vector <- integer_vector_create(c(9L, 4L, 12L, 0L, 56L))

  vector_scatter(
    vector,
    c(1L, 3L),
    c(42L, 67L))

  expect_identical(vector_values(vector), c(42L, 4L, 67L, 0L, 56L))
})

mob_test("integer vector bitset scatter", {
  vector <- integer_vector_create(c(9L, 4L, 12L, 0L, 56L))

  vector_scatter(
    vector,
    bitset_from_vector(5, c(1L, 3L)),
    c(42L, 67L)
  )

  expect_identical(vector_values(vector),
                   c(42L, 4L, 67L, 0L, 56L))
})

mob_test("integer vector bitset scatter random", {
  v1 <- integer_vector_create(rep(0, 500))
  v2 <- integer_vector_create(rep(0, 500))

  idx <- sample.int(500, 100)
  vs <- sample.int(1000, 100)

  vector_scatter(
    v1,
    bitset_from_vector(500, sort(idx)),
    vs[order(idx)])

  vector_scatter(v2, idx, vs)

  expect_equal(vector_values(v1),
               vector_values(v2))
})

mob_test("integer vector scatter scalar", {
  vector <- integer_vector_create(c(9L, 4L, 12L, 0L, 56L))

  vector_scatter_scalar(vector, c(1L, 3L), 42L)

  expect_identical(vector_values(vector),
                   c(42L, 4L, 42L, 0L, 56L))
})

mob_test("integer vector gather", {
  vector <- integer_vector_create(c(9L, 4L, 12L, 0L, 56L))

  expect_identical(
    vector_values(vector_gather(vector, c(3L, 5L, 2L))),
    c(12L, 56L, 4L))

  expect_identical(
    vector_values(vector_gather(vector, c(3L, 2L, 2L))),
    c(12L, 4L, 4L))

  expect_identical(
    vector_values(vector_gather(vector, integer(0))),
    integer(0))
})

mob_test("integer_vector_match", {
  values <- c(9L, 4L, 12L, 0L, 56L, 12L, 3L, 4L)
  v <- integer_vector_create(values)

  expect_identical(integer_vector_match_eq(v, 12L), c(3L, 6L))
  expect_identical(integer_vector_match_eq(v, 9L), 1L)
  expect_identical(integer_vector_match_eq(v, 42L), integer(0))

  expect_identical(integer_vector_match_gt(v, 9L), c(3L, 5L, 6L))
  expect_identical(integer_vector_match_gt(v, 100L), integer(0))
})

mob_test("integer vector match bitset", {
  values <- c(9L, 4L, 12L, 0L, 56L, 12L, 3L, 4L)
  v <- integer_vector_create(values)

  expect_identical(bitset_to_vector(integer_vector_match_eq_as_bitset(v, 12L)), c(3L, 6L))
  expect_identical(bitset_to_vector(integer_vector_match_eq_as_bitset(v, 9L)), 1L)
  expect_identical(bitset_to_vector(integer_vector_match_eq_as_bitset(v, 42L)), integer(0))

  expect_identical(bitset_to_vector(integer_vector_match_gt_as_bitset(v, 9L)), c(3L, 5L, 6L))
  expect_identical(bitset_to_vector(integer_vector_match_gt_as_bitset(v, 100L)), integer(0))
})

mob_test("integer vector match random", {
  values <- sample.int(200, 150, replace = TRUE)
  v <- integer_vector_create(values)

  for (x in 1L:200L) {
    expect_identical(
      integer_vector_match_eq(v, x),
      which(values == x)
    )
    expect_identical(
      bitset_to_vector(integer_vector_match_eq_as_bitset(v, x)),
      which(values == x)
    )

    expect_identical(
      integer_vector_match_gt(v, x),
      which(values > x)
    )
    expect_identical(
      bitset_to_vector(integer_vector_match_gt_as_bitset(v, x)),
      which(values > x)
    )
  }

  expect_identical(integer_vector_match_eq(v, 0), integer(0))
  expect_identical(integer_vector_match_eq(v, 1000), integer(0))
})

mob_test("integer_vector_add_scalar", {
  values <- sample.int(200, 150, replace = TRUE)
  v <- integer_vector_create(values)

  vector_add_scalar(v, 12L)

  expect_identical(vector_values(v), values + 12L)

  vector_add_scalar(v, -4L)

  expect_identical(vector_values(v), values + 8L)
})

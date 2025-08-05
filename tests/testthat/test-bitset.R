mob_test("bitset_from/to_vector", {
  data <- c(1,4,5)
  b <- bitset_from_vector(10, data)
  expect_equal(bitset_to_vector(b), data)
})

mob_test("bitset_size", {
  expect_equal(bitset_size(bitset_create(10)), 0)
  expect_equal(bitset_size(bitset_from_vector(10, c(1,2,4))), 3)
  expect_equal(bitset_size(bitset_from_vector(10, c(2,4,6,8))), 4)
  expect_equal(bitset_size(bitset_from_vector(10, 1:10)), 10)
})

mob_test("bitset_choose", {
  cap <- 10
  data <- c(2,3,5,7,8)
  rng <- random_create(cap)
  b <- bitset_from_vector(cap, data)

  bitset_choose(b, rng, 4)

  expect_equal(bitset_size(b), 4)
  expect_in(bitset_to_vector(b), data)
})

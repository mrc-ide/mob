mob_test("random_uniform returns a vector of the right size", {
  rngs <- mob:::random_create(1000)
  expect_length(mob:::random_uniform(rngs, 10, 0, 1), 10)
  expect_length(mob:::random_uniform(rngs, 100, 0, 1), 100)
  expect_length(mob:::random_uniform(rngs, 1000, 0, 1), 1000)
  expect_error(mob:::random_uniform(rngs, 10000, 0, 1), "RNG state is too small")
})

test_that("host and system produce identical numbers", {
  skip_on_ci() # No CUDA on CI

  size <- 1e4
  seed <- sample.int(1e9, 1)
  host_values <- withr::with_options(list(mob.system = "host"), {
    rngs <- mob:::random_create(size, seed)
    mob:::random_uniform(rngs, size, min = 0, max = 0)
  })

  device_values <- withr::with_options(list(mob.system = "device"), {
    rngs <- mob:::random_create(size, seed)
    mob:::random_uniform(rngs, size, min = 0, max = 0)
  })

  expect_length(host_values, size)
  expect_length(device_values, size)
  expect_equal(host_values, device_values)
})

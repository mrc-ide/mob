test_that("selection_sampler returns a subset", {
  data <- runif(100)

  sample <- selection_sampler(data, k = 10, seed = 1)
  expect_length(sample, 10)
  expect_contains(data, sample)
})

test_that("betabinomial_sampler returns a subset", {
  data <- runif(100)

  sample <- betabinomial_sampler(data, k = 10, seed = 1)
  expect_length(sample, 10)
  expect_contains(data, sample)
})

test_that("bernoulli_sampler returns a subset", {
  data <- runif(100)

  p <- 0.2
  expected <- bernoulli_sampler_simulate(length(data), p, seed = 1)
  sample <- bernoulli_sampler(data, p, seed = 1)
  expect_length(sample, expected)
  expect_contains(data, sample)
})

test_that("selection_sampler returns a subset", {
  data <- c(4,1,5,9,2)

  sample <- selection_sampler(data, k = 3, seed = 1)
  expect_length(sample, 3)
  expect_contains(data, sample)
})

test_that("betabinomial_sampler returns a subset", {
  data <- c(4,1,5,9,2)

  sample <- betabinomial_sampler(data, k = 3, seed = 1)
  expect_length(sample, 3)
  expect_contains(data, sample)
})

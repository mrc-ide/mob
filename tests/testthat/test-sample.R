mob_test("selection_sampler returns a subset", {
  data <- runif(100)

  sample <- selection_sampler(data, k = 10)
  expect_length(sample, 10)
  expect_contains(data, sample)
})

mob_test("betabinomial_sampler returns a subset", {
  data <- runif(100)

  sample <- betabinomial_sampler(data, k = 10)
  expect_length(sample, 10)
  expect_contains(data, sample)
})

mob_test("bernoulli_sampler returns a subset", {
  data <- runif(100)

  for (p in c(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)) {
    sample <- bernoulli_sampler(data, p)
    expect_contains(data, sample)
  }
})

mob_test("bernoulli_sampler with p=0 returns an empty set", {
  data <- runif(100)
  sample <- bernoulli_sampler(data, p = 0)
  expect_length(sample, 0)
})

mob_test("bernoulli_sampler with p=1 returns entire dataset", {
  data <- runif(100)
  sample <- bernoulli_sampler(data, p = 1)
  expect_equal(sample, data)
})

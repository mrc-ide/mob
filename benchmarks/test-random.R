mob_bench("Initializing dust parallel random number generator", {
  bench::mark({
    mob:::random_create(size)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

mob_bench("Generating uniformly distributed variates", {
  rngs <- mob:::random_create(size)
  bench::mark({
    mob:::random_uniform_benchmark(rngs, size, 0, 1)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))


mob_bench("Generating and copying uniformly distributed variates", {
  rngs <- mob:::random_create(size)
  bench::mark({
    mob:::random_uniform(rngs, size, 0, 1)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

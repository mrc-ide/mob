# R holds references to C++ objects, which in turn own GPU memory.
#
# Unfortunately the R GC doesn't know about memory pressure on the GPU (or in
# the non-R heap) and doesn't trigger garbage collection automatically even
# when we desperately need it. Because of this when allocating GPU memory in a
# tight loop it never gets released and we run out of it pretty quickly.
#
# This is why we have to call gc() in every loop iteration. Ideally we'd
# exclude that from the timing measurement from there does not seem to be an
# option to do this with `bench` Ideally we'd exclude that from the timing
# measurement from there does not seem to be an option to do this with `bench`.

mob_bench("Initializing dust parallel random number generator", {
  bench::mark({
    mob::random_create(size)
    gc()
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

mob_bench("Uniform distribution", {
  rngs <- mob::random_create(size)
  bench::mark({
    mob::random_uniform(rngs, size, 0, 1)
    gc()
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

mob_bench("Binomial distribution", {
  rngs <- mob::random_create(size)
  bench::mark({
    mob::random_binomial(rngs, size, 1000, 0.5)
    gc()
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

mob_bench("Poisson distribution", {
  rngs <- mob::random_create(size)
  bench::mark({
    mob::random_poisson(rngs, size, 0.5)
    gc()
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

mob_bench("Gamma distribution", {
  rngs <- mob::random_create(size)
  bench::mark({
    mob::random_gamma(rngs, size, 5, 1)
    gc()
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7))

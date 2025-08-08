mob_bench("bitset_create", {
  bench::mark({
    mob::bitset_create(size)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8))

mob_bench("bitset_clone", {
  b <- mob::bitset_create(size)
  bench::mark({
    mob::bitset_clone(b)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8))

mob_bench("bitset_or", {
  left <- mob::bitset_create(size)
  right <- mob::bitset_create(size)
  bench::mark({
    b <- mob::bitset_clone(left)
    mob::bitset_or(b, right)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8))


mob_bench("bitset_size", {
  b <- mob::bitset_create(size)
  bench::mark({
    mob::bitset_size(b)
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8))

mob_bench("bitset_invert", {
  b <- mob::bitset_create(size)
  bench::mark({
    mob::bitset_invert(mob::bitset_clone(b))
  }, check = FALSE)
}, size = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8))

mob_bench("bitset_sample", {
    rngs <- mob::random_create(size)
    b <- mob::bitset_create(size)
    mob::bitset_invert(b)
    bench::mark({
      mob::bitset_sample(mob::bitset_clone(b), rngs, prob)
    }, check = FALSE)
  },
  size = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
  prob = 0.5)

mob_bench("bitset_choose", {
    rngs <- mob::random_create(1)
    b <- mob::bitset_create(size)
    mob::bitset_invert(b)
    bench::mark({
      mob::bitset_choose(mob::bitset_clone(b), rngs, size * ratio)
    }, check = FALSE)
  },
  size = c(1e3, 1e4, 1e5, 1e6),
  ratio = 0.5)

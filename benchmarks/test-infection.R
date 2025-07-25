mob_bench("homogeneous_infection_process", {
    rngs <- mob:::random_create(size)

    I <- mob:::bitset_create(size)
    mob:::bitset_invert(I)
    mob:::bitset_choose(I, rngs, size * ratio)

    S <- mob:::bitset_clone(I)
    mob:::bitset_invert(S)

    prob <- 1 - exp(-beta / size)

    bench::mark({
      result <- mob:::infection_list_create()
      mob:::homogeneous_infection_process(rngs, result, S, I, prob)
    }, check = FALSE)
  },
  size = c(1e3, 1e4, 1e5, 1e6, 1e7),
  ratio = 0.1,
  beta = 0.1)

mob_bench("household_infection_process (average size = {house_size})", {
    rngs <- mob:::random_create(size)

    houses <- size %/% 5
    allocation <- sample(1:houses, size, replace=TRUE)
    house_sizes <- tabulate(allocation, nbins = houses)
    expect_equal(sum(house_sizes), size)

    partition <- mob:::partition_create(houses, allocation - 1)

    I <- mob:::bitset_create(size)
    mob:::bitset_invert(I)
    mob:::bitset_choose(I, rngs, size * ratio)

    S <- mob:::bitset_clone(I)
    mob:::bitset_invert(S)

    prob <- 1 - exp(-beta / house_sizes)

    bench::mark({
      result <- mob:::infection_list_create()
      mob:::household_infection_process(rngs, result, S, I, partition, prob)
    }, check = FALSE)
  },
  size = c(1e3, 1e4, 1e5, 1e6, 1e7),
  ratio = 0.1,
  beta = 0.1,
  house_size = c(5, 1000))

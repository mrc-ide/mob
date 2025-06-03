mob_bench("homogeneous_infection_process", {
    rngs <- mob:::random_create(size)
  
    I <- mob:::bitset_create(size)
    mob:::bitset_invert(I)
    mob:::bitset_choose(I, rngs, size * ratio)
  
    S <- mob:::bitset_clone(I)
    mob:::bitset_invert(S)
  
    bench::mark({
      result <- mob:::infection_list_create()
      mob:::homogeneous_infection_process(rngs, result, S, I, prob)
      gc()
    }, check = FALSE)
  },
  size = list(
    host = c(1e3, 1e4, 1e5),
    device = c(1e3, 1e4, 1e5)),
  ratio = 0.1,
  prob = 0.1)

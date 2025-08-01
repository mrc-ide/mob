check_table_invariant <- function(weights) {
  table <- alias_table_create(weights)
  values <- alias_table_values(table)

  expect_true(all(values$probability >= 0 & values$probability <= 1))
  expect_true(all(values$alias >= 0L & values$alias < length(weights)))

  # Figure out how much external contributions each item receives
  extra <- values %>%
    dplyr::group_by(alias) %>%
    dplyr::summarize(extra = sum(1 - probability))

  # Add up an item's own probability with the external contributions
  result <- values %>%
    dplyr::mutate(index = dplyr::row_number() - 1) %>% # adjust for 1 indexing
    dplyr::left_join(extra, by = dplyr::join_by(index == alias)) %>%
    dplyr::mutate(total = probability + replace(extra, is.na(extra), 0))

  # The "total probability" of each item needs to be proportional to
  # its weight.
  expect_equal(result$total / sum(result$total),
               weights / sum(weights))
}

mob_test("alias table invariants", {
  for (i in seq_len(50)) {
    N <- sample.int(100, 1)
    weights <- runif(N)
    check_table_invariant(weights)
  }

  # Some interesting corner cases that probably won't be covered by the
  # randomly generated values.
  check_table_invariant(numeric(0))
  check_table_invariant(c(1,1,1))
  check_table_invariant(c(1,1,1))
  check_table_invariant(c(0,0,1,1))
  check_table_invariant(c(0,2,1,1))
})

mob_test("alias table sample matrix", {
  N <- 100
  weights <- c(0, runif(N))
  table <- alias_table_create(weights)

  rows <- 20
  ks <- sample(0:N, rows, replace = TRUE)
  maxk <- N

  rngs <- mob:::random_create(rows)

  values <- mob:::alias_table_sample_wor_ragged_matrix(table, rngs, ks, maxk)
  expect_equal(dim(values), c(rows, maxk))

  for (i in seq_len(rows)) {
    head <- values[i, seq_len(ks[i])]
    tail <- values[i, -seq_len(ks[i])]
    expect_in(head, 1:N)
    expect_true(all(tail == 0))
  }
})

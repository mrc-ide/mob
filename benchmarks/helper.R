build_grid <- function(...) {
  inputs <- purrr::imap(list(...), function(value, name) {
    if (is.list(value)) {
      dplyr::bind_rows(purrr::imap(value, function(inner, system) {
        tibble::tibble(system = system, !!name := inner)
      }))
    } else {
      tidyr::expand_grid(
        system = c("host", "device"),
        !!name := value)
    }
  })
  purrr::reduce(inputs, dplyr::inner_join,
                by = dplyr::join_by(system),
                relationship = "many-to-many")
}

mob_bench <- function(name, code, ...) {
  reporter <- getOption("mob.benchmark_reporter")
  captured <- rlang::enquo(code)
  testthat::test_that(name, {
    grid <- build_grid(...)

    result <- bench::press({
      # We want to forward all of .data to the captured quosure.
      # rlang's tidy evaluation doesn't seem to have a good way
      # of doing this. Additionally, bench::press doesn't setup
      # the .data pronoun.
      data <- purrr::map(names(grid), function(x) get(x))
      names(data) <- names(grid)

      withr::with_options(list("mob.system" = system), {
        testthat::expect_no_error(rlang::eval_tidy(captured, data))
      })
    }, .grid = grid, .quiet = TRUE)

    if (!is.null(reporter)) {
      reporter$add_result(name, result)
    }
  })
}

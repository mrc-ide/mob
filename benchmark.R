BenchmarkReporter <- R6::R6Class("BenchmarkReporter",
  public = list(
    results = list(),
    add_result = function(desc, grid, data) {
      self$results <- append(self$results, list(list(
        description = desc,
        grid = grid,
        data = data
      )))
    }
  )
)

local_benchmark_reporter <- function(.env = parent.frame()) {
  reporter <- BenchmarkReporter$new()
  withr::local_options("mob.benchmark_reporter" = reporter, .local_envir = .env)
  reporter
}

run_benchmarks <- function(path = ".", ..., load_package = "source") {
  package <- pkgload::pkg_name(path)
  path <- file.path(pkgload::pkg_path(path), "benchmarks")

  reporter <- local_benchmark_reporter()
  testthat::test_dir(path, package = package, ..., load_package = load_package)
  reporter$results
}

library(ggplot2)

# Make sure device is initialized
invisible(mob:::bitset_create(1, system = "device"))

results <- run_benchmarks(load_package = "installed")

expanded_results <- purrr::lmap(results, function(result) {
  result <- result[[1]]
  result$grid %>%
    dplyr::select(-c(system, size)) %>%
    dplyr::distinct() %>%
    dplyr::group_split(dplyr::row_number(), .keep = FALSE) %>%
    purrr::map(function(row) {
      list(
        description = glue::glue_data(row, result$description),
        data = tibble::as_tibble(result$data) %>%
          dplyr::filter((dplyr::if_all(names(row), function(v) {
          v == row[[dplyr::cur_column()]]
        })))
      )
    })
})

plots <- purrr::map(expanded_results, function(result) {
  data <- result$data %>%
    dplyr::select(system, size, time, gc) %>%
    tidyr::unnest_wider(gc) %>%
    tidyr::unnest_longer(c(time, level0, level1, level2)) %>%
    dplyr::filter(level0 == 0, level1 == 0, level2 == 0)

  ggplot(data, aes(x = time, y = system, colour = system)) +
    bench::scale_x_bench_time() +
    facet_grid(rows = vars(size)) +
    ggbeeswarm::geom_quasirandom(orientation = "y") +
    ggtitle(result$description)
})

ggpubr::ggexport(plotlist = plots, filename = "output.pdf")

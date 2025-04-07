list_catch_tests <- function() {
  output <- withr::local_tempfile()
  run_catch(c("--reporter=json", "--list-tests", "--out", output))
  data <- jsonlite::read_json(output, simplifyVector=TRUE)
  data$listings$tests
}

run_catch_tests <- function(..., reporter = NA, out = NA, fork = TRUE) {
  args <- NULL
  if (!is.na(reporter)) {
    args <- c(args, "--reporter", reporter)
  }
  if (!is.na(out)) {
    args <- c(args, "--out", out)
  }
  args <- c(args, ...)
  run_catch(args, fork = fork)
}

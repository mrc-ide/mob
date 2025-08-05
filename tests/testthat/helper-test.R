skip_if_no_cuda <- function() {
  skip_if(
    isTRUE(as.logical(Sys.getenv("SKIP_DEVICE_TESTS", "false"))),
    "device tests are disabled")
}

mob_test <- function(name, code) {
  captured <- rlang::enquo(code)
  patrick::with_parameters_test_that(name, {
    if (system == "device") {
      skip_if_no_cuda()
    }
    withr::with_options(list("mob.system" = system), {
      rlang::eval_tidy(captured)
    })
  }, system = c("host", "device"))
}

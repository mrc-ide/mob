mob_test <- function(name, code) {
  captured <- rlang::enquo(code)
  patrick::with_parameters_test_that(name, {
    if (system == "device") {
      skip_on_ci()
    }
    withr::with_options(list("mob.system" = system), {
      rlang::eval_tidy(captured)
    })
  }, system = c("host", "device"))
}

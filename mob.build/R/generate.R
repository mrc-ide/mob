#' @export
load_device <- function(...) {
  workdir <- withr::local_tempdir()
  mob.build::generate_device(workdir, ...)
  pkgload::load_all(workdir, export_all = FALSE, attach = FALSE)
}

#' @export
install_device <- function(...) {
  workdir <- withr::local_tempdir()
  target <- withr::local_tempdir()
  mob.build::generate_device(workdir, ...)
  path <- pkgbuild::build(workdir, target, needs_compilation = TRUE, compile_attributes = TRUE)

  pkgbuild::with_build_tools(
    required = FALSE,
    callr::rcmd("INSTALL", path, show = TRUE, spinner = FALSE, stderr = "2>&1", fail_on_status = TRUE)
  )

  invisible(NULL)
}

#' @export
generate_file <- function(src, dst, data) {
  path <- system.file(file.path("template", src), package = "mob.build", mustWork = TRUE)
  template <- paste(readLines(path), collapse = "\n")
  output <- glue::glue_data(data, template, .open = "{{", .close = "}}", .trim = FALSE)
  writeLines(output, dst)
}

#' @export
generate_host <- function(path = getwd()) {
  data <- list(system = "host")

  fs::dir_delete(fs::path(path, "src"))
  fs::dir_create(fs::path(path, "src"))
  generate_file("mob_types.h", file.path(path, glue::glue("src/mob_types.h")), data)
  generate_file("Makevars.cpu", file.path(path, "src/Makevars"), data)

  Rcpp::compileAttributes(path)
}

#' @export
generate_device <- function(path, arch = "native") {
  data <- list(
    system = "device",
    arch = arch
  )

  fs::dir_create(path, "src")

  generate_file("DESCRIPTION", file.path(path, "DESCRIPTION"), data)
  generate_file("NAMESPACE", file.path(path, "NAMESPACE"), data)
  generate_file("mob_types.h", file.path(path, glue::glue("src/mob.device_types.h")), data)
  generate_file("Makevars.cuda", file.path(path, "src/Makevars"), data)

  Rcpp::compileAttributes(path)
}

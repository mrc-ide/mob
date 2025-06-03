install.packages("pak", repos = sprintf("https://r-lib.github.io/p/pak/stable/%s/%s/%s", .Platform$pkgType, R.Version()$os, R.Version()$arch))
pak::lockfile_create(
  c("deps::.", "rcmdcheck"),
  lockfile = ".buildkite/pkg.lock",
  dependencies = TRUE)
pak::lockfile_install(".buildkite/pkg.lock")

local({
  # We run the Catch2 tests and store the results as a JUnit XML file. We can
  # then parse the report and feed its results into testthat. This could
  # probably be used with any test framework that can produce results in the
  # JUnit XML format.
  #
  # testthat actually has a similar implementation of this, but it uses a very
  # old version of Catch and it is a bit buggy.
  #
  # Catch tests have a peculiar structure, allowing arbitrary nesting of
  # sections. If we were to use the native report format (with
  # `--reporter=xml`) we'd have to deal with that ourselves and flatten it to
  # something testthat understands. By using the JUnit reporter we can make
  # Catch do the heavy lifting for us.
  #
  # This does not support streaming of results, which would allow the testthat
  # console to print which test is being executed in real-time. Instead we have
  # to wait until the end before reporting the results. This is both a flaw in
  # the JUnit reporter but also much simpler to implement on our end. We could
  # either write a native testthat reporter for Catch or use a different
  # existing reporter which supports streaming (eg. TAP).

  output <- withr::local_tempfile()
  success <- run_catch_tests(c("--reporter=junit", "--out", output))

  document <- xml2::read_xml(output)

  reporter <- testthat::get_reporter()

  for (testsuite in xml2::xml_find_all(document, "./testsuite")) {
    for (testcase in xml2::xml_find_all(testsuite, "./testcase")) {
      name <- xml2::xml_attr(testcase, "name")

      reporter$start_context(name)
      failure <- xml2::xml_find_first(testcase, "./failure")
      if (!is.na(failure)) {
        message <- xml2::xml_text(failure)
        reporter$add_result(context = name, test = name, result = expectation("failure", message))
      } else {
        reporter$add_result(context = name, test = name, result = expectation("success", ""))
      }
      reporter$end_context(name)
    }
  }
})

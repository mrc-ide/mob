#define CATCH_CONFIG_RUNNER
#include <Rcpp.h>
#include <catch2/catch_session.hpp>
#include <sys/wait.h>
#include <unistd.h>

// [[Rcpp::export]]
bool run_catch_tests(Rcpp::Nullable<Rcpp::StringVector> args = R_NilValue) {
  // Forking here helps protect the R session from buggy native code.
  pid_t pid = fork();
  if (pid < 0) {
    perror("cannot fork");
    return false;
  } else if (pid == 0) {
    std::vector<const char *> argv;
    argv.push_back("catch");

    if (args.isNotNull()) {
      for (auto s : args.as()) {
        argv.push_back(s);
      }
    }

    Catch::Session instance;
    exit(instance.run(argv.size(), argv.data()));
  } else {
    int status;
    pid_t err = waitpid(pid, &status, 0);
    if (err < 0) {
      perror("cannot wait for child process");
      return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
}

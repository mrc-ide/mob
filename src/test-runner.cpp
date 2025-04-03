#define CATCH_CONFIG_RUNNER
#include <Rcpp.h>
#include <catch2/catch_session.hpp>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

static int run_catch_tests_internal(Rcpp::Nullable<Rcpp::StringVector> args) {
  std::vector<const char *> argv;
  argv.push_back("catch");

  if (args.isNotNull()) {
    for (auto s : args.as()) {
      argv.push_back(s);
    }
  }

  // Catch crashes if you instantiate multiple Session objects in the same
  // process.
  static Catch::Session instance;
  return instance.run(argv.size(), argv.data());
}

static void reset_signal_handlers() {
  // Annoyingly, fork inherits all of the parent's signal dispositions,
  // including whatever R would have setup. It would be nice to use `clone3()`
  // with CLONE_CLEAR_SIGHAND, but using the clone API is non-trivial though,
  // and clone3 doesn't even have a glibc wrapper yet.
  signal(SIGABRT, SIG_DFL);
  signal(SIGALRM, SIG_DFL);
  signal(SIGBUS, SIG_DFL);
  signal(SIGCHLD, SIG_DFL);
  signal(SIGCONT, SIG_DFL);
  signal(SIGFPE, SIG_DFL);
  signal(SIGHUP, SIG_DFL);
  signal(SIGILL, SIG_DFL);
  signal(SIGINT, SIG_DFL);
  signal(SIGPIPE, SIG_DFL);
  signal(SIGPOLL, SIG_DFL);
  signal(SIGPROF, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
  signal(SIGSEGV, SIG_DFL);
  signal(SIGSTOP, SIG_DFL);
  signal(SIGTSTP, SIG_DFL);
  signal(SIGSYS, SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGTRAP, SIG_DFL);
  signal(SIGTTIN, SIG_DFL);
  signal(SIGTTOU, SIG_DFL);
  signal(SIGURG, SIG_DFL);
  signal(SIGUSR1, SIG_DFL);
  signal(SIGUSR2, SIG_DFL);
  signal(SIGVTALRM, SIG_DFL);
  signal(SIGXCPU, SIG_DFL);
  signal(SIGXFSZ, SIG_DFL);
  signal(SIGWINCH, SIG_DFL);
}

template <typename Fn>
static bool run_test_process(Fn &&fn) {
  pid_t pid = fork();
  if (pid < 0) {
    perror("cannot fork");
    return false;
  } else if (pid == 0) {
    // Replace stdin with /dev/null
    int fd = open("/dev/null", O_RDWR);
    if (fd >= 0) {
      dup2(fd, STDIN_FILENO);
      if (fd != STDIN_FILENO) {
        close(fd);
      }
    }
    reset_signal_handlers();
    exit(std::forward<Fn>(fn)());
  } else {
    int status;
    pid_t err = waitpid(pid, &status, 0);
    if (err < 0) {
      perror("cannot wait for child process");
      return false;
    }
    if (WIFSIGNALED(status)) {
      int signal = WTERMSIG(status);
      fprintf(stderr, "Test process was killed by signal %s (%d)\n",
              sigabbrev_np(signal), signal);
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
}

// [[Rcpp::export(invisible = true)]]
bool run_catch_tests(Rcpp::Nullable<Rcpp::StringVector> args = R_NilValue,
                     bool fork = true) {
  if (fork) {
    // Forking here helps protect the R session from buggy tests that would
    // crash the process. It can however make running a debugger a little more
    // difficult, so we have a parameter to skip it.
    return run_test_process([&] { return run_catch_tests_internal(args); });
  } else {
    return run_catch_tests_internal(args) == 0;
  }
}

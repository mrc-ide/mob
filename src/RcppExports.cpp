// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// parallel_runif
Rcpp::NumericVector parallel_runif(size_t n, double min, double max, int seed);
RcppExport SEXP _mob_parallel_runif(SEXP nSEXP, SEXP minSEXP, SEXP maxSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< size_t >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type min(minSEXP);
    Rcpp::traits::input_parameter< double >::type max(maxSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(parallel_runif(n, min, max, seed));
    return rcpp_result_gen;
END_RCPP
}
// parallel_rbinom
Rcpp::NumericVector parallel_rbinom(size_t n, size_t size, double prob, int seed);
RcppExport SEXP _mob_parallel_rbinom(SEXP nSEXP, SEXP sizeSEXP, SEXP probSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< size_t >::type n(nSEXP);
    Rcpp::traits::input_parameter< size_t >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< double >::type prob(probSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(parallel_rbinom(n, size, prob, seed));
    return rcpp_result_gen;
END_RCPP
}
// betabinomial_sampler_wrapper
Rcpp::NumericVector betabinomial_sampler_wrapper(Rcpp::NumericVector data, size_t k, int seed);
RcppExport SEXP _mob_betabinomial_sampler_wrapper(SEXP dataSEXP, SEXP kSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type data(dataSEXP);
    Rcpp::traits::input_parameter< size_t >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(betabinomial_sampler_wrapper(data, k, seed));
    return rcpp_result_gen;
END_RCPP
}
// selection_sampler_wrapper
Rcpp::NumericVector selection_sampler_wrapper(Rcpp::NumericVector data, size_t k, int seed);
RcppExport SEXP _mob_selection_sampler_wrapper(SEXP dataSEXP, SEXP kSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type data(dataSEXP);
    Rcpp::traits::input_parameter< size_t >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(selection_sampler_wrapper(data, k, seed));
    return rcpp_result_gen;
END_RCPP
}
// bernouilli_sampler_wrapper
std::vector<double> bernouilli_sampler_wrapper(Rcpp::NumericVector data, double p, int seed);
RcppExport SEXP _mob_bernouilli_sampler_wrapper(SEXP dataSEXP, SEXP pSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type data(dataSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(bernouilli_sampler_wrapper(data, p, seed));
    return rcpp_result_gen;
END_RCPP
}
// bernouilli_sampler_simulate
size_t bernouilli_sampler_simulate(size_t n, double p, int seed);
RcppExport SEXP _mob_bernouilli_sampler_simulate(SEXP nSEXP, SEXP pSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< size_t >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(bernouilli_sampler_simulate(n, p, seed));
    return rcpp_result_gen;
END_RCPP
}
// run_catch_tests
bool run_catch_tests(Rcpp::Nullable<Rcpp::StringVector> args);
RcppExport SEXP _mob_run_catch_tests(SEXP argsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type args(argsSEXP);
    rcpp_result_gen = Rcpp::wrap(run_catch_tests(args));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mob_parallel_runif", (DL_FUNC) &_mob_parallel_runif, 4},
    {"_mob_parallel_rbinom", (DL_FUNC) &_mob_parallel_rbinom, 4},
    {"_mob_betabinomial_sampler_wrapper", (DL_FUNC) &_mob_betabinomial_sampler_wrapper, 3},
    {"_mob_selection_sampler_wrapper", (DL_FUNC) &_mob_selection_sampler_wrapper, 3},
    {"_mob_bernouilli_sampler_wrapper", (DL_FUNC) &_mob_bernouilli_sampler_wrapper, 3},
    {"_mob_bernouilli_sampler_simulate", (DL_FUNC) &_mob_bernouilli_sampler_simulate, 3},
    {"_mob_run_catch_tests", (DL_FUNC) &_mob_run_catch_tests, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_mob(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

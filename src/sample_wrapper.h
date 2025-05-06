#include <mob/bernoulli.h>
#include <mob/parallel_random.h>
#include <mob/sample.h>
#include <mob/system.h>

#include <Rcpp.h>
#include <dust/random/binomial.hpp>
#include <dust/random/uniform.hpp>

template <typename System>
Rcpp::NumericVector
selection_sampler_wrapper(Rcpp::NumericVector data, size_t k,
                          Rcpp::Nullable<Rcpp::NumericVector> seed) {
  auto rng = mob::make_unique<System, dust::random::xoroshiro128plus>(
      dust::random::seed<dust::random::xoroshiro128plus>(from_seed(seed)));
  auto rng_ptr = rng.get();

  mob::vector<System, double> input{data.begin(), data.end()};
  mob::vector<System, double> result(k);

  mob::ds::span input_view(input);
  mob::ds::span result_view(result);

  mob::execute<System>([=] __device__ __host__ {
    mob::selection_sampler(thrust::raw_reference_cast(*rng_ptr),
                           input_view.begin(), input_view.end(),
                           result_view.begin(), result_view.end());
  });

  return {result.begin(), result.end()};
}

template <typename System>
Rcpp::NumericVector
betabinomial_sampler_wrapper(Rcpp::NumericVector data, size_t k,
                             Rcpp::Nullable<Rcpp::NumericVector> seed) {
  auto rng = mob::make_unique<System, dust::random::xoroshiro128plus>(
      dust::random::seed<dust::random::xoroshiro128plus>(from_seed(seed)));
  auto rng_ptr = rng.get();

  mob::vector<System, double> input{data.begin(), data.end()};
  mob::vector<System, double> result(k);

  mob::ds::span input_view(input);
  mob::ds::span result_view(result);

  mob::execute<System>([=] __host__ __device__ {
    mob::betabinomial_sampler(thrust::raw_reference_cast(*rng_ptr),
                              input_view.begin(), input_view.end(),
                              result_view.begin(), result_view.end());
  });

  return {result.begin(), result.end()};
}

template <typename System>
Rcpp::NumericVector
bernoulli_sampler_wrapper(Rcpp::NumericVector data, double p,
                          Rcpp::Nullable<Rcpp::NumericVector> seed) {
  auto rng = mob::make_unique<System, dust::random::xoroshiro128plus>(
      dust::random::seed<dust::random::xoroshiro128plus>(from_seed(seed)));
  mob::vector<System, double> input(data.begin(), data.end());

  auto rng_ptr = rng.get();
  mob::ds::span input_view(input);

  auto count = mob::execute<System>([=] __device__ __host__() -> size_t {
    dust::random::xoroshiro128plus rng_copy = *rng_ptr;
    return cuda::std::ranges::distance(mob::bernoulli(input_view, p, rng_copy));
  });

  mob::vector<System, double> result(count);
  auto result_begin = result.begin();
  mob::execute<System>([=] __device__ __host__ {
    mob::compat::copy(
        mob::bernoulli(input_view, p, thrust::raw_reference_cast(*rng_ptr)),
        result_begin);
  });

  return {result.begin(), result.end()};
}

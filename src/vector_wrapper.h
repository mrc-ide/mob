#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

template <typename System>
Rcpp::XPtr<mob::vector<System, size_t>>
integer_vector_create(Rcpp::IntegerVector values) {
  return Rcpp::XPtr(
      new mob::vector<System, size_t>(values.begin(), values.end()));
}

template <typename System>
Rcpp::IntegerVector
integer_vector_values(Rcpp::XPtr<mob::vector<System, size_t>> v) {
  return asRcppVector(*v);
}

template <typename System>
void integer_vector_scatter(Rcpp::XPtr<mob::vector<System, size_t>> v,
                            Rcpp::IntegerVector indices,
                            Rcpp::IntegerVector values) {
  if (indices.size() != values.size()) {
    Rcpp::stop("argument sizes mismatch: %d != %d", indices.size(),
               values.size());
  }
  for (size_t i : indices) {
    if (i >= v->size()) {
      Rcpp::stop("index out-of-bound: %d >= %d", i, v->size());
    }
  }

  mob::vector<System, size_t> indices_v(indices.begin(), indices.end());
  mob::vector<System, size_t> values_v(values.begin(), values.end());

  thrust::scatter(values_v.begin(), values_v.end(), indices_v.begin(),
                  v->begin());
}

template <typename System>
void integer_vector_scatter_scalar(Rcpp::XPtr<mob::vector<System, size_t>> v,
                                   Rcpp::IntegerVector indices, size_t value) {
  for (size_t i : indices) {
    if (i >= v->size()) {
      Rcpp::stop("index out-of-bound: %d >= %d", i, v->size());
    }
  }

  mob::vector<System, size_t> indices_v(indices.begin(), indices.end());

  thrust::scatter(
      thrust::constant_iterator<size_t, size_t>(value, 0),
      thrust::constant_iterator<size_t, size_t>(value, indices_v.size()),
      indices_v.begin(), v->begin());
}

template <typename System>
Rcpp::IntegerVector
integer_vector_gather(Rcpp::XPtr<mob::vector<System, size_t>> v,
                      Rcpp::IntegerVector indices) {
  for (size_t i : indices) {
    if (i >= v->size()) {
      Rcpp::stop("index out-of-bound: %d >= %d", i, v->size());
    }
  }

  mob::vector<System, size_t> indices_v(indices.begin(), indices.end());
  mob::vector<System, size_t> result(indices.size());

  thrust::gather(indices_v.begin(), indices_v.end(), v->begin(),
                 result.begin());

  return asRcppVector(result);
}

template <typename System>
Rcpp::IntegerVector
integer_vector_match(Rcpp::XPtr<mob::vector<System, size_t>> v, size_t value) {
  mob::vector<System, size_t> result(v->size());

  auto last = thrust::copy_if(
      thrust::counting_iterator<size_t>(0),
      thrust::counting_iterator<size_t>(v->size()), v->begin(), result.begin(),
      [value] __host__ __device__(size_t x) { return x == value; });

  result.erase(last, result.end());

  return asRcppVector(result);
}

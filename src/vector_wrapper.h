#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

template <typename System>
Rcpp::XPtr<mob::vector<System, size_t>>
integer_vector_create(Rcpp::IntegerVector values) {
  auto v = fromRcppVector<System, size_t, ConvertIndex::No>(values);
  return Rcpp::XPtr(new mob::vector<System, size_t>(std::move(v)));
}

template <typename System>
Rcpp::IntegerVector
integer_vector_values(Rcpp::XPtr<mob::vector<System, size_t>> v) {
  return asRcppVector<ConvertIndex::No>(*v);
}

template <typename System>
void integer_vector_scatter_bitset(Rcpp::XPtr<mob::vector<System, size_t>> v,
                                   Rcpp::XPtr<mob::bitset<System>> indices,
                                   Rcpp::IntegerVector values) {
  mob::bitset_view bs(*indices);
  if (bs.size() != static_cast<size_t>(values.size())) {
    Rcpp::stop("argument sizes mismatch: %d != %d", bs.size(), values.size());
  }
  if (bs.capacity() != v->size()) {
    Rcpp::stop("bitset capacity does not match target vector size: %d != %d",
               indices->capacity(), v->size());
  }

  auto values_v = fromRcppVector<System, size_t, ConvertIndex::No>(values);

  bs.scatter(v->begin(), values_v.begin());
}

template <typename System>
void integer_vector_scatter(Rcpp::XPtr<mob::vector<System, size_t>> v,
                            Rcpp::IntegerVector indices,
                            Rcpp::IntegerVector values) {
  if (indices.size() != values.size()) {
    Rcpp::stop("argument sizes mismatch: %d != %d", indices.size(),
               values.size());
  }
  checkIndices(indices, v->size());

  auto indices_v = fromRcppVector<System, size_t, ConvertIndex::Yes>(indices);
  auto values_v = fromRcppVector<System, size_t, ConvertIndex::No>(values);

  thrust::scatter(values_v.begin(), values_v.end(), indices_v.begin(),
                  v->begin());
}

template <typename System>
void integer_vector_scatter_scalar(Rcpp::XPtr<mob::vector<System, size_t>> v,
                                   Rcpp::IntegerVector indices, size_t value) {
  checkIndices(indices, v->size());

  auto indices_v = fromRcppVector<System, size_t, ConvertIndex::Yes>(indices);

  thrust::scatter(
      thrust::constant_iterator<size_t, size_t>(value, 0),
      thrust::constant_iterator<size_t, size_t>(value, indices_v.size()),
      indices_v.begin(), v->begin());
}

template <typename System>
Rcpp::IntegerVector
integer_vector_gather(Rcpp::XPtr<mob::vector<System, size_t>> v,
                      Rcpp::IntegerVector indices) {
  checkIndices(indices, v->size());

  auto indices_v = fromRcppVector<System, size_t, ConvertIndex::Yes>(indices);
  mob::vector<System, size_t> result(indices.size());

  thrust::gather(indices_v.begin(), indices_v.end(), v->begin(),
                 result.begin());

  return asRcppVector<ConvertIndex::No>(std::move(result));
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

  return asRcppVector<ConvertIndex::Yes>(std::move(result));
}

template <typename System>
Rcpp::XPtr<mob::bitset<System>>
integer_vector_match_bitset(Rcpp::XPtr<mob::vector<System, size_t>> v,
                            size_t value) {
  size_t capacity = v->size();

  auto result = Rcpp::XPtr(new mob::bitset<System>(capacity));

  using word_type = mob::bitset<System>::word_type;
  constexpr size_t num_bits = mob::bitset<System>::num_bits;
  auto buckets = result->data();
  auto input = v->data();

  thrust::for_each_n(
      thrust::make_zip_iterator(thrust::counting_iterator<size_t>(0),
                                buckets.begin()),
      buckets.size(),
      thrust::make_zip_function([input, value, capacity] __host__ __device__(
                                    size_t i, word_type &out) {
        word_type word = 0;
        for (size_t j = 0; j < 64; j++) {
          size_t offset = num_bits * i + j;
          if (offset < capacity && input[offset] == value) {
            word |= 1 << j;
          }
          out = word;
        }
      }));

  return result;
}

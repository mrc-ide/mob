#pragma once

#include <dust/random/prng.hpp>
#include <dust/random/xoroshiro128.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace mob {

/**
 * A container for dust random states, optimized for GPU programming.
 *
 * The PRNG state is stored interleaved to optimize memory accesses on
 * neighbouring GPU threads.
 *
 * By default, all random state is kept in GPU memory. The class can be
 * instatiated with `std::vector` or `thrust::host_vector` instead to use on
 * the CPU. This is useful for testing purposes, but the interleaved memory
 * layout will probably have an adverse performance effect.
 *
 * The container offers iterators that are designed to work together with
 * thrust's algorithms. For example the following example will produce N
 * uniformly distributed real numbers, in parallel:
 *
 *    device_random<> rngs(N);
 *    thrust::device_vector<double> result(N);
 *    thrust::transform(
 *      rngs.begin(), rngs.end(), result.begin(),
 *      [] __device__ (auto& rng) {
 *        return dust::random::random_real<double>(rng);
 *      });
 *
 *  The `rng` object above is a proxy object that can be used in any
 *  dust::random function, and will use the interleaved random state and the
 *  underlying PRNG algorithm.
 *
 * TODO: provide methods for use without thrust, eg. direct access to the i-th
 * RNG stream.
 */
template <template <typename> typename Vector,
          typename T = dust::random::xoroshiro128plus>
struct parallel_random {
  using rng_state = T;
  using vector_type = Vector<typename rng_state::int_type>;

  static constexpr size_t width = rng_state::size();

  parallel_random(size_t capacity, int seed = 0)
      : data(capacity * width), capacity(capacity) {
    dust::random::prng<rng_state> states(capacity, seed);

    // TODO: use something like cudamemcpy2d, which should be able to do strided
    // copies.
    for (size_t i = 0; i < capacity; i++) {
      rng_state rng = states.state(i);
      for (size_t j = 0; j < width; j++) {
        data[i + j * capacity] = rng.state[j];
      }
    }
  }

  struct proxy;
  struct iterator;

  iterator begin() {
    return iterator(data.begin(), capacity);
  }

  iterator end() {
    return iterator(data.begin() + capacity, capacity);
  }

  struct proxy {
    using int_type = typename rng_state::int_type;
    static constexpr bool deterministic = false;

    __host__ __device__ proxy(typename vector_type::pointer ptr, size_t stride)
        : ptr(ptr), stride(stride) {}

    __host__ __device__ rng_state get() const {
      rng_state state;
      for (size_t j = 0; j < width; j++) {
        state[j] = ptr[j * stride];
      }
      return state;
    }

    __host__ __device__ void put(rng_state state) {
      for (size_t j = 0; j < width; j++) {
        ptr[j * stride] = state[j];
      }
    }

    // Used for ADL by dust
    friend __host__ __device__ auto next(proxy p) {
      auto state = p.get();
      auto value = next(state);
      p.put(state);
      return value;
    }

  private:
    size_t stride;
    typename vector_type::pointer ptr;
  };

  struct iterator : public thrust::iterator_adaptor<
                        /* Derived */ iterator,
                        /* Base */ typename vector_type::iterator,
                        /* Value */ proxy,
                        /* System */ thrust::use_default,
                        /* Traversal */ thrust::use_default,
                        /* Reference */ proxy> {
    using super_t =
        thrust::iterator_adaptor<iterator, typename vector_type::iterator,
                                 proxy, thrust::use_default,
                                 thrust::use_default, proxy>;

    iterator(typename vector_type::iterator underlying, size_t stride)
        : super_t(underlying), stride(stride) {}

    friend class thrust::iterator_core_access;

  private:
    __host__ __device__ proxy dereference() const {
      return proxy(this->base().base(), stride);
    }

    size_t stride;
  };

private:
  vector_type data;
  size_t capacity;
};

using device_random =
    parallel_random<thrust::device_vector, dust::random::xoroshiro128plus>;

using host_random =
    parallel_random<thrust::host_vector, dust::random::xoroshiro128plus>;

} // namespace mob

#pragma once

#include <mob/random.h>
#include <mob/system.h>

#include <cuda/std/array>
#include <dust/random/prng.hpp>
#include <dust/random/xoroshiro128.hpp>
#include <thrust/execution_policy.h>

namespace mob {

// For a generator with period 2^N, this table contains the jump polynomials for
// 2^(N/2) up to 2^(N/2+31).
//
// The numbers were generated using https://github.com/nessan/xoshiro and
// https://gist.github.com/plietar/ab99d2a6fec65dc20a438c712a8a703b
constexpr __device__ __host__
    cuda::std::array<cuda::std::array<uint64_t, 2>, 32>
        xoroshiro128plus_jump_table = {{
            {0xdf900294d8f554a5, 0x170865df4b3201fc},
            {0x2992ead4972eaed2, 0xb2a7b279a8cb1f50},
            {0xc026a7d9e04a7700, 0xe7859c665be57882},
            {0xb4cb6197dea2b1fe, 0x4b4a7aa8c389701c},
            {0x0dcfc5b909e7df4d, 0xadb7753d55646eef},
            {0x468431669864f789, 0xc80926301806a352},
            {0x22b6c1736285fcc8, 0xc05da051ec96af1d},
            {0x74c1daac8729d8bb, 0xf88f6bac8fd30448},
            {0x847757c126b23e45, 0x752b98d002c408f7},
            {0x0f9eaa62d0c9e2a3, 0x1aa7bc96dbace110},
            {0x7475d71b98314377, 0xc469b29353a4984b},
            {0xbbb7d266d61c85ea, 0x4b6dd41bce3bb499},
            {0xc419b3742570e16f, 0xe023777e70b3a2f8},
            {0x2a71db3a3ce8b968, 0x131e94fb35203d80},
            {0x2897bb8961b4dce9, 0x9240c95b1e7fa08b},
            {0xf0fc3553d7881d5f, 0xb879fca0915f893f},
            {0xe754db3fbc7536bc, 0x2adca86fbefe1366},
            {0x0a9e201adfe7baa9, 0x0a40a688d77855ba},
            {0x1d0d601e49c35837, 0x17771c905e0775a8},
            {0x9b031395aec7b584, 0x2cf775e419a607e0},
            {0x79ead2eeddf66699, 0x93a7cf27dec9b306},
            {0xe1b9805c107679fc, 0x93615189fe85b7d5},
            {0x2c3925dcd790e3d6, 0x466421124b50fbfb},
            {0xdca9b0fa4e95600e, 0x1cda7bd04e3bb94b},
            {0xefc7905e1cbb5ffb, 0x5ec431d73bbfe49f},
            {0x854414811d534483, 0x31a1f85fd532f302},
            {0xadb9ba2958f30b6e, 0xed9b991c09177e2f},
            {0x76f8fdf26b0d1cbb, 0x38d9e87dffdfca70},
            {0x51f21cddcebdb8c7, 0xd8e9e7254052af4d},
            {0xa03f796efb295305, 0x62769780d13fbc08},
        }};

// This comes from dust, but isn't marked as __device__ there
template <typename T>
__host__ __device__ void
jump_polynomial(T &state,
                cuda::std::array<typename T::int_type, T::size()> coef) {
  using int_type = typename T::int_type;
  constexpr auto N = T::size();
  int_type work[N] = {};                     // enforced zero-initialisation
  constexpr int bits = sizeof(int_type) * 8; // bit_size<int_type>();
  for (size_t i = 0; i < N; ++i) {
    for (int b = 0; b < bits; b++) {
      if (coef[i] & static_cast<int_type>(1) << b) {
        for (size_t j = 0; j < N; ++j) {
          work[j] ^= state[j];
        }
      }
      next(state);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    state[i] = work[i];
  }
}

/**
 * Perform the equivalent of n jump calls.
 */
__host__ __device__ void jump_n(dust::random::xoroshiro128plus &rng,
                                uint32_t n) {
  for (size_t i = 0; i < 32; i++) {
    if (n & static_cast<size_t>(1) << i) {
      jump_polynomial(rng, xoroshiro128plus_jump_table[i]);
    }
  }
}

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
template <typename System,
          random_state_storage T = dust::random::xoroshiro128plus>
struct parallel_random {
  using rng_state = T;
  using int_type = typename rng_state::int_type;
  using vector_type = mob::vector<System, int_type>;

  static constexpr size_t width = rng_state::size();

  parallel_random(size_t size, int seed = 0)
      : size_(size), data_(size * width) {
    auto initial = dust::random::seed<rng_state>(seed);

    populate(initial, thrust::iterator_system_t<iterator>{});
  }

  void populate(rng_state state, thrust::host_system_tag) {
    for (auto it = begin(); it != end(); it++) {
      (*it).put(state);
      dust::random::jump(state);
    }
  }

  // When running on the GPU, each state is initialized independently and
  // concurrently. It uses the offset of each state `n` as the argument to
  // `jump_n`. `jump_n` is implemented by doing repeated 2^k sized jumps, where
  // k are the set bits of `n`.
  //
  // There is some amount of repeated work here among the different threads,
  // which is fine since it is parallelized but could be improved by sharing
  // the initial computations, especially when size >>> number of concurrent
  // GPU threads. I cannot find a suitable abstraction in thrust, nor can I
  // figure out what it should be called. This is similar, but not quite the
  // same, as a prefix sum. Or possibly some kind of tree traversal.
  void populate(const rng_state &initial, thrust::device_system_tag) {
    if (size_ >= std::numeric_limits<uint32_t>::max()) {
      throw std::logic_error("Maximum size is 32 bits");
    }

    // Serialize the initial state so it can be captured onto the GPU more
    // easily.
    cuda::std::array<int_type, width> initial_data;
    std::copy_n(initial.state, width, initial_data.begin());

    iterator output = begin();
    thrust::for_each(thrust::device, thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(size_),
                     [=] __device__(size_t n) {
                       rng_state state;
                       for (size_t i = 0; i < width; i++) {
                         state.state[i] = initial_data[i];
                       }
                       jump_n(state, n);
                       output[n].put(state);
                     });
  }

  struct iterator;

  iterator begin() {
    return iterator(data_.begin(), size_);
  }

  iterator end() {
    return iterator(data_.begin() + size_, size_);
  }

  struct proxy_base {
    using int_type = typename rng_state::int_type;
    static constexpr bool deterministic = false;

    __host__ __device__ rng_state get() const {
      rng_state state;
      for (size_t j = 0; j < width; j++) {
        state[j] = ptr[j * stride];
      }
      return state;
    }

    __host__ __device__ void put(rng_state state) const {
      for (size_t j = 0; j < width; j++) {
        ptr[j * stride] = state[j];
      }
    }

    // Used for ADL by dust
    friend __host__ __device__ auto next(const proxy_base &p) {
      auto state = p.get();
      auto value = next(state);
      p.put(state);
      return value;
    }

    friend iterator;

  private:
    __host__ __device__ proxy_base(typename vector_type::pointer ptr,
                                   size_t stride)
        : ptr(ptr), stride(stride) {}

    typename vector_type::pointer ptr;
    size_t stride;
  };

  using proxy = const proxy_base;

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

    iterator() = default;
    iterator(typename vector_type::iterator underlying, size_t stride)
        : super_t(underlying), stride(stride) {}

    friend class thrust::iterator_core_access;

  private:
    __host__ __device__ proxy dereference() const {
      return proxy(this->base().base(), stride);
    }

    size_t stride = 0;
  };

  static_assert(std::random_access_iterator<iterator>);

  size_t size() {
    return size_;
  }

private:
  size_t size_;
  vector_type data_;
};

template <typename System>
using random_proxy = parallel_random<System>::proxy;

using host_random =
    parallel_random<mob::system::host, dust::random::xoroshiro128plus>;

static_assert(std::ranges::random_access_range<host_random>);
static_assert(random_state<host_random::proxy>);

using device_random =
    parallel_random<mob::system::device, dust::random::xoroshiro128plus>;

#ifdef __NVCC__
static_assert(std::ranges::random_access_range<device_random>);
static_assert(random_state<device_random::proxy>);
#endif

} // namespace mob

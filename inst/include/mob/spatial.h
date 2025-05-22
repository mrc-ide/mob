#pragma once

#include <mob/compat/views.h>
#include <mob/ds/partition.h>
#include <mob/ds/span.h>
#include <mob/infection.h>
#include <mob/parallel_random.h>
#include <mob/system.h>

#include <thrust/transform_reduce.h>

namespace mob {

struct point {
  double x;
  double y;
};

template <typename T>
__host__ __device__ T abs_diff(T x, T y) {
  return (x < y) ? (y - x) : (x - y);
}

struct rect {
  point start;
  point end;

  __host__ __device__ rect() : start({0, 0}), end({0, 0}) {}
  __host__ __device__ rect(point p) : start(p), end(p) {}
  __host__ __device__ rect(point start, point end) : start(start), end(end) {}

  __host__ __device__ double xmin() const {
    return start.x;
  }
  __host__ __device__ double ymin() const {
    return start.y;
  }
  __host__ __device__ double xmax() const {
    return end.x;
  }
  __host__ __device__ double ymax() const {
    return end.y;
  }
  __host__ __device__ double width() const {
    return end.x - start.x;
  }
  __host__ __device__ double height() const {
    return end.y - start.y;
  }
};

__host__ __device__ rect operator|(const rect &r1, const rect &r2) {
  return {{cuda::std::min(r1.xmin(), r2.xmin()),
           cuda::std::min(r1.ymin(), r2.ymin())},
          {cuda::std::max(r1.xmax(), r2.xmax()),
           cuda::std::max(r1.ymax(), r2.ymax())}};
}

template <typename System>
struct spatial {
  mob::vector<System, double> x;
  mob::vector<System, double> y;
};

template <typename System>
struct spatial_view {
  mob::ds::span<System, const double> x;
  mob::ds::span<System, const double> y;
  spatial_view(const spatial<System> &data) : x(data.x), y(data.y) {}

  __host__ __device__ point operator[](size_t i) const {
    return {x[i], y[i]};
  }
};

__host__ __device__ double distance(point p1, point p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  // TODO: don't do an sqrt here, just factor that as part of the exponent.
  return cuda::std::sqrt(dx * dx + dy * dy);
}

/**
 * Return the bounding box for a span of particles.
 *
 * If the span is empty, an empty rectangle at (0,0) is returned.
 */
template <typename System>
rect bounding_box(spatial_view<System> coordinates,
                  mob::ds::span<System, uint32_t> particles) {
  if (particles.empty()) {
    return rect({0, 0});
  }

  // Annoyingly transform_reduce requires an initial element, but the
  // rectangle-merging operator does not have an identity element.
  //
  // We use the first particle's position as the initial element.
  return thrust::transform_reduce(
      particles.begin() + 1, particles.end(),
      [=] __host__ __device__(uint32_t i) -> rect { return coordinates[i]; },
      rect(coordinates[particles.front()]),
      [] __host__ __device__(rect b1, rect b2) -> rect { return b1 | b2; });
}

__host__ __device__ double spatial_infection_probability(double base, double k,
                                                         double d) {
  double foi = base * cuda::std::pow(d, k);
  return 1 - cuda::std::exp(-foi);
}

template <random_state rng_state_type, typename Proj>
struct probability_filter_impl {
  template <typename T>
  __host__ __device__ bool operator()(const T &value) const {
    return dust::random::random_real<double>(rng.get()) < proj(value);
  }

  cuda::std::reference_wrapper<rng_state_type> rng;
  Proj proj;
};

template <random_state rng_state_type, typename Proj>
__host__ __device__ auto probability_filter(rng_state_type &rng, Proj proj) {
  return compat::filter(probability_filter_impl{cuda::std::ref(rng), proj});
}

template <typename System>
__host__ __device__ double
spatial_infection_filter_impl(const spatial_view<System> &coordinates,
                              point origin, double base, double k, double pnorm,
                              uint32_t j) {
  double d = distance(origin, coordinates[j]);
  return spatial_infection_probability(base, k, d) / pnorm;
}

template <random_state rng_state_type, typename System>
__host__ __device__ auto
spatial_infection_filter(rng_state_type &rng,
                         const spatial_view<System> &coordinates, point origin,
                         double base, double k, double pnorm) {
  return probability_filter(
      rng, cuda::std::bind_front(spatial_infection_filter_impl<System>,
                                 coordinates, origin, base, k, pnorm));
}

template <typename System>
size_t spatial_infection_naive(mob::parallel_random<System> &rngs,
                               infection_list<System> &output,
                               mob::ds::span<System, uint32_t> infected,
                               mob::ds::span<System, uint32_t> susceptible,
                               spatial_view<System> coordinates, double base,
                               double k) {
  return infection_process<System>(
      rngs, output, infected,
      [=] __host__ __device__(uint32_t i, mob::random_proxy<System> &rng) {
        return susceptible | spatial_infection_filter(rng, coordinates,
                                                      coordinates[i], base, k,
                                                      /* pnorm */ 1);
      });
}

template <typename System>
size_t spatial_infection_sieve(mob::parallel_random<System> &rngs,
                               infection_list<System> &output,
                               mob::ds::span<System, uint32_t> infected,
                               mob::ds::span<System, uint32_t> susceptible,
                               spatial_view<System> coordinates, double base,
                               double k) {
  return infection_process<System>(
      rngs, output, infected,
      [=] __host__ __device__(uint32_t i, mob::random_proxy<System> &rng) {
        // TODO: precompute pmax so we don't have to do it twice as part of the
        // infection process. It's a bit tricky because `i` is the individual's
        // ID, not its offset within `infected`.
        auto origin = coordinates[i];
        double pmax = 0;
        for (uint32_t j : susceptible) {
          double d = distance(origin, coordinates[j]);
          double p = spatial_infection_probability(base, k, d);
          if (p > pmax) {
            pmax = p;
          }
        }

        return susceptible | bernoulli(rng, pmax) |
               spatial_infection_filter(rng, coordinates, origin, base, k,
                                        pmax);
      });
}

/**
 * This class describes a geohash, mapping real valued 2D coordinates to a 1
 * dimensional index. The mapping is bijective (ie. no collisions) unlike a
 * typical hash.
 *
 * To make processing easier, the space around the input coordinates is always
 * padded with empty cells, meaning a cell containing a particle always has 8
 * valid neighbours.
 */
struct spatial_hash {
  double width;
  int32_t xmin;
  int32_t xmax;
  int32_t ymin;
  int32_t ymax;

  spatial_hash(const rect &bb, double width)
      : width(width),
        // TODO: +2 is necessary here for when xmax() / width is already a
        // whole number. ceil just returns the number back instead of going up
        xmin(static_cast<int32_t>(cuda::std::floor(bb.xmin() / width)) - 1),
        xmax(static_cast<int32_t>(cuda::std::ceil(bb.xmax() / width)) + 2),
        ymin(static_cast<int32_t>(cuda::std::floor(bb.ymin() / width)) - 1),
        ymax(static_cast<int32_t>(cuda::std::ceil(bb.ymax() / width)) + 2) {}

  __host__ __device__ uint32_t operator()(point p) const {
    auto [cx, cy] = map(p);
    return cell_index(cx, cy);
  }

  __host__ __device__ uint32_t cell_index(uint32_t cx, uint32_t cy) const {
    return cx + cy * (xmax - xmin);
  }

  /**
   * This returns the Moore neighborhood, ie. the indices of the 9 tiles
   * surrounding the given coordinates.
   */
  __host__ __device__ cuda::std::array<uint32_t, 9>
  neighbourhood(point p) const {
    auto [cx, cy] = map(p);

    // Assuming p is contained in the bounding box, all these indices are always
    // non-negative and in-bounds.
    return {
        // clang-format off
        cell_index(cx - 1, cy - 1), cell_index(cx, cy - 1), cell_index(cx + 1, cy - 1),
        cell_index(cx - 1, cy),     cell_index(cx, cy),     cell_index(cx + 1, cy),
        cell_index(cx - 1, cy + 1), cell_index(cx, cy + 1), cell_index(cx + 1, cy + 1),
        // clang-format on
    };
  }

  /**
   * Return the Chebyshev distance on the grid between two points.
   */
  __host__ __device__ uint32_t distance(point p1, point p2) const {
    auto [cx1, cy1] = map(p1);
    auto [cx2, cy2] = map(p2);

    return cuda::std::max(abs_diff(cx1, cx2), abs_diff(cy1, cy2));
  }

  __host__ __device__ bool are_neighbours(point p1, point p2) const {
    return distance(p1, p2) <= 1;
  }

  // Return the number of bins needed by this hash.
  __host__ __device__ uint32_t size() const {
    return (xmax - xmin) * (ymax - ymin);
  }

  /**
   * Map a point onto the grid.
   */
  __host__ __device__ cuda::std::pair<uint32_t, uint32_t> map(point p) const {
    uint32_t cx = static_cast<int32_t>(cuda::std::floor(p.x / width)) - xmin;
    uint32_t cy = static_cast<int32_t>(cuda::std::floor(p.y / width)) - ymin;
    return {cx, cy};
  }

  /**
   * TODO: finish this.
   * This is the "strips" optimization described in "Improved GPU Near
   * Neighbours Performance for Multi-Agent Simulations" by Chisholm et al.
   *
   * Get all the elements from the given index that are in the same row as (x,
   * y), with an optional yshift (-1, 0 or 1).
   *
   * Consecutive x coordinates map to consecutive cell indices. Ragged vectors
   * store consecutive indices adjacent to one another, allowing us to return an
   * entire row as one contiguous span.
   */
  /*
  template <typename T>
  __host__ __device__ ds::span<System, T>
  row(mob::ds::ragged_vector_view<System, T> index, double x, double y,
      int32_t yshift) const {
    int32_t cx = static_cast<int32_t>(cuda::std::floor(x / width)) - xmin;
    int32_t cy =
        static_cast<int32_t>(cuda::std::floor(y / width)) - ymin + yshift;

    if (cy < 0 || cy >= (ymax - ymin)) {
      // Out of bounds on the Y axis, return an empty slice
      return {};
    }
    if (cx == 0 && cx + 1 >= (xmax - xmin)) {
      return index[cell_index(cx, cy)];
    } else if (cx == 0) {
      return index.slice(cell_index(cx, cy), cell_index(cx + 1, cy));
    } else if (cx + 1 >= (xmax - xmin)) {
      return index.slice(cell_index(cx - 1, cy), cell_index(cx, cy));
    } else {
      return index.slice(cell_index(cx - 1, cy), cell_index(cx + 1, cy));
    }
  }
  */
};

template <typename System>
struct spatial_partition {
  spatial_hash hash;
  mob::ds::ragged_vector<System, uint32_t> data;

  spatial_partition(spatial_hash hash, spatial_view<System> coordinates,
                    mob::vector<System, uint32_t> individuals)
      : hash(hash), data(hash.size()) {
    assign(coordinates, individuals);
  }

  // Can't have __hd__ lambdas in a constructor
  void assign(spatial_view<System> coordinates,
              mob::vector<System, uint32_t> individuals) {
    auto hash = this->hash;
    mob::vector<System, uint32_t> cells(individuals.size());
    thrust::transform(
        individuals.begin(), individuals.end(), cells.begin(),
        [=] __host__ __device__(uint32_t i) { return hash(coordinates[i]); });

    data.assign(std::move(cells), std::move(individuals));
  }
};

template <typename System>
struct spatial_partition_view {
  spatial_hash hash;
  mob::ds::ragged_vector_view<System, uint32_t> data;

  spatial_partition_view(const spatial_partition<System> &partition)
      : hash(partition.hash), data(partition.data) {}

  __host__ __device__ auto neighbours(point origin) const {
    return hash.neighbourhood(origin) |
           mob::compat::transform(cuda::std::bind_front(
               &mob::ds::ragged_vector_view<System, uint32_t>::operator[],
               data)) |
           mob::compat::join;
  }
};

template <typename System>
size_t local_infection_process(mob::parallel_random<System> &rngs,
                               infection_list<System> &output,
                               mob::ds::span<System, uint32_t> infected,
                               mob::ds::span<System, uint32_t> susceptible,
                               spatial_view<System> coordinates,
                               spatial_partition_view<System> partition,
                               double base, double k) {
  return infection_process<System>(
      rngs, output, infected,
      [=] __host__ __device__(uint32_t i, mob::random_proxy<System> &rng) {
        auto origin = coordinates[i];

        return partition.neighbours(origin) |
               spatial_infection_filter(rng, coordinates, origin, base, k, 1);
      });
}

template <typename System>
size_t distant_infection_process(mob::parallel_random<System> &rngs,
                                 infection_list<System> &output,
                                 mob::ds::span<System, uint32_t> infected,
                                 mob::ds::span<System, uint32_t> susceptible,
                                 spatial_view<System> coordinates,
                                 const spatial_hash &hash, double base,
                                 double k) {
  double pthreshold = spatial_infection_probability(base, k, hash.width);

  auto is_distant = [=] __host__ __device__(point origin, uint32_t j)
      -> bool { return !hash.are_neighbours(origin, coordinates[j]); };

  return infection_process<System>(
      rngs, output, infected,
      [=] __host__ __device__(uint32_t i, mob::random_proxy<System> &rng) {
        auto origin = coordinates[i];
        return susceptible | bernoulli(rng, pthreshold) |
               compat::filter(cuda::std::bind_front(is_distant, origin)) |
               spatial_infection_filter(rng, coordinates, origin, base, k,
                                        pthreshold);
      });
}

template <typename System>
std::pair<size_t, size_t> spatial_infection_hybrid(
    mob::parallel_random<System> &rngs, infection_list<System> &output,
    mob::ds::span<System, uint32_t> infected,
    mob::ds::span<System, uint32_t> susceptible,
    spatial_view<System> coordinates, double base, double k, double width) {

  // We could use just the susceptible bounding box, but that would require
  // extra bounds checks during queries. Having the bounding box include
  // everything makes this easier.
  auto bb = bounding_box(coordinates, infected) |
            bounding_box(coordinates, susceptible);

  spatial_hash hash(bb, width);
  spatial_partition partition(hash, coordinates,
                              {susceptible.begin(), susceptible.end()});

  size_t n_local = local_infection_process<System>(
      rngs, output, infected, susceptible, coordinates, partition, base, k);
  size_t n_distant = distant_infection_process(
      rngs, output, infected, susceptible, coordinates, hash, base, k);

  return {n_local, n_distant};
}

} // namespace mob

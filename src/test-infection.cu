#include <mob/infection.h>
#include <mob/parallel_random.h>
#include <mob/roaring/bitset.h>

#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <optional>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

rc::Gen<mob::host_random> genRandomState(size_t size) {
  return rc::gen::map(rc::gen::noShrink(rc::gen::arbitrary<int>()),
                      [=](int seed) { return mob::host_random(size, seed); });
}

rc::Gen<std::pair<mob::roaring::bitset, mob::roaring::bitset>>
genCompartments(uint32_t population) {
  // This isn't a great test as it makes each compartment contiguous. It should
  // be rewritten to use sample I from the population, and then produce I by
  // taking the complement.
  auto makeCompartments = [=](uint32_t infected_size) {
    mob::roaring::bitset infected{
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(infected_size)};

    mob::roaring::bitset susceptible{
        thrust::counting_iterator<uint32_t>(infected_size),
        thrust::counting_iterator<uint32_t>(population)};

    return std::make_pair(std::move(susceptible), std::move(infected));
  };

  return rc::gen::map(rc::gen::inRange<uint32_t>(0, population + 1),
                      makeCompartments);
}

rc::Gen<double> genProbability() {
  return rc::gen::map(rc::gen::inRange<uint32_t>(0, 4097),
                      [](uint32_t value) { return (double)value / 4097.f; });
}

// TEST_CASE("infection") {
//   rc::prop("homogeneous_infection", []() {
//     uint32_t population = *rc::gen::inRange<uint32_t>(0, 1000);
//     double propability = *genProbability();
//
//     auto rngs = *genRandomState(population);
//     auto [susceptible, infected] = *genCompartments(population);
//
//     auto result =
//         homogeneous_infection_process(rngs, susceptible, infected,
//         propability);
//
//     RC_ASSERT(result.first.size() == result.second.size());
//     for (auto i : result.first) {
//       RC_ASSERT(susceptible.contains(i));
//     }
//     for (auto i : result.second) {
//       RC_ASSERT(infected.contains(i));
//     }
//   });
// }

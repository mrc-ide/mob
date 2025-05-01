#include <mob/ds/intersection.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

TEST_CASE("intersection") {
  rc::prop("lazy_intersection",
           [](std::vector<uint32_t> x, std::vector<uint32_t> y) {
             std::vector<uint32_t> result;
             for (auto i : mob::ds::lazy_intersection(x, y)) {
               result.push_back(i);
             }

             std::vector<uint32_t> expected;
             std::set_intersection(x.begin(), x.end(), y.begin(), y.end(),
                                   std::back_inserter(expected));

             RC_ASSERT(expected == result);
           });
}

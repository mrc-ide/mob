#pragma once

#include <mob/system.h>

namespace mob {

template <typename System>
using integer_vector = mob::vector<System, uint32_t>;

template <typename System>
using double_vector = mob::vector<System, double>;

} // namespace mob

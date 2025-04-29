#pragma once

#include <mob/parallel_random.h>

struct random_wrapper {
  template <typename... Args>
  random_wrapper(Args &&...args) : inner(std::forward<Args>(args)...) {}

  mob::device_random &operator*() {
    return inner;
  }

  mob::device_random *operator->() {
    return &inner;
  }

  mob::device_random inner;
};

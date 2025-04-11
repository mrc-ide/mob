#pragma once

namespace mob::ds {

// https://quuxplusone.github.io/blog/2019/02/06/arrow-proxy/
template <typename T>
struct arrow_proxy {
  T value;
  T *operator->() {
    return &value;
  }
};

} // namespace mob::ds

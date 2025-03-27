#pragma once

__host__ __device__ struct counting_output_iterator {
  struct proxy;

  counting_output_iterator(size_t offset = 0) : offset_(offset) {}

  proxy operator*() {
    return proxy();
  }

  counting_output_iterator &operator++() {
    offset_++;
    return *this;
  }

  counting_output_iterator operator++(int) {
    counting_output_iterator temp = *this;
    ++(*this);
    return temp;
  }

  size_t offset() const {
    return offset_;
  }

  struct proxy {
    template <typename T>
    void operator=(T &&) {}
  };

private:
  size_t offset_;
};

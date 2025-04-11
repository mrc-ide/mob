#pragma once

#include <mob/compat.h>
#include <mob/ds/arrow_proxy.h>

#include <algorithm>
#include <thrust/advance.h>
#include <variant>

namespace mob {
namespace ds {

template <typename TupleLike, template <typename> typename F>
struct tuple_map;

template <typename... T, template <typename> typename F>
struct tuple_map<cuda::std::tuple<T...>, F> {
  using type = cuda::std::tuple<F<T>...>;
};

template <typename T, typename U, template <typename> typename F>
struct tuple_map<cuda::std::pair<T, U>, F> {
  using type = cuda::std::pair<F<T>, F<U>>;
};

template <typename TupleLike, template <typename> typename F>
using tuple_map_t = typename tuple_map<TupleLike, F>::type;

template <typename T>
using iter_reference_t = typename std::iterator_traits<T>::reference;

template <typename T>
using iter_value_t = typename std::iterator_traits<T>::value_type;

template <typename C>
using container_value_type = typename C::value_type;

template <typename C>
using container_iterator = typename C::iterator;

template <typename C>
using container_const_iterator = typename C::const_iterator;

template <typename Tuple>
struct soa_iterator {
  soa_iterator(Tuple iterators) : iterators_(iterators) {}

  using tuple_type = Tuple;
  using reference = tuple_map_t<Tuple, iter_reference_t>;
  using value_type = tuple_map_t<Tuple, iter_value_t>;
  using pointer = arrow_proxy<reference>;

  bool operator==(const soa_iterator &other) const {
    return cuda::std::get<0>(iterators_) == cuda::std::get<0>(other.iterators_);
  }

  bool operator!=(const soa_iterator &other) const {
    return cuda::std::get<0>(iterators_) != cuda::std::get<0>(other.iterators_);
  }

  soa_iterator &operator++() {
    increment(index_seq{});
    return *this;
  }

  soa_iterator &operator+=(ptrdiff_t n) {
    advance(n, index_seq{});
    return *this;
  }

  soa_iterator operator+(ptrdiff_t n) const {
    soa_iterator copy = *this;
    copy += n;
    return copy;
  }

  reference operator*() const {
    return dereference(index_seq{});
  }

  pointer operator->() const {
    return pointer{**this};
  }

  const Tuple &get_iterator_tuple() const {
    return iterators_;
  }

private:
  using index_seq = std::make_index_sequence<cuda::std::tuple_size_v<Tuple>>;

  template <size_t... I>
  void advance(ptrdiff_t n, std::index_sequence<I...>) {
    // This should be cuda::std::advance, but for some reason that doesn't work?
    (thrust::advance(cuda::std::get<I>(iterators_), n), ...);
  }

  template <size_t... I>
  void increment(std::index_sequence<I...>) {
    (++cuda::std::get<I>(iterators_), ...);
  }

  template <size_t... I>
  reference dereference(std::index_sequence<I...>) const {
    return reference{*cuda::std::get<I>(iterators_)...};
  }

  Tuple iterators_;
};

template <typename Tuple>
struct soa_array {
  using value_type = tuple_map_t<Tuple, container_value_type>;
  using iterator = soa_iterator<tuple_map_t<Tuple, container_iterator>>;
  using reference = typename iterator::reference;
  using const_iterator =
      soa_iterator<tuple_map_t<Tuple, container_const_iterator>>;

  const_iterator begin() const {
    return apply([](const auto &...column) {
      return const_iterator({column.begin()...});
    });
  }

  const_iterator end() const {
    return apply([](const auto &...column) {
      return const_iterator({column.end()...});
    });
  }

  iterator begin() {
    return apply([](auto &...column) { return iterator({column.begin()...}); });
  }

  iterator end() {
    return apply([](auto &...column) { return iterator({column.end()...}); });
  }

  iterator insert(iterator position, value_type values) {
    return insert_impl(position, std::move(values), index_seq{});
  }

  void push_back(value_type values) {
    push_back_impl(values, index_seq{});
  }

  reference back() {
    return apply([](auto &...column) { return reference(column.back()...); });
  }

  template <
      typename... Packs,
      typename = std::enable_if_t<sizeof...(Packs) == std::tuple_size_v<Tuple>>>
  void emplace_back(Packs &&...packs) {
    emplace_back_impl(std::forward_as_tuple(std::forward<Packs>(packs)...),
                      index_seq{});
  }

  template <
      typename... Packs,
      typename = std::enable_if_t<sizeof...(Packs) == std::tuple_size_v<Tuple>>>
  iterator emplace(iterator position, Packs &&...packs) {
    return emplace_impl(position,
                        std::forward_as_tuple(std::forward<Packs>(packs)...),
                        index_seq{});
  }

  template <size_t I>
  cuda::std::tuple_element_t<I, Tuple> &get() {
    return cuda::std::get<I>(columns_);
  }

  template <size_t I>
  const cuda::std::tuple_element_t<I, Tuple> &get() const {
    return cuda::std::get<I>(columns_);
  }

  size_t size() const {
    return cuda::std::get<0>(columns_).size();
  }

private:
  using index_seq = std::make_index_sequence<cuda::std::tuple_size_v<Tuple>>;

  template <size_t... I>
  iterator insert_impl(iterator position, value_type values,
                       std::index_sequence<I...>) {
    const auto &iterators = position.get_iterator_tuple();
    return iterator({cuda::std::get<I>(columns_).insert(
        cuda::std::get<I>(iterators),
        cuda::std::get<I>(std::move(values)))...});
  }

  template <size_t... I>
  void push_back_impl(value_type values, std::index_sequence<I...>) {
    (cuda::std::get<I>(columns_).push_back(cuda::std::get<I>(values)), ...);
  }

  template <typename Column, typename... Args, size_t... I>
  static void emplace_back_one(Column &column, std::tuple<Args...> &&args,
                               std::index_sequence<I...>) {
    column.emplace_back(std::get<I>(std::move(args))...);
  }

  template <typename Column, typename... Args, size_t... I>
  static typename Column::iterator
  emplace_one(Column &column, typename Column::iterator position,
              std::tuple<Args...> &&args, std::index_sequence<I...>) {
    return column.emplace(position, std::get<I>(std::move(args))...);
  }

  template <typename Packs, size_t... I>
  void emplace_back_impl(Packs &&packs, std::index_sequence<I...>) {
    // https://www.lapthorn.net/posts/2024/piecewise-tuple/
    (emplace_back_one(
         cuda::std::get<I>(columns_), std::get<I>(std::forward<Packs>(packs)),
         std::make_index_sequence<std::tuple_size_v<
             std::remove_reference_t<std::tuple_element_t<I, Packs>>>>()),
     ...);
  }

  template <typename Packs, size_t... I>
  iterator emplace_impl(iterator position, Packs &&packs,
                        std::index_sequence<I...>) {
    // https://www.lapthorn.net/posts/2024/piecewise-tuple/
    const auto &iterators = position.get_iterator_tuple();
    return iterator({emplace_one(
        cuda::std::get<I>(columns_), cuda::std::get<I>(iterators),
        std::get<I>(std::forward<Packs>(packs)),
        std::make_index_sequence<std::tuple_size_v<
            std::remove_reference_t<std::tuple_element_t<I, Packs>>>>())...});
  }

  template <typename Fn>
  auto apply(Fn &&fn) {
    return apply_impl(std::forward<Fn>(fn), index_seq{});
  }

  template <typename Fn>
  auto apply(Fn &&fn) const {
    return apply_impl(std::forward<Fn>(fn), index_seq{});
  }

  template <typename Fn, size_t... I>
  auto apply_impl(Fn &&fn, std::index_sequence<I...>) {
    return std::forward<Fn>(fn)(cuda::std::get<I>(columns_)...);
  }

  template <typename Fn, size_t... I>
  auto apply_impl(Fn &&fn, std::index_sequence<I...>) const {
    return std::forward<Fn>(fn)(cuda::std::get<I>(columns_)...);
  }

  Tuple columns_;
};

template <typename Key, typename Value>
struct soa_map {
  using array_type = soa_array<cuda::std::pair<Key, Value>>;
  using iterator = typename array_type::iterator;
  using const_iterator = typename array_type::const_iterator;
  using value_type = typename iterator::value_type;
  using key_type = typename Key::value_type;

  bool contains(const typename Key::value_type &key) {
    const auto &key_container = contents_.template get<0>();
    return compat::binary_search(key_container.begin(), key_container.end(),
                                 key);
  }

  const_iterator find(const typename Key::value_type &key) const {
    auto it = lower_bound(key);
    if (it != end() && (*it).first == key) {
      return it;
    } else {
      return end();
    }
  }

  const_iterator lower_bound(const typename Key::value_type &key) const {
    auto &key_container = contents_.template get<0>();
    auto &value_container = contents_.template get<1>();

    auto key_it =
        compat::lower_bound(key_container.begin(), key_container.end(), key);
    if (key_it == key_container.end()) {
      return const_iterator({key_it, value_container.end()});
    } else {
      auto distance = key_it - key_container.begin();
      return const_iterator({key_it, value_container.begin() + distance});
    }
  }

  iterator find(const typename Key::value_type &key) {
    auto it = lower_bound(key);
    if (it != end() && (*it).first == key) {
      return it;
    } else {
      return end();
    }
  }

  iterator lower_bound(const typename Key::value_type &key) {
    auto &key_container = contents_.template get<0>();
    auto &value_container = contents_.template get<1>();

    auto key_it =
        compat::lower_bound(key_container.begin(), key_container.end(), key);
    if (key_it == key_container.end()) {
      return iterator({key_it, value_container.end()});
    } else {
      auto distance = key_it - key_container.begin();
      return iterator({key_it, value_container.begin() + distance});
    }
  }

  std::pair<iterator, bool> insert(value_type value) {
    const auto &key = cuda::std::get<0>(value);
    auto position = lower_bound(key);
    if (position != end() && (*position).first == key) {
      return {position, false};
    } else {
      return {contents_.insert(position, std::move(value)), true};
    }
  }

  template <typename... Args>
  std::pair<iterator, bool> try_emplace(const key_type &key, Args &&...args) {
    auto position = lower_bound(key);
    if (position != end() && (*position).first == key) {
      return {position, false};
    } else {
      auto inserted =
          contents_.emplace(position, std::make_tuple(key),
                            std::forward_as_tuple(std::forward<Args>(args)...));
      return {inserted, true};
    }
  }

  size_t size() const {
    return contents_.size();
  }

  iterator begin() {
    return contents_.begin();
  }

  iterator end() {
    return contents_.end();
  }

  const_iterator begin() const {
    return contents_.begin();
  }

  const_iterator end() const {
    return contents_.end();
  }

  typename iterator::reference front() {
    return *begin();
  }

  typename iterator::reference back() {
    return *end();
  }

  typename const_iterator::reference front() const {
    return *begin();
  }

  typename const_iterator::reference back() const {
    return *end();
  }

private:
  array_type contents_;
};

template <typename, typename S, typename... U>
struct find_variant_index;

template <typename S, typename... U>
struct find_variant_index<void, S, S, U...> {
  static inline constexpr size_t value = 0;
};

template <typename S, typename U1, typename... U>
struct find_variant_index<std::enable_if_t<!std::is_same_v<S, U1>>, S, U1,
                          U...> {
  using base = find_variant_index<void, S, U...>;
  static inline constexpr size_t value = base::value + 1;
};

template <typename S, typename... U>
static inline constexpr size_t find_variant_index_v =
    find_variant_index<void, S, U...>::value;

template <typename T>
void destroy(T *ptr) {
  ptr->~T();
}

template <template <typename> typename Vector, typename... Ts>
struct soa_variant {
  static_assert(sizeof...(Ts) < 256);
  using tag_type = uint8_t;

private:
  struct storage {
    alignas(Ts...) std::byte buffer[std::max({sizeof(Ts)...})];
  };

  using array_type =
      soa_array<cuda::std::pair<Vector<tag_type>, Vector<storage>>>;

public:
  template <typename Iterator>
  struct proxy {
    static inline constexpr bool is_const = std::is_const_v<
        std::remove_reference_t<typename Iterator::reference::second_type>>;

    template <typename U>
    using adapt = std::conditional_t<is_const, std::add_const_t<U>, U>;

    explicit proxy(Iterator it) : iterator_(it) {}

    size_t index() {
      return (*iterator_).first;
    }

    template <typename R = void, typename Fn>
    R visit_ptr(Fn &&fn) {
      return visit_impl<0, R>(std::forward<Fn>(fn));
    }

    template <typename R = void, typename Fn>
    R visit(Fn &&fn) {
      return visit_ptr<R>(
          [&fn](auto *ptr) -> R { return std::forward<Fn>(fn)(*ptr); });
    }

    template <typename U, typename... Args>
    void emplace(Args &&...args) {
      destroy();

      (*iterator_).first = find_variant_index_v<U, Ts...>;
      new ((*iterator_).second.buffer) U(std::forward<Args>(args)...);
    }

    void destroy() {
      visit_ptr([](auto *ptr) { mob::ds::destroy(ptr); });
      // TODO: poison the index
    }

    template <typename U>
    bool holds_alternative() {
      static constexpr size_t index = find_variant_index_v<U, Ts...>;
      return this->index() == index;
    }

    template <typename U>
    adapt<U> *get_if() {
      if (this->holds_alternative<U>()) {
        return this->get_unchecked<U>();
      } else {
        return nullptr;
      }
    }

  private:
    template <size_t I, typename R, typename Fn>
    R visit_impl(Fn &&fn) {
      if constexpr (I < sizeof...(Ts)) {
        if (index() == I) {
          return std::forward<Fn>(fn)(this->get_unchecked<I>());
        } else {
          return visit_impl<I + 1, R>(std::forward<Fn>(fn));
        }
      } else {
        throw std::logic_error("unreachable");
      }
    }

    template <tag_type I>
    adapt<std::tuple_element_t<I, std::tuple<Ts...>>> *get_unchecked() {
      using element_type = std::tuple_element_t<I, std::tuple<Ts...>>;
      auto &storage = (*iterator_).second;
      return reinterpret_cast<adapt<element_type> *>(storage.buffer);
    }

    template <typename U>
    adapt<U> *get_unchecked() {
      static constexpr size_t index = find_variant_index_v<U, Ts...>;
      return this->get_unchecked<index>();
    }

    Iterator iterator_;
  };

  template <typename Iterator>
  struct base_iterator {
    friend soa_variant;
    using iterator_category = std::forward_iterator_tag;
    using reference = proxy<Iterator>;
    using pointer = arrow_proxy<reference>;
    using value_type = std::variant<Ts...>;
    using difference_type = ptrdiff_t;

    base_iterator(Iterator it) : underlying_(it) {}

    base_iterator &operator++() {
      ++underlying_;
      return *this;
    }

    base_iterator &operator+=(difference_type n) {
      underlying_ += n;
      return *this;
    }

    base_iterator operator+(difference_type n) {
      return base_iterator(underlying_ + n);
    }

    bool operator==(const base_iterator &other) const {
      return underlying_ == other.underlying_;
    }

    bool operator!=(const base_iterator &other) const {
      return underlying_ != other.underlying_;
    }

    reference operator*() const {
      return proxy(underlying_);
    }

    pointer operator->() const {
      return pointer{**this};
    }

  private:
    Iterator underlying_;
  };

  using value_type = std::variant<Ts...>;
  // using reference = proxy;

  using iterator = base_iterator<typename array_type::iterator>;
  using const_iterator = base_iterator<typename array_type::const_iterator>;

  iterator begin() {
    return iterator{contents_.begin()};
  }

  iterator end() {
    return iterator{contents_.end()};
  }

  const_iterator begin() const {
    return const_iterator{contents_.begin()};
  }

  const_iterator end() const {
    return const_iterator{contents_.end()};
  }

  template <typename U, typename... Args>
  void emplace_back(std::in_place_type_t<U>, Args &&...args) {
    size_t index = find_variant_index_v<U, Ts...>;
    contents_.emplace_back(std::make_tuple(index), std::make_tuple());

    void *ptr = contents_.back().second.buffer;
    new (ptr) U(std::forward<Args>(args)...);
  }

  template <typename U, typename... Args>
  iterator emplace(iterator position, std::in_place_type_t<U>, Args &&...args) {
    size_t index = find_variant_index_v<U, Ts...>;

    auto it = contents_.emplace(position.underlying_, std::make_tuple(index),
                                std::make_tuple());

    void *ptr = it->second.buffer;
    new (ptr) U(std::forward<Args>(args)...);
    return iterator{it};
  }

  iterator insert(iterator position, value_type value) {
    iterator it(contents_.emplace(position.underlying_,
                                  std::make_tuple(value.index()),
                                  std::make_tuple()));

    it->visit_ptr([&](auto *ptr) {
      using type = typename std::pointer_traits<decltype(ptr)>::element_type;
      new (ptr) type(std::get<type>(value));
    });

    return it;
  }

  soa_variant() = default;

  soa_variant(const soa_variant &) = delete;
  soa_variant &operator=(const soa_variant &) = delete;

  soa_variant(soa_variant &&) = default;
  soa_variant &operator=(soa_variant &&) = default;

  ~soa_variant() {
    for (auto it = begin(); it != end(); ++it) {
      (*it).destroy();
    }
  }

private:
  array_type contents_;
};

} // namespace ds
} // namespace mob

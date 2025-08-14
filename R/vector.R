#' @export
integer_vector_create <- create_wrapper("integer_vector_create")

#' @export
double_vector_create <- create_wrapper("double_vector_create")

#' @export
vector_rep <- new_generic("vector_rep", c("vector"))
method(vector_rep, class_integer_vector) <- create_wrapper("integer_vector_rep")
method(vector_rep, class_double_vector) <- create_wrapper("double_vector_rep")

#' @export
vector_clone <- new_generic("vector_rep", c("vector"))
method(vector_clone, class_integer_vector) <- create_wrapper("integer_vector_clone")
method(vector_clone, class_double_vector) <- create_wrapper("double_vector_clone")

#' Get the contents of the vector as an R atomic vector.
#' @export
vector_values <- new_generic("vector_values", c("vector"))
method(vector_values, class_integer_vector) <- create_wrapper("integer_vector_values")
method(vector_values, class_double_vector) <- create_wrapper("double_vector_values")

#' @export
vector_scatter <- new_generic("vector_scatter", c("vector", "indices", "values"))
method(vector_scatter, list(class_any, class_numeric, class_any)) <- function(vector, indices, values) {
  vector_scatter(vector, integer_vector_create(indices), values)
}

method(vector_scatter, list(class_integer_vector, class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_scatter")
method(vector_scatter, list(class_integer_vector, class_bitset, class_integer_vector)) <- create_wrapper("integer_vector_scatter_bitset")
method(vector_scatter, list(class_integer_vector, class_any, class_numeric)) <- function(vector, indices, values) {
  vector_scatter(vector, indices, integer_vector_create(values))
}

method(vector_scatter, list(class_double_vector, class_integer_vector, class_double_vector)) <- create_wrapper("double_vector_scatter")
method(vector_scatter, list(class_double_vector, class_bitset, class_double_vector)) <- create_wrapper("double_vector_scatter_bitset")
method(vector_scatter, list(class_double_vector, class_any, class_numeric)) <- function(vector, indices, values) {
  vector_scatter(vector, indices, double_vector_create(values))
}

#' @export
vector_scatter_scalar <- new_generic("vector_scatter_scalar", c("vector", "indices"))
method(vector_scatter_scalar, list(class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_scatter_scalar")
method(vector_scatter_scalar, list(class_double_vector, class_integer_vector)) <- create_wrapper("double_vector_scatter_scalar")
method(vector_scatter_scalar, list(class_any, class_numeric)) <- function(vector, indices, value) {
  vector_scatter_scalar(vector, integer_vector_create(indices), value)
}

#' @export
vector_gather <- new_generic("vector_gather", c("vector", "indices"))
method(vector_gather, list(class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_gather")
method(vector_gather, list(class_double_vector, class_integer_vector)) <- create_wrapper("double_vector_gather")
method(vector_gather, list(class_any, class_numeric)) <- function(vector, indices) {
  vector_gather(vector, integer_vector_create(indices))
}

#' Find indices of elements equal to the given value.
#' @export
integer_vector_match_eq <- create_wrapper("integer_vector_match_eq")

#' Find indices of elements greaher than the given value.
#' @export
integer_vector_match_gt <- create_wrapper("integer_vector_match_gt")

#' @export
integer_vector_match_eq_as_bitset <- create_wrapper("integer_vector_match_eq_as_bitset")

#' @export
integer_vector_match_gt_as_bitset <- create_wrapper("integer_vector_match_gt_as_bitset")

#' @export
vector_add_scalar <- new_generic("vector_add_scalar", c("vector"))
method(vector_add_scalar, class_integer_vector) <- create_wrapper("integer_vector_add_scalar")
method(vector_add_scalar, class_double_vector) <- create_wrapper("double_vector_add_scalar")

#' @export
vector_mul_scalar <- new_generic("vector_mul_scalar", c("vector"))
method(vector_mul_scalar, class_integer_vector) <- create_wrapper("integer_vector_mul_scalar")
method(vector_mul_scalar, class_double_vector) <- create_wrapper("double_vector_mul_scalar")

#' @export
vector_div_scalar <- new_generic("vector_div_scalar", c("vector"))
method(vector_div_scalar, class_double_vector) <- create_wrapper("double_vector_div_scalar")

#' @export
vector_add <- new_generic("vector_add", c("left", "right"))
method(vector_add, list(class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_add")
method(vector_add, list(class_double_vector, class_double_vector)) <- create_wrapper("double_vector_add")
method(vector_add, list(class_integer_vector, class_numeric)) <- function(left, right) {
  vector_add(left, integer_vector_create(right))
}
method(vector_add, list(class_double_vector, class_numeric)) <- function(left, right) {
  vector_add(left, double_vector_create(right))
}

#' @export
vector_mul <- new_generic("vector_mul", c("left", "right"))
method(vector_mul, list(class_integer_vector, class_integer_vector)) <- create_wrapper("integer_vector_mul")
method(vector_mul, list(class_double_vector, class_double_vector)) <- create_wrapper("double_vector_mul")
method(vector_mul, list(class_integer_vector, class_numeric)) <- function(left, right) {
  vector_mul(left, integer_vector_create(right))
}
method(vector_mul, list(class_double_vector, class_numeric)) <- function(left, right) {
  vector_mul(left, double_vector_create(right))
}

#' @export
vector_div <- new_generic("vector_div", c("left", "right"))
method(vector_div, list(class_double_vector, class_double_vector)) <- create_wrapper("double_vector_div")
method(vector_div, list(class_double_vector, class_numeric)) <- function(left, right) {
  vector_div(left, double_vector_create(right))
}

#' @export
vector_neg <- new_generic("vector_neg", c("vector"))
method(vector_neg, class_double_vector) <- create_wrapper("double_vector_neg")

#' @export
vector_exp <- new_generic("vector_exp", c("vector"))
method(vector_exp, class_double_vector) <- create_wrapper("double_vector_exp")

#' @export
vector_reciprocal <- new_generic("vector_reciprocal", c("vector"))
method(vector_reciprocal, class_double_vector) <- create_wrapper("double_vector_reciprocal")

#' Round values of the vector to the nearest integer.
#' @export
vector_lround <- new_generic("vector_lround", c("vector"))
method(vector_lround, class_double_vector) <- create_wrapper("double_vector_lround")

#' @export
vector_to_double <- new_generic("vector_to_double", c("vector"))
method(vector_to_double, class_integer_vector) <- create_wrapper("integer_vector_to_double")

#pragma once

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <type_traits>

namespace sparsexx::detail {

template <typename SpMatType, typename = void>
struct is_csr_matrix : public std::false_type {};

template <typename SpMatType>
struct is_csr_matrix< SpMatType,
  std::enable_if_t< 
    std::is_base_of_v< 
      csr_matrix< typename SpMatType::value_type,
                  typename SpMatType::index_type,
                  typename SpMatType::allocator_type >, SpMatType >
  >
> : public std::true_type {};


template <typename SpMatType>
inline constexpr bool is_csr_matrix_v = is_csr_matrix<SpMatType>::value;

template <typename SpMatType, typename U = void>
struct enable_if_csr_matrix {
  using type = std::enable_if_t< is_csr_matrix_v<SpMatType>, U>;
};

template <typename SpMatType, typename U = void>
using enable_if_csr_matrix_t = typename enable_if_csr_matrix<SpMatType,U>::type;


}

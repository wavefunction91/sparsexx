#pragma once

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>
#include <iostream>

namespace sparsexx::spblas {

namespace detail {

  template <typename SpMatType, typename ALPHAT, typename BETAT>
  struct are_alpha_beta_convertible {
    inline static constexpr bool value =
      std::is_convertible_v< ALPHAT, typename SpMatType::value_type > and
      std::is_convertible_v< BETAT,  typename SpMatType::value_type >;
  };

  template <typename SpMatType, typename ALPHAT, typename BETAT>
  inline constexpr bool are_alpha_beta_convertible_v = 
    are_alpha_beta_convertible<SpMatType, ALPHAT, BETAT>::value;
  

  template <typename SpMatType, typename ALPHAT, typename BETAT>
  struct spmbv_uses_mkl {
    inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::mkl::is_mkl_sparse_matrix_v<SpMatType>;
  };

  template <typename SpMatType, typename ALPHAT, typename BETAT>
  inline constexpr bool spmbv_uses_mkl_v =
    spmbv_uses_mkl<SpMatType, ALPHAT, BETAT>::value;


  template <typename SpMatType, typename ALPHAT, typename BETAT>
  struct spmbv_uses_generic_csr {
    inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::is_csr_matrix_v<SpMatType> and
      not spmbv_uses_mkl_v<SpMatType, ALPHAT, BETAT>;
  };

  template <typename SpMatType, typename ALPHAT, typename BETAT>
  inline constexpr bool spmbv_uses_generic_csr_v =
    spmbv_uses_generic_csr<SpMatType, ALPHAT, BETAT>::value;
}

template <typename SpMatType, typename ALPHAT, typename BETAT>
std::enable_if_t< detail::spmbv_uses_generic_csr_v<SpMatType, ALPHAT, BETAT> >
  gespmbv( int64_t K, ALPHAT ALPHA, const SpMatType& A,
    const typename SpMatType::value_type* V,  int64_t LDV,  BETAT BETA,
          typename SpMatType::value_type* AV, int64_t LDAV ) {

  using value_type = typename SpMatType::value_type;
  static_assert( std::is_same_v<value_type,double>, 
                 "MKL SPMBV is only implemented for double precision" );

  const value_type alpha = ALPHA;
  const value_type beta  = BETA;

  const auto  N = A.n();
  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  //std::cout << "IN GENERIC" << std::endl;
  #pragma omp parallel for collapse(2)
  for( int64_t k = 0; k < K; ++k )
  for( int64_t i = 0; i < N; ++i ) {
    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
    const auto j_ext = j_en - j_st;

    const auto* V_k    = V + k*LDV - indexing; 
    const auto* Anz_st = Anz + j_st;
    const auto* Aci_st = Aci + j_st;

    value_type av = 0.;
    for( int64_t j = 0; j < j_ext; ++j )
      av += Anz_st[j] * V_k[ Aci_st[j] ];


    AV[ i + k*LDV ] = alpha * av + beta * AV[ i + k*LDV ];
  }


}

template <typename SpMatType, typename ALPHAT, typename BETAT>
std::enable_if_t< detail::spmbv_uses_mkl_v<SpMatType, ALPHAT, BETAT> >
  gespmbv( int64_t K, ALPHAT ALPHA, const SpMatType& A,
    const typename SpMatType::value_type* V,  int64_t LDV,  BETAT BETA,
          typename SpMatType::value_type* AV, int64_t LDAV ) {


  //std::cout << "IN MKL" << std::endl;
  sparse_status_t stat;

  using value_type = typename SpMatType::value_type;
  static_assert( std::is_same_v<value_type,double>, 
                 "MKL SPMBV is only implemented for double precision" );

  value_type alpha = ALPHA;
  value_type beta  = BETA;

  stat = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle(),
    A.descr(), SPARSE_LAYOUT_COLUMN_MAJOR, V, K, LDV, beta, AV, LDAV );

  if( stat != SPARSE_STATUS_SUCCESS ) 
    throw sparsexx::detail::mkl::mkl_sparse_exception(stat);

}


}

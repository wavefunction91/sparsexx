#pragma once
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>

namespace sparsexx {

template <typename SpMatType, 
  typename = detail::enable_if_csr_matrix_t<SpMatType>
> SpMatType extract_upper_triangle( const SpMatType& A ) {

  const auto M = A.m();
  const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  // Determine NNZ and row counts in the upper triangle
  size_t nnz_ut = 0;
  std::vector<typename SpMatType::index_type> rowptr_ut( M + 1 );
  rowptr_ut[0] = indexing;
  for( int64_t i = 0; i < M; ++i ) {
    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
  
    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;

    auto nnz_row = std::count_if( Aci_st, Aci_en, 
      [&](const auto j){ return (j-indexing) >= i; } );

    nnz_ut += nnz_row;
    rowptr_ut[i+1] = rowptr_ut[i] + nnz_row;
  }


  // Extract the Upper triangle
  SpMatType U( M, N, nnz_ut, indexing );
  //std::copy( rowptr_ut.begin(), rowptr_ut.end(), U.rowptr().begin() );
  U.rowptr() = std::move( rowptr_ut );
  auto* Unz = U.nzval().data();
  auto* Urp = U.rowptr().data();
  auto* Uci = U.colind().data();
  for( int64_t i = 0; i < M; ++i ) {
    const auto Aj_st  = Arp[i]   - indexing;
    const auto Aj_en  = Arp[i+1] - indexing;
    const auto Uj_st  = Urp[i]   - indexing;
  
    const auto* Aci_st = Aci + Aj_st;
    const auto* Aci_en = Aci + Aj_en;
          auto* Uci_st = Uci + Uj_st;

    const auto* Anz_st = Anz + Aj_st;
    const auto* Anz_en = Anz + Aj_en;
          auto* Unz_st = Unz + Uj_st;
  
    const auto* Aci_ut_st = std::find_if( Aci_st, Aci_en,
      [&](const auto x){ return (x-indexing) >= i; } );
    const auto ioff = std::distance( Aci_st, Aci_ut_st );
    const auto* Anz_ut_st = Anz_st + ioff;

    std::copy( Aci_ut_st, Aci_en, Uci_st );
    std::copy( Anz_ut_st, Anz_en, Unz_st );
  }

  return U;
}

}

#pragma once

#include "csr_matrix.hpp"
#include <stdexcept>
namespace sparsexx {

template <typename... Args>
void convert_to_dense( const csr_matrix<Args...>& A, 
  typename csr_matrix<Args...>::value_type* A_dense, int64_t LDAD ) {

  const int64_t M = A.m();
  const int64_t N = A.n();

  if( M > LDAD ) throw std::runtime_error("M > LDAD");

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto  indexing = A.indexing();

  for( int64_t i = 0; i < M; ++i ) {
    const auto j_st  = Arp[i]   - indexing;
    const auto j_en  = Arp[i+1] - indexing;
    const auto j_ext = j_en - j_st;

    auto* Ad_i = A_dense + i - indexing*LDAD;

    const auto* Anz_st = Anz + j_st;
    const auto* Aci_st = Aci + j_st;
    for( int64_t j = 0; j < j_ext; ++j )
      Ad_i[ Aci_st[j]*LDAD ] = Anz_st[j];
  }

}

}

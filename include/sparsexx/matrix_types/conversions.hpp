#pragma once

#include "csr_matrix.hpp"
#include "coo_matrix.hpp"

#include "coo_matrix_ops.hpp"

namespace sparsexx {

template <typename T, typename index_t, typename Alloc>
csr_matrix<T,index_t,Alloc>::csr_matrix( const coo_matrix<T,index_t,Alloc>& other ) :
  csr_matrix( other.m(), other.n(), other.nnz(), other.indexing() ) {

  if( not other.is_sorted_by_row_index() ) {
    throw 
      std::runtime_error("COO -> CSR Conversion Requires COO To Be Row Sorted");
  }

  // Seed rowptr
  rowptr_[0] = other.indexing();

  const auto& rowind_coo = other.rowind();
  const auto& colind_coo = other.colind();
  const auto& nzval_coo  = other.nzval();

  auto colind_it = colind_.begin();
  auto nzval_it  = nzval_.begin();
  for( index_type i = 0; i < m_; ++i ) {

    auto row_begin = 
      std::find( rowind_coo.begin(), rowind_coo.end(), i + indexing_ );
    auto row_end = std::find( row_begin, rowind_coo.end(), i + 1 + indexing_ );
     
    auto begin_idx = std::distance( rowind_coo.begin(), row_begin );
    auto end_idx   = std::distance( rowind_coo.begin(), row_end   );

    // Calculate next rowptr
    rowptr_[i + 1] = rowptr_[i] + end_idx - begin_idx;

    for( index_type j = begin_idx; j < end_idx; ++j ) {
     *(colind_it++) = colind_coo[j];
     *(nzval_it++ ) = nzval_coo[j];
    }
    
  }

}


}

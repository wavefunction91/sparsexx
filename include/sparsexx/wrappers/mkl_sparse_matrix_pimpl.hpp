#pragma once

#include "mkl_types.hpp"
#include "mkl_exceptions.hpp"

#include <cassert>
#include <iostream>
namespace sparsexx::detail::mkl {

struct sparse_wrapper_pimpl {

  struct matrix_descr descr;
  sparse_matrix_t     mat;

  ~sparse_wrapper_pimpl() noexcept {
    mkl_sparse_destroy( mat );
  }

  void optimize() { 
    mkl_sparse_optimize( mat );
  }
};

template <typename T>
enable_if_mkl_type_supported_t<T,std::unique_ptr<sparse_wrapper_pimpl>>
  mkl_csr_factory( int64_t indexing, int64_t m, int64_t n, 
    int_type* rowptr, int_type* colind, 
    T* nzval ) {

  assert( indexing == 0 or indexing == 1 );
  auto mkl_indexing = 
    (indexing == 0) ? SPARSE_INDEX_BASE_ZERO : SPARSE_INDEX_BASE_ONE;

  auto pimpl = std::make_unique<sparse_wrapper_pimpl>();

  sparse_status_t stat;
  if constexpr ( std::is_same_v<T, float> )
    stat = mkl_sparse_s_create_csr( &pimpl->mat, mkl_indexing, m, n, rowptr,
      rowptr+1, colind, nzval );
  else if constexpr ( std::is_same_v<T, double> )
    stat = mkl_sparse_d_create_csr( &pimpl->mat, mkl_indexing, m, n, rowptr,
      rowptr+1, colind, nzval );
  else if constexpr ( std::is_same_v<T, std::complex<float>> )
    stat = mkl_sparse_c_create_csr( &pimpl->mat, mkl_indexing, m, n, rowptr,
      rowptr+1, colind, nzval );
  else if constexpr ( std::is_same_v<T, std::complex<double>> )
    stat = mkl_sparse_z_create_csr( &pimpl->mat, mkl_indexing, m, n, rowptr,
      rowptr+1, colind, nzval );

  if( stat != SPARSE_STATUS_SUCCESS ) throw mkl_sparse_exception(stat);

  pimpl->descr.type = SPARSE_MATRIX_TYPE_GENERAL; // TODO handle symmetric

  return pimpl;
}






template <typename SpMatType>
enable_if_mkl_compatible_t<SpMatType,std::unique_ptr<sparse_wrapper_pimpl>>
  sparse_matrix_factory( int64_t indexing, int64_t m, int64_t n, 
    int_type* ptr, int_type* ind, 
    typename SpMatType::value_type* nzval ) {

  if constexpr (is_csr_matrix_v<SpMatType>)
    return mkl_csr_factory( indexing, m, n, ptr, ind, nzval );
  else
    throw std::runtime_error("Unrecognized Sparse Matrix Type in MKL Factory");

}





}

#pragma once

#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <fstream>
#include <cassert>
#include <string>
#include <iostream>
#include <stdexcept>

namespace sparsexx {

/**
 *  @brief Write a sparse matrix to binary triplet file.
 *
 *  COO variant.
 *
 *  Binary Triplet Format:
 *    4/8 bytes                Number of Rows
 *    4/8 bytes                Number of Cols
 *    8 bytes                  Number of Non-zero elements (NNZ)
 *    NNZ * 4/8 bytes          Row indices of non-zero elements 
 *    NNZ * 4/8 bytes          Col indices of non-zero elements 
 *    NNZ * sizeof(T) bytes    Non-zero elements
 *
 *    4/8 for LP64 / ILP64 indexing, T is the storage type of the
 *    non-zero elements of the matrix. All types are deduced from
 *    the template parameter.
 *
 *    @tparam SpMatType Type of Sparse Matrix to write in triplet format
 *
 *    @param[in] A     Sparse matrix to write to disk.
 *    @param[in] fname File name of resulting file.
 *
 */
template <typename SpMatType>
detail::enable_if_coo_matrix_t<SpMatType>
  write_binary_triplet( const SpMatType& A, std::string fname ) {

  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ofstream f_out( fname, std::ios::binary );
  size_t nnz = A.nnz();;
  index_t m = A.m(), n = A.n();
  f_out.write( (char*)&m,   sizeof(index_t) );
  f_out.write( (char*)&n,   sizeof(index_t) );
  f_out.write( (char*)&nnz, sizeof(size_t)  ); 

  f_out.write( (char*) A.rowind().data(), nnz * sizeof(index_t) );
  f_out.write( (char*) A.colind().data(), nnz * sizeof(index_t) );
  f_out.write( (char*) A.nzval().data(),  nnz * sizeof(value_t) );

}






/**
 *  @brief Write a sparse matrix to binary triplet file.
 *
 *  CSR variant.
 *
 *  Binary Triplet Format:
 *    4/8 bytes                Number of Rows
 *    4/8 bytes                Number of Cols
 *    8 bytes                  Number of Non-zero elements (NNZ)
 *    NNZ * 4/8 bytes          Row indices of non-zero elements 
 *    NNZ * 4/8 bytes          Col indices of non-zero elements 
 *    NNZ * sizeof(T) bytes    Non-zero elements
 *
 *    4/8 for LP64 / ILP64 indexing, T is the storage type of the
 *    non-zero elements of the matrix. All types are deduced from
 *    the template parameter.
 *
 *    @tparam SpMatType Type of Sparse Matrix to write in triplet format
 *
 *    @param[in] A     Sparse matrix to write to disk.
 *    @param[in] fname File name of resulting file.
 *
 */

template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType>
  write_binary_triplet( const SpMatType& A, std::string fname ) {

  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ofstream f_out( fname, std::ios::binary );
  size_t nnz = A.nnz();;
  index_t m = A.m(), n = A.n();
  f_out.write( (char*)&m,   sizeof(index_t) );
  f_out.write( (char*)&n,   sizeof(index_t) );
  f_out.write( (char*)&nnz, sizeof(size_t)  ); 

  // Construct rowind
  std::vector<index_t> rowind(nnz);
  auto rowind_it = rowind.begin();
  for( size_t i = 0; i < m; ++i ) {
    const auto row_count = A.rowptr()[i+1] - A.rowptr()[i];
    rowind_it = std::fill_n( rowind_it, row_count, i + A.indexing() );
  }
  assert( rowind_it == rowind.end() );

  f_out.write( (char*) rowind.data(),     nnz * sizeof(index_t) );
  f_out.write( (char*) A.colind().data(), nnz * sizeof(index_t) );
  f_out.write( (char*) A.nzval().data(),  nnz * sizeof(value_t) );

}

}

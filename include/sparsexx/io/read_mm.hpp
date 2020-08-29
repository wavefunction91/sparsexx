#pragma once

#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <fstream>
#include <cassert>
#include <string>
#include <iostream>
#include <stdexcept>

namespace sparsexx {

namespace detail {

/**
 *  @brief Read a sparse matrix in Matrix Market (MTX) format in COO format
 *
 *  @tparam T       Storage type of sparse matrix
 *  @tparam index_t Indexing type
 *  @tparam Alloc   Allocator type for the sparse matrix
 *
 *  @param[in] fname Filename for MTX file
 *
 *  @returns COO matrix reflecting the contents of the input file.
 *
 */
template <typename T, typename index_t, typename Alloc>
coo_matrix<T,index_t,Alloc> read_mm_as_coo( std::string fname ) {

  std::ifstream f_in(fname);

  std::string line;

  int64_t m, n, nnz;
  bool is_sym = false;
  {
    std::getline( f_in, line );
    auto tokens = tokenize( line );

    // Check if this is actually a MM file...

    if( tokens[0].compare("%%MatrixMarket") or tokens.size() != 5)
      throw std::runtime_error(fname + " is not a MM file");

    is_sym = !tokens[4].compare("symmetric");
    
    while(std::getline( f_in, line )) {
      if( line[0] != '%' ) break;
    }

    //std::getline( f_in, line );
    tokens = tokenize( line );
    if( tokens.size() != 3 )
      throw std::runtime_error(fname + 
            " contains an invalid spec for problem dimension");

    m   = std::stoll(tokens[0]);
    n   = std::stoll(tokens[1]);
    nnz = std::stoll(tokens[2]);

    if( is_sym and m != n )
      throw std::runtime_error( fname + " symmetric not compatible with M!=N" );

    if( is_sym ) nnz = 2*nnz - n;
  }

  coo_matrix<T,index_t,Alloc> A(m, n, nnz);
  auto& rowind = A.rowind();
  auto& colind = A.colind();
  auto& nzval  = A.nzval();

  size_t nnz_idx = 0;
  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    int64_t i = std::stoll( tokens[0] );
    int64_t j = std::stoll( tokens[1] );
    T       v = std::stod(  tokens[2] );

    rowind[nnz_idx] = i;
    colind[nnz_idx] = j;
    nzval[nnz_idx]  = v;
    nnz_idx++;

    if( is_sym and i != j ) {
      rowind[nnz_idx] = j;
      colind[nnz_idx] = i;
      nzval[nnz_idx]  = v;
      nnz_idx++;
    }

  }

  assert( nnz == nnz_idx );

  A.determine_indexing_from_adj();
  A.sort_by_row_index();

  assert( A.is_sorted_by_row_index() );

  return A;
}

}




/**
 *  @brief Read sparse matrix in Matrix Market (MTX) format
 *
 *  @tparam SpMatType Sparse matrix type to store the resulting matrix
 *
 *  @param[in] fname Filename for MTX file
 *
 *  @returns Sparse matrix reflecting the contents of the input file.
 *
 */
template <typename SpMatType>
SpMatType read_mm( std::string fname ) {

  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  if constexpr ( detail::is_coo_matrix_v<SpMatType> )
    return detail::read_mm_as_coo<value_t,index_t,allocator_t>( fname );
  else
    return SpMatType( detail::read_mm_as_coo<value_t,index_t,allocator_t>( fname ) );

}













}

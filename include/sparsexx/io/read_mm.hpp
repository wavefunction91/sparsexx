#pragma once

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/util/string.hpp>
#include <fstream>
#include <cassert>
#include <string>
#include <iostream>
#include <stdexcept>

namespace sparsexx {

template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
csr_matrix<T,index_t,Alloc> read_mm( std::string fname ) {



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
      throw std::runtime_error(fname + " contains an invalid spec for problem dimension");

    m   = std::stoll(tokens[0]);
    n   = std::stoll(tokens[1]);
    nnz = std::stoll(tokens[2]);

    if( is_sym and m != n )
      throw std::runtime_error( fname + " symmetric not compatible with M!=N" );

    if( is_sym ) nnz = 2*nnz - n;
  }


  std::vector< std::tuple< int64_t, int64_t, T > > coo;
  coo.reserve( nnz );

  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    int64_t i = std::stoll( tokens[0] );
    int64_t j = std::stoll( tokens[1] );
    T       v = std::stod(  tokens[2] );

    coo.push_back({i, j, v});
    if( i != j and is_sym ) 
      coo.push_back({j, i, v});

  }

  assert( coo.size() == (size_t)nnz );



  // Sort based on row
  std::stable_sort( coo.begin(), coo.end(), 
    []( auto a, auto b ){ return std::get<0>(a) < std::get<0>(b); } 
  );

  // Determine if we're zero based
  bool zero_based = std::any_of( coo.begin(), coo.end(),
    [](auto x){ return std::get<0>(x)==0 or std::get<1>(x)==0; }
  );

  // Allocate matrix
  csr_matrix<T,index_t,Alloc> A(m,n,nnz, !zero_based);
  // Init rowptr accordingly
  A.rowptr()[0] = !zero_based;

  

  auto begin_row = coo.begin();
  auto* colind = A.colind().data();
  auto* nzval = A.nzval().data();
  for( int64_t i = 0; i < m; ++i ) {
  
    // Get start of next row
    auto next_row = std::find_if( begin_row, coo.end(), 
      [&](auto x){ return std::get<0>(x) == i + 1 + !zero_based; });

    // Sort within row
    std::stable_sort( begin_row, next_row,
      []( auto a, auto b ){ return std::get<1>(a) < std::get<1>(b); } 
    );
      
    // Calculate row pointer for subsequent iteration
    A.rowptr()[i + 1] = A.rowptr()[i] + std::distance( begin_row, next_row );

    for( auto it = begin_row; it != next_row; ++it ) {
      *(colind++) = std::get<1>(*it);
      *(nzval++)  = std::get<2>(*it);
    }

    begin_row = next_row;
  }

  assert( std::distance( A.colind().data(), colind ) == nnz );
  assert( std::distance( A.nzval().data(), nzval ) == nnz );

  return A;
} // read_mm

}

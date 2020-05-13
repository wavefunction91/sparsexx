#pragma once

#include "coo_matrix.hpp"
#include <range/v3/all.hpp>
#include <iostream>

namespace sparsexx {

template <typename T, typename index_t, typename Alloc>
void coo_matrix<T,index_t,Alloc>::sort_by_row_index() {

  auto coo_zip = ranges::views::zip( rowind_, colind_, nzval_ );

  // Sort lex by row index
  using coo_el = std::tuple<index_type, index_type, value_type>;
  ranges::sort( coo_zip, []( const coo_el& el1, const coo_el& el2 ) {
    const auto i1 = std::get<0>(el1);
    const auto i2 = std::get<0>(el2);
    const auto j1 = std::get<1>(el1);
    const auto j2 = std::get<1>(el2);

    if( i1 < i2 )      return true;
    else if( i1 > i2 ) return false;
    else               return j1 < j2;
  });


}

template <typename T, typename index_t, typename Alloc>
void coo_matrix<T,index_t,Alloc>::expand_from_triangle() {

  std::cout << "Expanding Triangle" << std::endl;
  auto idx_zip = ranges::views::zip( rowind_, colind_ );


  auto lt_check = []( const std::tuple<index_type,index_type>& p ) {
    return std::get<0>(p) <= std::get<1>(p);
  };
  auto ut_check = []( const std::tuple<index_type,index_type>& p ) {
    return std::get<0>(p) >= std::get<1>(p);
  };

  bool lower_triangle = ranges::all_of( idx_zip, lt_check );
  bool upper_triangle = ranges::all_of( idx_zip, ut_check );
  bool diagonal       = lower_triangle and upper_triangle;
  bool full_matrix    = (not lower_triangle) and (not upper_triangle);
 
  std::cout << std::boolalpha;
  std::cout << "LT " << lower_triangle << std::endl;
  std::cout << "UT " << upper_triangle << std::endl;
  if( diagonal or full_matrix ) return;

  std::cout << "Performing Expansion..." << std::endl;
  size_t new_nnz = 2*nnz_ - n_;
  rowind_.reserve( new_nnz );
  colind_.reserve( new_nnz );
  nzval_.reserve(  new_nnz );

  for( size_t i = 0; i < nnz_; ++i )    
  if( rowind_[i] != colind_[i] ) {
    rowind_.emplace_back( colind_[i] );
    colind_.emplace_back( rowind_[i] );
    nzval_ .emplace_back( nzval_[i]  );
  }

  assert( rowind_.size() == new_nnz );
  assert( colind_.size() == new_nnz );
  assert( nzval_.size()  == new_nnz );

  nnz_ = new_nnz;
}



}

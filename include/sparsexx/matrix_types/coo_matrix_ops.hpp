#pragma once

#include "coo_matrix.hpp"
#include <range/v3/all.hpp>

namespace sparsexx {

template <typename T, typename index_t, typename Alloc>
void coo_matrix<T,index_t,Alloc>::sort_by_row_index() {

  auto coo_zip = ranges::views::zip( rowind_, colind_, nzval_ );

  // Sort by row index
  using coo_el = std::tuple<index_type, index_type, value_type>;
  ranges::sort( coo_zip, []( const coo_el& el1, const coo_el& el2 ) {
    return std::get<0>(el1) < std::get<0>(el2);
  });

  auto begin_row = rowind_.begin();
  for( index_type i = 0; i < m_; ++i ) {

    // Get start of next row
    auto next_row = std::find( begin_row, rowind_.end(), i+1+indexing_ );

    auto begin_idx = std::distance( rowind_.begin(), begin_row );
    auto end_idx   = std::distance( rowind_.begin(), next_row  );

    // Sort the row internally
    ranges::sort( coo_zip | ranges::views::slice(begin_idx, end_idx),
      []( const coo_el& el1, const coo_el& el2) {
        return std::get<1>(el1) < std::get<1>(el2);
      });

    begin_row = next_row;

  }

}

}

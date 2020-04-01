#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace sparsexx {

template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
class csr_matrix {

public:

  using value_type     = T;
  using index_type     = index_t;
  using size_type      = int64_t;
  using allocator_type = Alloc;

protected:

  using alloc_traits = typename std::allocator_traits<Alloc>;

  template <typename U>
  using rebind_alloc = typename alloc_traits::template rebind_alloc<U>;

  template <typename U>
  using internal_storage = typename std::vector< U, rebind_alloc<U> >;

  size_type m_;
  size_type n_;
  size_type nnz_;
  size_type indexing_;

  internal_storage< T >       nzval_;
  internal_storage< index_t > colind_;
  internal_storage< index_t > rowptr_;

public:

  csr_matrix( size_type m, size_type n, size_type nnz,
    size_type indexing = 1) :
    m_(m), n_(n), nnz_(nnz), nzval_(nnz), colind_(nnz),
    rowptr_(m+1), indexing_(indexing) { }

  size_type m()   const { return m_; };
  size_type n()   const { return n_; };
  size_type nnz() const { return nnz_; };

  size_type indexing() const { return indexing_; }

  auto& nzval()  { return nzval_; };
  auto& colind() { return colind_; };
  auto& rowptr() { return rowptr_; };

  const auto& nzval () const { return nzval_; };
  const auto& colind() const { return colind_; };
  const auto& rowptr() const { return rowptr_; };

#if 0
  template <typename... Args>
  csr_matrix( const csr_matrix< Args... >& other ) :
    m_( other.m() ), n_( other.n() ), nnz_( other.nnz() ) {
 
    rowptr_.resize( m_ + 1 );
    colind_.resize( nnz_ );
    nzval_.resize( nnz_ );

    std::copy( other.rowptr().begin(), other.rowptr.end(), rowptr_.begin() );
    std::copy( other.rowptr().begin(), other.rowptr.end(), rowptr_.begin() );

  }
#endif
};

}

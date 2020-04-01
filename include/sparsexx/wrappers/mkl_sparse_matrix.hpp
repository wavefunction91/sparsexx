#pragma once

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <memory>

#include "mkl_type_traits.hpp"
#include "mkl_sparse_matrix_pimpl.hpp"

namespace sparsexx {

template <class SpMatType,
  typename = detail::mkl::enable_if_mkl_compatible_t<SpMatType> >
class mkl_sparse_wrapper : public SpMatType {

public:

  using value_type = typename SpMatType::value_type;
  using index_type = typename SpMatType::index_type;
  using size_type  = typename SpMatType::size_type;
  using allocator_type = typename SpMatType::allocator_type;

protected:

  using alloc_traits = typename SpMatType::alloc_traits;

  template <typename U>
  using rebind_alloc = typename SpMatType::template rebind_alloc<U>;

  template <typename U>
  using internal_storage = typename SpMatType::template internal_storage<U>;

  using pimpl_type = detail::mkl::sparse_wrapper_pimpl;

  std::unique_ptr< pimpl_type > pimpl_;

public:

  // Forward arguments to SpMatType base class
  template <typename... Args>
  mkl_sparse_wrapper( Args&&... args ) :
    SpMatType( std::forward<Args>(args)... ),
    pimpl_(detail::mkl::sparse_matrix_factory<SpMatType>(
      this->indexing(), this->m(), this->n(), this->rowptr().data(),
      this->colind().data(), this->nzval().data() )){ }


  auto&       handle()       { return pimpl_->mat; }
  const auto& handle() const { return pimpl_->mat; }

  auto&       descr()        { return pimpl_->descr; }
  const auto& descr()  const { return pimpl_->descr; }

  void optimize() { pimpl_->optimize(); }
};


// Useful typedefs
template <typename... Args>
using mkl_csr_matrix = mkl_sparse_wrapper<csr_matrix<Args...>>;


namespace detail::mkl {

template <typename SpMatType>
struct is_mkl_sparse_matrix : public std::false_type {};

template <typename SpMatType>
struct is_mkl_sparse_matrix< mkl_sparse_wrapper<SpMatType> > :
  public std::true_type {};

template <typename SpMatType>
inline constexpr bool is_mkl_sparse_matrix_v = 
  is_mkl_sparse_matrix<SpMatType>::value;

template <typename SpMatType, typename U = void>
struct enable_if_mkl_sparse_matrix {
  using type = std::enable_if_t< is_mkl_sparse_matrix_v<SpMatType>, U>;
};

template <typename SpMatType, typename U = void>
using enable_if_mkl_sparse_matrix_t = typename
  enable_if_mkl_sparse_matrix<SpMatType, U>::type;

}

}

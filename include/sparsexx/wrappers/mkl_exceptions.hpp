#pragma once

#include "mkl_types.hpp"
#include <stdexcept>
#include <cassert>

namespace sparsexx::detail::mkl {

const char*  get_mkl_error_string( sparse_status_t s ) {

  switch( s ) {
    case SPARSE_STATUS_SUCCESS:
    return "The operation was successful.";
    
    case SPARSE_STATUS_NOT_INITIALIZED:
    return "The routine encountered an empty handle or matrix array.";
    
    case SPARSE_STATUS_ALLOC_FAILED:
    return "Internal memory allocation failed.";
    
    case SPARSE_STATUS_INVALID_VALUE:
    return "The input parameters contain an invalid value.";
    
    case SPARSE_STATUS_EXECUTION_FAILED:
    return "Execution failed.";
    
    case SPARSE_STATUS_INTERNAL_ERROR:
    return "An error in algorithm implementation occurred.";
    
    case SPARSE_STATUS_NOT_SUPPORTED:
    return "NOT SUPPORTED";

    default:
    return "UNKNOWN";
  }
}


class mkl_sparse_exception : public std::exception {

  sparse_status_t stat_;

  const char* what() const throw() {
    return get_mkl_error_string( stat_ );
  }

public:

  mkl_sparse_exception() = delete;
  mkl_sparse_exception( sparse_status_t s ):
    stat_(s) { }

};


}

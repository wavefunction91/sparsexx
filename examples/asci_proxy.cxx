#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/coo_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/io/read_binary_triplets.hpp>
#include <sparsexx/io/write_binary_triplets.hpp>

#include <sparsexx/util/submatrix.hpp>
#include <sparsexx/util/reorder.hpp>

#include <iostream>
#include <iterator>
#include <iomanip>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>

int main( int argc, char** argv ) {

  MPI_Init( &argc, &argv );
  auto world_size = sparsexx::detail::get_mpi_size( MPI_COMM_WORLD );
  auto world_rank = sparsexx::detail::get_mpi_rank( MPI_COMM_WORLD );
  {

  assert( argc == 3 );
  using spmat_type = sparsexx::coo_matrix<double, int32_t>;

  auto A = sparsexx::read_binary_triplet<spmat_type>( std::string( argv[1] ) );

  }
  MPI_Finalize();
  return 0;
}

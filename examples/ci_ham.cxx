#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/coo_matrix.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>


#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>

template <typename Op>
double time_op( const Op& op ) {
  MPI_Barrier( MPI_COMM_WORLD);
  auto st = std::chrono::high_resolution_clock::now();

  op();

  MPI_Barrier( MPI_COMM_WORLD);
  auto en = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double,std::milli>(en - st).count();
}


inline auto popcnt( uint64_t a ) {
   return __builtin_popcountll(a) ;
}




template <typename... Args>
auto dets_to_coo_matrix_double_loop( 
  size_t M, size_t N, const uint64_t* ialpha, const uint64_t* ibeta, 
  const uint64_t* jalpha, const uint64_t* jbeta 
) {

  using index_t = typename sparsexx::coo_matrix<Args...>::index_type;
  
  std::vector<index_t> is, js;
  for(size_t i = 0; i < M; ++i )
  for(size_t j = 0; j < N; ++j ) {
    auto hamming_dist =  popcnt(ialpha[i] ^ jalpha[j]);
         hamming_dist += popcnt(ibeta[i]  ^ jbeta[j] );
    if( hamming_dist <= 4 ) {
      is.emplace_back( i );
      js.emplace_back( j );
    }
  }

  auto nnz = is.size();
  sparsexx::coo_matrix<Args...> H( N, N, nnz );
  H.rowind() = std::move( is );
  H.colind() = std::move( js );

  return H;
}

template <typename SpMatType, typename... Args>
SpMatType dets_to_sparse_matrix( Args&&... args ) {

  using index_t = typename SpMatType::index_type;
  using value_t = typename SpMatType::value_type;
  using alloc_t = typename SpMatType::allocator_type;

  auto H_coo = dets_to_coo_matrix_double_loop<value_t,index_t,alloc_t>( 
    std::forward<Args>(args)... );

  if constexpr ( sparsexx::detail::is_coo_matrix_v<SpMatType> )
    return H_coo;
  else
    return SpMatType(H_coo);

}







int main( int argc, char** argv ) {

  using coo_matrix      = sparsexx::coo_matrix<double, int32_t>;
  using csr_matrix      = sparsexx::csr_matrix<double, int32_t>;
  using spmat_type      = coo_matrix;
  using dist_spmat_type = sparsexx::dist_sparse_matrix<spmat_type>;

  MPI_Init( &argc, &argv );
  auto world_size = sparsexx::detail::get_mpi_size( MPI_COMM_WORLD );
  auto world_rank = sparsexx::detail::get_mpi_rank( MPI_COMM_WORLD );


  assert( argc == 2 );
  std::string det_file_name = argv[1];
  {

  // Read in Det file
  std::ifstream det_file( det_file_name );

  std::vector<uint64_t> det_strs;

  //int dummy; det_file >> dummy >> dummy >> dummy >> dummy;
  
  std::string line;
  std::getline( det_file, line );
  std::getline( det_file, line );
  
  while( std::getline( det_file, line ) ){
    uint64_t a = std::stoull( line );
    det_strs.emplace_back(a);
  }

  assert( det_strs.size() % 2 == 0 );
  size_t ndets      = det_strs.size() / 2;
  size_t ndets_keep = std::min( ndets, 50000ul );

  if( !world_rank ) {
    std::cout << "READ IN " << ndets << " DETERMINANT STRINGS" << std::endl;
    std::cout << "KEEPING " << ndets_keep << std::endl;
  }

  std::vector<uint64_t> alpha_strs( ndets_keep ), beta_strs( ndets_keep );
  for( size_t i = 0; i < ndets_keep; ++i ) {
    alpha_strs[i] = det_strs[ 2*i + 0 ];
    beta_strs[i]  = det_strs[ 2*i + 1 ];
  }

  int32_t N = ndets_keep;

  // Serial formation
  spmat_type H_replicated;
  auto serial_formation_dur = time_op([&]() {

    H_replicated = dets_to_sparse_matrix<spmat_type>(
      N, N, alpha_strs.data(), beta_strs.data(), alpha_strs.data(),
      beta_strs.data()
    );
    
    std::fill( H_replicated.nzval().begin(),
               H_replicated.nzval().end(), 1 );
  } );

  if( !world_rank )
    std::cout << "Hamiltonian has NNZ = " << H_replicated.nnz() << std::endl;


  // Distributed formation
  dist_spmat_type H_dist;
  auto dist_formation_dur = time_op([&]() {
    
    int32_t nrow_per_rank = N / world_size;
    std::vector<int32_t> row_tiling(world_size + 1);
    for( auto i = 0; i < world_size; ++i )
      row_tiling[i] = i * nrow_per_rank;
    row_tiling.back() = N;

    std::vector<int32_t> col_tiling = { 0, N };

    H_dist = dist_spmat_type( MPI_COMM_WORLD, N, N, row_tiling, col_tiling);

    for( auto& [tile_index, local_tile] : H_dist ) {

      auto [row_st, row_end] = local_tile.global_row_extent;
      auto [col_st, col_end] = local_tile.global_col_extent;

      auto m_local = row_end - row_st; 
      auto n_local = col_end - col_st; 
      
      std::cout << "m = " << m_local << " n = " << n_local << std::endl;

      const auto* ialpha = alpha_strs.data() + row_st;
      const auto* jalpha = alpha_strs.data() + col_st;
      const auto* ibeta  = beta_strs.data()  + row_st;
      const auto* jbeta  = beta_strs.data()  + col_st;

      local_tile.local_matrix = dets_to_sparse_matrix<spmat_type>(
        m_local, n_local, ialpha, ibeta, jalpha, jbeta );

    }

  });

  if( !world_rank ) {
    std::cout << "Serial Dur = " << serial_formation_dur << std::endl;
    std::cout << "Dist   Dur = " << dist_formation_dur   << std::endl;
  }

  }
  MPI_Finalize();

}

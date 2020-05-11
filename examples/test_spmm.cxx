#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/io/read_rb.hpp>
#include <sparsexx/io/read_mm.hpp>


#include <iomanip>
#include <random>
#include <algorithm>
#include <chrono>

int main( int argc, char** argv ) {

  std::cout << std::scientific << std::setprecision(5);

  assert( argc == 2 );
  std::shared_ptr<sparsexx::mkl_csr_matrix<double,MKL_INT>> A_mkl_ptr = nullptr;
  {
  auto read_st = std::chrono::high_resolution_clock::now();
  auto A = sparsexx::read_mm<double,MKL_INT>( std::string( argv[1] ) );
  auto read_en = std::chrono::high_resolution_clock::now();

  auto copy_st = std::chrono::high_resolution_clock::now();
  A_mkl_ptr = std::make_shared<sparsexx::mkl_csr_matrix<double,MKL_INT>>( A );
  auto copy_en = std::chrono::high_resolution_clock::now();

  auto read_dur = 
    std::chrono::duration<double,std::milli>( read_en-read_st ).count();
  auto copy_dur = 
    std::chrono::duration<double,std::milli>( copy_en-copy_st ).count();

  std::cout << "Read Duration = " << read_dur << " ms" << std::endl;
  std::cout << "Copy Duration = " << copy_dur << " ms" << std::endl;
  }

  auto& A_mkl = *A_mkl_ptr;
  const MKL_INT N = A_mkl.m();
  const MKL_INT K = 10;
  const MKL_INT NREP = 10;

  mkl_sparse_set_mm_hint( A_mkl.handle(), SPARSE_OPERATION_NON_TRANSPOSE, 
    A_mkl.descr(),SPARSE_LAYOUT_COLUMN_MAJOR, K, NREP ); 

  auto opt_st = std::chrono::high_resolution_clock::now();
  //A_mkl.optimize();
  auto opt_en = std::chrono::high_resolution_clock::now();
  auto opt_dur = std::chrono::duration<double,std::milli>(opt_en-opt_st).count();

  std::cout << "Opt Duration  = " << opt_dur << " ms" << std::endl;

  std::vector<double> V( N*K ), AV( N*K );

  std::vector<double> mm_times;
  for( auto i = 0; i < NREP; ++i ) {
    auto mm_st = std::chrono::high_resolution_clock::now();
    sparsexx::spblas::gespmbv( K, 1., A_mkl, V.data(), N, 0., AV.data(), N );
    auto mm_en = std::chrono::high_resolution_clock::now();

    mm_times.emplace_back(
      std::chrono::duration<double,std::milli>(mm_en-mm_st).count()
    );

    std::copy( V.begin(), V.end(), AV.begin() );
  }

  auto mm_dur = std::accumulate( mm_times.begin(), mm_times.end(), 0. );
  auto mm_avg = mm_dur / mm_times.size();

  auto mm_std = std::accumulate( mm_times.begin(), mm_times.end(), 0.,
    [&](auto a, auto b) {
      auto x = b - mm_avg;
      return a + x*x;
    } );
  mm_std = std::sqrt( mm_std / ( mm_times.size() - 1 ) );

  std::cout << "MM Duration   = " << mm_dur << " ms" << std::endl;
  std::cout << "MM Average    = " << mm_avg << " ms" << std::endl;
  std::cout << "MM StdDev     = " << mm_std << " ms" << std::endl;

  return 0;
}

cmake_minimum_required( VERSION 3.11 FATAL_ERROR )
project( sparsexx VERSION 0.1.0 LANGUAGES C CXX )

# Append local cmake directory to find CMAKE Modules
if( CMAKE_MODULE_PATH )
  list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
else()
  set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
endif()
list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/linalg-modules")


find_package( IntelMKL REQUIRED COMPONENTS lp64 )


add_executable( sparsexx_test test_ls.cxx )
target_compile_features( sparsexx_test PUBLIC cxx_std_17 )
target_include_directories( sparsexx_test PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_link_libraries( sparsexx_test  PUBLIC IntelMKL::mkl )

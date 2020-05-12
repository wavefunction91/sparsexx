cmake_minimum_required( VERSION 3.14 FATAL_ERROR )
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
find_package( MPI      REQUIRED                 )
find_package( OpenMP   REQUIRED                 )


#cmake_policy( SET CMP0079 NEW )
cmake_policy( VERSION 3.14 )
include( FetchContent )

FetchContent_Declare(
  metis_download
  URL http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
  #GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
)

FetchContent_Declare(
  range_v3_download
  GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
)

if(NOT metis_download_POPULATED)
  FetchContent_Populate(metis_download)
  set( GKLIB_PATH ${metis_download_SOURCE_DIR}/GKlib CACHE PATH  "path to GKlib" )
  add_subdirectory(${metis_download_SOURCE_DIR} ${metis_download_BINARY_DIR})
  target_link_libraries( metis OpenMP::OpenMP_C )
  target_include_directories( metis PUBLIC ${metis_download_SOURCE_DIR}/include )
endif()


if(NOT range_v2_download_POPULATED)
  FetchContent_Populate( range_v3_download )
  add_subdirectory(${range_v3_download_SOURCE_DIR} ${range_v3_download_BINARY_DIR})
endif()


add_library               ( sparsexx INTERFACE                               )
target_compile_features   ( sparsexx INTERFACE cxx_std_17                    )
target_link_libraries     ( sparsexx INTERFACE IntelMKL::mkl MPI::MPI_CXX OpenMP::OpenMP_CXX metis range-v3 )
target_include_directories( sparsexx INTERFACE ${PROJECT_SOURCE_DIR}/include )

target_compile_options( sparsexx INTERFACE "-Wall" "-Wextra" "-Wpedantic" "-Wnon-virtual-dtor")

add_subdirectory( examples )


# sparsexx Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( sparsexx VERSION 0.1.0 LANGUAGES C CXX )

option( SPARSEXX_ENABLE_TESTS "Enable Testing" ON )


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include( FetchContent )
  FetchContent_Declare( linalg-cmake-modules
    GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
    GIT_TAG        main
  )
  FetchContent_GetProperties( linalg-cmake-modules )
  if( NOT linalg-cmake-modules_POPULATED )
    FetchContent_Populate( linalg-cmake-modules )
    list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
  endif()
endif()


option( SPARSEXX_ENABLE_MKL       "Enable MKL Bindings"             OFF )
option( SPARSEXX_ENABLE_CEREAL    "Enable serialization via cereal" OFF )
option( SPARSEXX_ENABLE_RANGES_V3 "Enable Ranges-v3"                OFF )
option( SPARSEXX_ENABLE_METIS     "Enable METIS"                    OFF )
option( SPARSEXX_ENABLE_OPENMP    "Enable OpenMP"                   ON  )
option( SPARSEXX_ENABLE_MPI       "Enable MPI"                      OFF )

if(SPARSEXX_ENABLE_MPI)
find_package( MPI   REQUIRED )
endif()
if(SPARSEXX_ENABLE_OPENMP)
find_package( OpenMP         )
endif()

if( SPARSEXX_ENABLE_MKL )
  find_package( IntelMKL COMPONENTS lp64 REQUIRED )
  if( TARGET IntelMKL::IntelMKL )
    set( SPARSEXX_EXTERNAL_LIBRARIES IntelMKL::IntelMKL )
  endif()
endif()





#cmake_policy( SET CMP0079 NEW )
cmake_policy( VERSION 3.18 )
include( FetchContent )





# METIS 
if(SPARSEXX_ENABLE_METIS)
  FetchContent_Declare(
    metis_download
    #URL http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    URL https://src.fedoraproject.org/lookaside/pkgs/metis/metis-5.1.0.tar.gz/5465e67079419a69e0116de24fce58fe/metis-5.1.0.tar.gz
    #GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
  )
  
  FetchContent_GetProperties(metis_download)
  if(NOT metis_download_POPULATED)
    FetchContent_Populate(metis_download)
    set( GKLIB_PATH ${metis_download_SOURCE_DIR}/GKlib CACHE PATH  "path to GKlib" )
    add_subdirectory(${metis_download_SOURCE_DIR} ${metis_download_BINARY_DIR})
    if( TARGET OpenMP::OpenMP_C )
      target_link_libraries( metis OpenMP::OpenMP_C )
    endif()
    target_include_directories( metis PUBLIC ${metis_download_SOURCE_DIR}/include )
  endif()
  list(APPEND SPARSEXX_EXTERNAL_LIBRARIES metis)
endif()



if( SPARSEXX_ENABLE_RANGES_V3 )
  # RANGES-V3
  FetchContent_Declare(
    range_v3_download
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  )
  FetchContent_GetProperties(range_v3_download)
  if(NOT range_v3_download_POPULATED)
    FetchContent_Populate( range_v3_download )
    add_subdirectory(${range_v3_download_SOURCE_DIR} ${range_v3_download_BINARY_DIR})
    add_library( ranges INTERFACE IMPORTED )
    target_link_libraries( ranges INTERFACE range-v3 )
    #target_compile_definitions( ranges INTERFACE "SPARSEXX_ENABLE_RANGES_V3=1" )
  endif()
  list(APPEND SPARSEXX_EXTERNAL_LIBRARIES ranges)
endif()




# CEREAL
if( SPARSEXX_ENABLE_CEREAL )
  FetchContent_Declare(
    cereal_download
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  )
  FetchContent_GetProperties(cereal_download)
  if(NOT cereal_download_POPULATED)
    FetchContent_Populate( cereal_download )
    add_library( cereal INTERFACE IMPORTED )
    set_target_properties( cereal PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${cereal_download_SOURCE_DIR}/include"
      INTERFACE_COMPILE_DEFINITIONS "CEREAL_THREAD_SAFE=1;SPARSEXX_ENABLE_CEREAL=1"
    )
  endif()
  list(APPEND SPARSEXX_EXTERNAL_LIBRARIES cereal)
endif()


add_library( sparsexx src/metis_wrapper.cxx )
target_compile_features( sparsexx PUBLIC cxx_std_17 )
if( TARGET MPI::MPI_CXX )
  target_link_libraries( sparsexx PUBLIC MPI::MPI_CXX  )
endif()
if( TARGET OpenMP::OpenMP_CXX )
  target_link_libraries( sparsexx PUBLIC OpenMP::OpenMP_CXX )
endif()
target_link_libraries     ( sparsexx PUBLIC ${SPARSEXX_EXTERNAL_LIBRARIES} )
target_include_directories( sparsexx PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

configure_file( 
  ${PROJECT_SOURCE_DIR}/include/sparsexx/sparsexx_config.hpp.in
  ${PROJECT_BINARY_DIR}/include/sparsexx/sparsexx_config.hpp
)

# Installation
include( GNUInstallDirs )
add_library(sparsexx::sparsexx ALIAS sparsexx)
install( TARGETS sparsexx 
  EXPORT sparsexx-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)



set_target_properties( sparsexx PROPERTIES EXPORT_NAME sparsexx )

# Export build tree
export(EXPORT sparsexx-targets
      NAMESPACE sparsexx::
      FILE "${PROJECT_BINARY_DIR}/sparsexx-targets.cmake")

# Install staic headers
install( 
  DIRECTORY   ${PROJECT_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING PATTERN "*.hpp"
)

# Install generated headers
install(
  FILES       ${PROJECT_BINARY_DIR}/include/sparsexx/sparsexx_config.hpp
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/sparsexx
)

   
# Export target to script
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/sparsexx )
install( EXPORT sparsexx-targets
  FILE         sparsexx-targets.cmake
  NAMESPACE    sparsexx::
  DESTINATION  ${INSTALL_CONFIGDIR}
)

# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/sparsexx-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)


# Setup sparsexx-config.cmake
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/sparsexx-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sparsexx-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install CMake files
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/sparsexx-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/sparsexx-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

add_subdirectory( examples )
include(CTest)
if( CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND SPARSEXX_ENABLE_TESTS AND BUILD_TESTING )
  add_subdirectory( tests )
endif()


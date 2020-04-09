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
add_library               ( sparsexx INTERFACE                               )
target_compile_features   ( sparsexx INTERFACE cxx_std_17                    )
target_link_libraries     ( sparsexx INTERFACE IntelMKL::mkl                 )
target_include_directories( sparsexx INTERFACE ${PROJECT_SOURCE_DIR}/include )



add_executable       ( sparsexx_ls_test test_ls.cxx     )
target_link_libraries( sparsexx_ls_test PUBLIC sparsexx )

add_executable       ( sparsexx_mkl_feast_test test_mkl_feast.cxx )
target_link_libraries( sparsexx_mkl_feast_test PUBLIC sparsexx    )

add_executable       ( sparsexx_submat_test test_submat.cxx     )
target_link_libraries( sparsexx_submat_test PUBLIC sparsexx )


#set( EVSL_DIR ${PROJECT_SOURCE_DIR}/EVSL/install_mic          )
#set( EVSL_LIBRARIES "${EVSL_DIR}/lib/libevsl.a;IntelMKL::mkl" )
#set( EVSL_INCLUDE_DIRECTORIES ${EVSL_DIR}/include             )
#add_library( evsl INTERFACE IMPORTED )
#set_target_properties( evsl 
#  PROPERTIES 
#  INTERFACE_LINK_LIBRARIES      "${EVSL_LIBRARIES}"
#  INTERFACE_INCLUDE_DIRECTORIES "${EVSL_INCLUDE_DIRECTORIES}"
#)
#
#add_executable       ( sparsexx_evsl_test test_evsl.cxx        )
#target_link_libraries( sparsexx_evsl_test PUBLIC evsl sparsexx ) 
#
#
#
#
#
#
#find_package( BLACS REQUIRED )
#set( FEAST_DIR ${PROJECT_SOURCE_DIR}/FEAST/4.0                           )
#set( FEAST_LIBRARIES "${FEAST_DIR}/lib/mic-knl/libpfeast.a;BLACS::BLACS" )
#set( FEAST_INCLUDE_DIRECTORIES ${FEAST_DIR}/include                      )
#add_library( feast INTERFACE IMPORTED )
#set_target_properties( feast 
#  PROPERTIES 
#  INTERFACE_LINK_LIBRARIES      "${FEAST_LIBRARIES}"
#  INTERFACE_INCLUDE_DIRECTORIES "${FEAST_INCLUDE_DIRECTORIES}"
#)
#
#add_executable       ( sparsexx_pfeast_test test_pfeast.cxx       )
#target_link_libraries( sparsexx_pfeast_test PUBLIC feast sparsexx )



set( lobpcgxx_DIR   "$ENV{HOME}/devel/lobpcgxx/install_lobpcgxx_xg/lib64/cmake/lobpcgxx"     )
set( blaspp_DIR     "$ENV{HOME}/devel/lobpcgxx/install_icl_xg/lib/blaspp" )
set( lapackpp_DIR   "$ENV{HOME}/devel/lobpcgxx/install_icl_xg/lib/lapackpp" )
find_package( lobpcgxx REQUIRED )

add_executable       ( sparsexx_lpcg_test test_lpcg.cxx                      )
target_link_libraries( sparsexx_lpcg_test PUBLIC lobpcgxx::lobpcgxx sparsexx ) 
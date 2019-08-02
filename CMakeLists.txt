# Basic cmake setup
cmake_minimum_required( VERSION 3.1 FATAL_ERROR )
project( sgdicp )


set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF) 
add_definitions(-DBOOST_UBLAS_NDEBUG) 
add_definitions("-std=c++11") 
add_definitions("-fopenmp")

add_definitions("-lgomp")

OPTION (USE_OpenMP "Use OpenMP" ON) 
find_package(OpenMP) 
if (OPENMP_FOUND) 
  message("OPENMP FOUND") 
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}") 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") 
endif() 

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -std=c++11 -fopenmp -fPIC")


# Compiler options
set( CMAKE_CXX_STANDARD 11 )

# Set output directories for libraries and executables
set( BASE_DIR ${CMAKE_SOURCE_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_DIR}/bin )

# Find libraries
find_package( Boost REQUIRED COMPONENTS program_options system )
find_package( Eigen3 REQUIRED )
find_package( PCL REQUIRED COMPONENTS common filters io )

include_directories(
    src
    ${PCL_INCLUDE_DIRS}
)
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# Add src folder
add_subdirectory(src)

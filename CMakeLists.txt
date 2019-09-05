cmake_minimum_required(VERSION 3.0)


set(CMAKE_BUILD_TYPE Release)

project(taton)

include_directories("${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src)


set(SRCS
  Test.cpp
  )

add_executable(Ex1 ${SRCS})


find_package(OpenMP)

target_link_libraries(Ex1 PUBLIC
  bayesregression
  -lmlpack
  -larmadillo
  -lboost_serialization
  -lboost_program_options
  OpenMP::OpenMP_CXX)  	

install (TARGETS Ex1 DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/Ex1Config.h"        
         DESTINATION include)

cmake_minimum_required(VERSION 3.0)

project(Ex1)

set(SRCS
    nbc_test.cpp	
    )

set(HEADERS
    )

# On indique que l'on veut l'executable gtlib en sortie compile a partir des SRCS et des HEADERS
add_executable(Ex1 ${SRCS} ${HEADERS})
	

find_package(OpenMP)

target_link_libraries(Ex1 PUBLIC -lmlpack
  -larmadillo
  -lboost_serialization
  -lboost_program_options
  OpenMP::OpenMP_CXX)  

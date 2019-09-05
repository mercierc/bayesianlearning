cmake_minimum_required(VERSION 3.0)



file(GLOB HEADER_FILES "*.hpp")

add_library(bayesregression bayesridge.cpp utils.cpp) 

install (TARGETS bayesregression DESTINATION bin)
install (FILES ${HEADER_FILE}  DESTINATION include)


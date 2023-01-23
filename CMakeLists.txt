# Project information
cmake_minimum_required(VERSION 3.15)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(NNFramework_test VERSION 1.0.0)

add_subdirectory(lib)

add_executable(NNFramework_test src/main.cpp)
target_include_directories(NNFramework_test PRIVATE inc)
target_link_libraries(NNFramework_test PUBLIC matplot)     # link matplotplusplus
target_link_libraries(NNFramework_test PUBLIC NNFramework) # link NNFramework library 